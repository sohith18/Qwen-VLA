import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], 'GPU')

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from policy_head_v3 import ActionDiT

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


# ==============================================================================
# NORMALIZER
# ==============================================================================

class ActionNormalizer:
    def __init__(self, min_val, max_val):
        self.min_val = torch.tensor(min_val).float()
        self.max_val = torch.tensor(max_val).float()

    def normalize(self, actions):
        device = actions.device
        min_v  = self.min_val.to(device)
        max_v  = self.max_val.to(device)
        norm   = actions.clone()
        norm[..., :-1] = 2 * ((actions[..., :-1] - min_v[:-1]) /
                               (max_v[:-1] - min_v[:-1] + 1e-8)) - 1
        norm[..., -1]  = 2 * (actions[..., -1] - 0.0) / (1.0 - 0.0) - 1
        norm[..., -1]  = torch.where(norm[..., -1] >= 0.0, 1.0, -1.0)
        return norm

    def unnormalize(self, actions):
        device = actions.device
        min_v  = self.min_val.to(device)
        max_v  = self.max_val.to(device)
        unnorm = actions.clone()
        unnorm[..., :-1] = ((actions[..., :-1] + 1) / 2) * (max_v[:-1] - min_v[:-1]) + min_v[:-1]
        unnorm[..., -1]  = torch.where(actions[..., -1] > 0.0, 1.0, -1.0)
        return unnorm


# ==============================================================================
# VLA MODEL
# ==============================================================================

class Qwen3_5_DiffusionPolicy(nn.Module):
    def __init__(self, action_dim=7, action_horizon=16, qwen_hidden_dim=1024,
                 ee_state_dim=16):
        super().__init__()
        self.action_dim     = action_dim
        self.action_horizon = action_horizon

        print(f"Loading Qwen3.5-0.8B... (Single-Cam + DiT + EE Dim: {ee_state_dim})")
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3.5-0.8B", trust_remote_code=True
        )
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = "left"

        self.vlm = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen3.5-0.8B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )

        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        self.vlm = get_peft_model(self.vlm, lora_config)
        print("ℹ️  Fresh LoRA initialized.")

        self.vlm_projector = nn.Sequential(
            nn.Linear(qwen_hidden_dim * 2, qwen_hidden_dim),
            nn.LayerNorm(qwen_hidden_dim),
            nn.GELU(),
            nn.Linear(qwen_hidden_dim, 256),
        )

        self.unet = ActionDiT(
            action_dim=action_dim,
            global_cond_dim=256,
            state_dim=ee_state_dim,
            hidden_size=384,
            depth=12,
            num_heads=6,
            action_horizon=action_horizon,
        )

    def extract_qwen_context(self, text_instructions, agent_images,
                              return_attention=False):
        formatted_prompts = []
        flat_images       = []
        obs_per_inst      = len(agent_images) // len(text_instructions)

        for idx, text in enumerate(text_instructions):
            user_content = [{"type": "image"} for _ in range(obs_per_inst)]
            user_content.append({
                "type": "text",
                "text": f"What action should the robot take to {text.lower()}?"
            })
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",   "content": user_content},
            ]
            formatted_prompts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            flat_images.extend(
                agent_images[idx * obs_per_inst: (idx + 1) * obs_per_inst]
            )

        inputs = self.processor(
            text=formatted_prompts, images=flat_images,
            return_tensors="pt", padding=True,
        )

        clean_inputs = {}
        for key, val in inputs.items():
            if key in ["input_ids", "attention_mask", "mm_token_type_ids", "image_grid_thw"]:
                clean_inputs[key] = val.to(self.vlm.device).long()
            elif key in ["pixel_values", "image_features"]:
                clean_inputs[key] = val.to(self.vlm.device, dtype=self.vlm.dtype)
            else:
                clean_inputs[key] = val.to(self.vlm.device)

        outputs = self.vlm(
            **clean_inputs,
            output_hidden_states=True,
            output_attentions=return_attention,
        )

        layer_minus_1 = outputs.hidden_states[-1]
        layer_minus_2 = outputs.hidden_states[-2]
        fused_hidden  = torch.cat([layer_minus_1, layer_minus_2], dim=-1)
        vlm_embeds    = self.vlm_projector(fused_hidden.to(torch.float32))

        if return_attention:
            attn = (outputs.attentions[-1]
                    if (outputs.attentions is not None and len(outputs.attentions) > 0)
                    else None)
            return vlm_embeds, attn
        return vlm_embeds

    def forward(self, noisy_actions, timesteps, text_instructions, agent_images,
                ee_states, return_attention=False):
        if return_attention:
            vlm_cond, qwen_attn_map = self.extract_qwen_context(
                text_instructions, agent_images, return_attention=True
            )
            noise_pred, unet_attn_list, vlm_seq_len = self.unet(
                sample=noisy_actions, timestep=timesteps,
                global_cond=vlm_cond, states=ee_states,
                return_unet_attn=True,
            )
            return noise_pred, qwen_attn_map, unet_attn_list, vlm_seq_len
        else:
            vlm_cond   = self.extract_qwen_context(text_instructions, agent_images)
            noise_pred = self.unet(
                sample=noisy_actions, timestep=timesteps,
                global_cond=vlm_cond, states=ee_states,
            )
            return noise_pred


# ==============================================================================
# CHECKPOINT LOADER
# ==============================================================================

def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, device):
    """
    Loads checkpoint into model, optimizer, and scheduler.
    Returns (start_epoch, global_step, best_val_loss, normalizer)
    """
    print(f"\n🔁 Resuming from checkpoint: '{checkpoint_path}'", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ── Model weights ──────────────────────────────────────────────────────
    missing, unexpected = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False
    )
    if missing:
        print(f"  ⚠️  Missing keys  : {missing}", flush=True)
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {unexpected}", flush=True)

    # ── Optimizer + scheduler ──────────────────────────────────────────────
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    # ── Move optimizer states to correct device ────────────────────────────
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    start_epoch  = checkpoint['epoch']          # resume AFTER this epoch
    best_val_loss = checkpoint['val_loss']

    # ── Rebuild normalizer from saved stats ───────────────────────────────
    normalizer = ActionNormalizer(
        min_val=checkpoint['action_min'],
        max_val=checkpoint['action_max'],
    )

    # ── EE dim sanity check ────────────────────────────────────────────────
    saved_ee_dim = checkpoint.get('ee_state_dim', None)
    print(f"  ✅ Loaded epoch {start_epoch} | val_loss={best_val_loss:.5f} "
          f"| ee_state_dim={saved_ee_dim}", flush=True)

    return start_epoch, best_val_loss, normalizer


# ==============================================================================
# DATASET
# ==============================================================================

class LiberoOpenVLADataset(Dataset):
    def __init__(self, data_dir, action_horizon=16, obs_horizon=1,
                 split='train', val_demos_per_task=5):
        self.action_horizon = action_horizon
        self.obs_horizon    = obs_horizon
        self.is_train       = (split == 'train')
        self.step_lookup    = []

        print(f"Mapping OpenVLA RLDS dataset indices from {data_dir}... (Low RAM Mode)")
        builder    = tfds.builder_from_directory(data_dir)
        tf_dataset = builder.as_dataset(split='all')

        tasks_map = {}
        for ep_idx, ep in enumerate(tfds.as_numpy(tf_dataset)):
            first_step = next(iter(ep['steps']))
            raw_inst   = first_step['language_instruction']
            inst       = raw_inst.decode('utf-8') if isinstance(raw_inst, bytes) else raw_inst
            inst       = inst.capitalize() + "."
            num_steps  = len(list(ep['steps']))
            tasks_map.setdefault(inst, []).append({
                'ep_idx': ep_idx, 'num_steps': num_steps, 'instruction': inst
            })

        print(f"Found {len(tasks_map)} unique tasks.")

        self.valid_ep_indices = set()
        for inst, demos in tasks_map.items():
            if split == 'train':
                target = demos[:-val_demos_per_task] if val_demos_per_task < len(demos) else demos
            else:
                target = demos[-val_demos_per_task:] if val_demos_per_task < len(demos) else demos
            for demo in target:
                self.valid_ep_indices.add(demo['ep_idx'])
                for step in range(demo['num_steps']):
                    self.step_lookup.append({
                        'ep_idx':      demo['ep_idx'],
                        'step':        step,
                        'num_steps':   demo['num_steps'],
                        'instruction': demo['instruction'],
                    })

        print(f"Mapped {len(self.step_lookup)} steps for {split} split.")

        self.episodes = {}
        print(f"Caching episode arrays for {split} split...")
        for ep_idx, ep in tqdm(
            enumerate(tfds.as_numpy(tf_dataset)),
            total=len(tasks_map) * sum(len(v) for v in tasks_map.values()) // len(tasks_map),
        ):
            if ep_idx in self.valid_ep_indices:
                agent_imgs, ee_states, actions = [], [], []
                for step in ep['steps']:
                    obs     = step['observation']
                    img_key = 'image' if 'image' in obs else 'agentview_image'
                    agent_imgs.append(obs[img_key])
                    ee_states.append(obs['state'])
                    actions.append(step['action'])
                self.episodes[ep_idx] = {
                    'agent_imgs': np.stack(agent_imgs),
                    'ee_states':  np.stack(ee_states),
                    'actions':    np.stack(actions),
                }

    def __len__(self):
        return len(self.step_lookup)

    def __getitem__(self, idx):
        item     = self.step_lookup[idx]
        ep       = self.episodes[item['ep_idx']]
        step_idx = item['step']
        inst     = item['instruction']

        agent_imgs_seq, ee_states_seq = [], []
        for t in range(step_idx - self.obs_horizon + 1, step_idx + 1):
            t_idx = max(0, t)
            img_a = ep['agent_imgs'][t_idx]
            pil_a = Image.fromarray(
                img_a if img_a.dtype == np.uint8 else (img_a * 255).astype(np.uint8)
            )
            agent_imgs_seq.append(pil_a)
            ee_states_seq.append(ep['ee_states'][t_idx])

        ee_state_tensor = torch.from_numpy(
            np.concatenate(ee_states_seq, axis=-1)
        ).float()

        end_step  = step_idx + self.action_horizon
        num_steps = item['num_steps']
        if end_step <= num_steps:
            action_chunk = ep['actions'][step_idx:end_step]
        else:
            avail        = ep['actions'][step_idx:]
            action_chunk = np.concatenate(
                [avail, np.repeat(avail[-1:], end_step - num_steps, axis=0)], axis=0
            )
        return agent_imgs_seq, inst, torch.from_numpy(action_chunk).float(), ee_state_tensor


def custom_collate_fn(batch):
    agent_imgs, instructions, actions, ee_states = [], [], [], []
    for item in batch:
        agent_imgs.extend(item[0])
        instructions.append(item[1])
        actions.append(item[2])
        ee_states.append(item[3])
    return agent_imgs, instructions, torch.stack(actions, dim=0), torch.stack(ee_states, dim=0)


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_qwen_vla(
    data_dir,
    output_dir="checkpoints_with_agentview_dit",
    resume_from="checkpoints_with_agentview_dit",   # ← folder to look for latest_model.pt
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    OBS_HORIZON = 1
    EPOCHS      = 100

    # ── Dataset first — needed to get dynamic_ee_dim and normalizer ────────
    train_set = LiberoOpenVLADataset(
        data_dir, obs_horizon=OBS_HORIZON, split='train', val_demos_per_task=5
    )
    val_set = LiberoOpenVLADataset(
        data_dir, obs_horizon=OBS_HORIZON, split='val', val_demos_per_task=5
    )

    print(f"\n[INFO] Total training steps loaded: {len(train_set)}")

    # ── Action normalizer from train set (overridden if resuming) ──────────
    print("\nCalculating dataset action bounds for normalizer...")
    all_actions = [train_set[i][2]
                   for i in tqdm(range(len(train_set)), desc="Reading actions")]
    all_actions_tensor = torch.cat(all_actions, dim=0)
    dataset_min        = all_actions_tensor.min(dim=0)[0].numpy()
    dataset_max        = all_actions_tensor.max(dim=0)[0].numpy()
    normalizer         = ActionNormalizer(min_val=dataset_min, max_val=dataset_max)

    dynamic_ee_dim = train_set[0][3].shape[0]

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True,
                              collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False,
                              collate_fn=custom_collate_fn)

    # ── Model ──────────────────────────────────────────────────────────────
    model = Qwen3_5_DiffusionPolicy(ee_state_dim=dynamic_ee_dim).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    total_training_steps = len(train_loader) * EPOCHS
    warmup_steps         = int(total_training_steps * 0.05)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # ── Resume from checkpoint if available ───────────────────────────────
    start_epoch   = 0
    best_val_loss = float('inf')
    global_step   = 0

    ckpt_path = os.path.join(resume_from, "latest_model.pt")
    if os.path.exists(ckpt_path):
        start_epoch, best_val_loss, normalizer = load_checkpoint(
            ckpt_path, model, optimizer, lr_scheduler, device
        )
        # global_step = steps already completed
        global_step = start_epoch * len(train_loader)
        print(f"▶️  Resuming training from epoch {start_epoch + 1}", flush=True)
    else:
        print(f"ℹ️  No checkpoint found at '{ckpt_path}' — starting fresh.", flush=True)

    # ── WandB — resume run if possible ────────────────────────────────────
    wandb.init(
        project="Qwen3.5-VLA-Libero",
        resume="allow",          # resumes existing run if id matches
        config={
            "lr":                 1e-4,
            "batch_size":         8,
            "epochs":             EPOCHS,
            "horizon":            16,
            "obs_horizon":        OBS_HORIZON,
            "val_demos_per_task": 5,
            "resumed_from_epoch": start_epoch,
        },
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2"
    )

    # ── Epoch loop — starts from start_epoch ──────────────────────────────
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_noise_loss        = 0
        train_action_loss       = 0
        running_action_loss_500 = 0.0
        running_noise_loss_500  = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")

        for batch_idx, (agent_imgs, insts, clean_actions, ee_states) in enumerate(pbar):
            global_step   += 1
            clean_actions  = clean_actions.to(device)
            ee_states      = ee_states.to(device)

            norm_ee_states = ee_states.clone()
            if torch.rand(1).item() < 0.2:
                norm_ee_states = torch.zeros_like(norm_ee_states)

            norm_actions = normalizer.normalize(clean_actions)

            if epoch == start_epoch and batch_idx == 0:
                print("\n" + "=" * 60)
                print("🛑 SANITY CHECK: FIRST BATCH DATA 🛑")
                print(f"Instruction [0]: '{insts[0]}'")
                recent_agent_img = agent_imgs[OBS_HORIZON - 1]
                recent_agent_img.save("sanity_check_agentview.png")
                wandb.log({"Sanity Check/Agent View": wandb.Image(recent_agent_img)})
                print("=" * 60 + "\n")

            noise       = torch.randn_like(norm_actions).to(device)
            timesteps_t = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (norm_actions.shape[0],), device=device,
            ).long()
            noisy_actions = noise_scheduler.add_noise(norm_actions, noise, timesteps_t)

            get_attn = (batch_idx % 500 == 0)

            if get_attn:
                pred_noise, qwen_attn_map, unet_attn_list, vlm_seq_len = model(
                    noisy_actions, timesteps_t, insts, agent_imgs, norm_ee_states,
                    return_attention=True,
                )

                # ── Qwen VLM self-attention ─────────────────────────────────
                try:
                    if qwen_attn_map is not None:
                        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
                        avg_qwen_heads   = qwen_attn_map[0].mean(dim=0).detach().cpu().float().numpy()
                        final_token_attn = avg_qwen_heads[-1, :]
                        ax1.plot(final_token_attn, color='blue', alpha=0.8)
                        ax1.fill_between(range(len(final_token_attn)),
                                         final_token_attn, color='blue', alpha=0.3)
                        ax1.set_title("Qwen VLM Attention (Context Aggregation)")
                        ax1.set_xlabel("Token Index")
                        ax1.set_ylabel("Attention Weight")
                        fig.suptitle(f"Epoch {epoch+1} | Inst: '{insts[0]}'", fontsize=14)
                        plt.tight_layout()
                        wandb.log({"VLM_Attention_Map": wandb.Image(plt)})
                        plt.close()
                except Exception:
                    pass

                # ── DiT → VLM attention ratio per block ────────────────────
                try:
                    if unet_attn_list is not None and len(unet_attn_list) > 0:
                        vlm_ratios = []
                        for attn_w in unet_attn_list:
                            w            = attn_w[0].detach().cpu().float()
                            action_rows  = w[-16:, :]
                            attn_to_vlm  = action_rows[:, :vlm_seq_len].sum(dim=-1).mean().item()
                            attn_to_self = action_rows[:, vlm_seq_len:].sum(dim=-1).mean().item()
                            ratio        = attn_to_vlm / (attn_to_vlm + attn_to_self + 1e-8)
                            vlm_ratios.append(ratio)

                        fig2, ax2 = plt.subplots(figsize=(12, 4))
                        colors = [
                            'green'  if r > 0.3 else
                            'orange' if r > 0.1 else
                            'red'
                            for r in vlm_ratios
                        ]
                        ax2.bar(range(len(vlm_ratios)), vlm_ratios, color=colors, alpha=0.85)
                        ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% VLM')
                        ax2.axhline(0.1, color='red',  linestyle='--', alpha=0.5, label='10% threshold')
                        ax2.set_xlabel("DiT Block Index")
                        ax2.set_ylabel("Fraction of attention → VLM tokens")
                        ax2.set_title(
                            f"DiT→VLM Attention Ratio | Epoch {epoch+1} | "
                            f"Avg={np.mean(vlm_ratios):.3f} | '{insts[0][:40]}'"
                        )
                        ax2.set_ylim(0, 1)
                        ax2.legend()
                        plt.tight_layout()
                        wandb.log({
                            "DiT_VLM_Attention_Per_Block": wandb.Image(plt),
                            "dit_attn/vlm_ratio_mean":     np.mean(vlm_ratios),
                            "dit_attn/vlm_ratio_min":      np.min(vlm_ratios),
                            "dit_attn/vlm_ratio_max":      np.max(vlm_ratios),
                            "dit_attn/vlm_ratio_block0":   vlm_ratios[0],
                            "dit_attn/vlm_ratio_block11":  vlm_ratios[-1],
                        })
                        plt.close()
                        print(
                            f"[DiT Attn] Epoch {epoch+1} | "
                            f"VLM ratio per block: {[f'{r:.2f}' for r in vlm_ratios]}",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[DiT attn viz error]: {e}", flush=True)

            else:
                pred_noise = model(
                    noisy_actions, timesteps_t, insts, agent_imgs, norm_ee_states
                )

            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            with torch.no_grad():
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                alpha_prod_t   = alphas_cumprod[timesteps_t].view(-1, 1, 1)
                beta_prod_t    = 1 - alpha_prod_t
                pred_norm_act  = (noisy_actions - beta_prod_t ** 0.5 * pred_noise) / alpha_prod_t ** 0.5
                pred_physical  = normalizer.unnormalize(pred_norm_act)
                action_l1_loss = F.l1_loss(pred_physical, clean_actions)

            train_noise_loss        += loss.item()
            train_action_loss       += action_l1_loss.item()
            running_action_loss_500 += action_l1_loss.item()
            running_noise_loss_500  += loss.item()

            if global_step % 500 == 0:
                wandb.log({
                    "global_step":              global_step,
                    "train/step_action_l1_500": running_action_loss_500 / 500.0,
                    "train/step_noise_mse_500": running_noise_loss_500  / 500.0,
                    "train/learning_rate":      lr_scheduler.get_last_lr()[0],
                })
                running_action_loss_500 = 0.0
                running_noise_loss_500  = 0.0

            pbar.set_postfix({
                "Noise MSE": f"{loss.item():.4f}",
                "Action L1": f"{action_l1_loss.item():.4f}",
            })

        avg_train_noise_loss  = train_noise_loss  / len(train_loader)
        avg_train_action_loss = train_action_loss / len(train_loader)

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_noise_loss  = 0
        val_action_loss = 0

        with torch.no_grad():
            for agent_imgs, insts, clean_actions, ee_states in tqdm(
                val_loader, desc=f"Epoch {epoch+1} Val"
            ):
                clean_actions  = clean_actions.to(device)
                ee_states      = ee_states.to(device)

                norm_ee_states = ee_states.clone()
                if torch.rand(1).item() < 0.2:
                    norm_ee_states = torch.zeros_like(norm_ee_states)

                norm_actions  = normalizer.normalize(clean_actions)
                noise         = torch.randn_like(norm_actions).to(device)
                timesteps_t   = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (norm_actions.shape[0],), device=device,
                ).long()
                noisy_actions = noise_scheduler.add_noise(norm_actions, noise, timesteps_t)

                pred_noise     = model(noisy_actions, timesteps_t, insts, agent_imgs, norm_ee_states)
                v_loss         = F.mse_loss(pred_noise, noise)
                val_noise_loss += v_loss.item()

                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                alpha_prod_t   = alphas_cumprod[timesteps_t].view(-1, 1, 1)
                beta_prod_t    = 1 - alpha_prod_t
                pred_norm_act  = (noisy_actions - beta_prod_t ** 0.5 * pred_noise) / alpha_prod_t ** 0.5
                pred_physical  = normalizer.unnormalize(pred_norm_act)
                v_action_l1    = F.l1_loss(pred_physical, clean_actions)
                val_action_loss += v_action_l1.item()

        avg_val_noise_loss  = val_noise_loss  / len(val_loader)
        avg_val_action_loss = val_action_loss / len(val_loader)

        wandb.log({
            "epoch":            epoch + 1,
            "global_step":      global_step,
            "train_noise_loss": avg_train_noise_loss,
            "train_action_l1":  avg_train_action_loss,
            "val_noise_loss":   avg_val_noise_loss,
            "val_action_l1":    avg_val_action_loss,
        })

        print(
            f"Epoch {epoch+1} | "
            f"Train Noise: {avg_train_noise_loss:.5f} | "
            f"Train L1: {avg_train_action_loss:.5f} | "
            f"Val Noise: {avg_val_noise_loss:.5f} | "
            f"Val L1: {avg_val_action_loss:.5f}"
        )

        checkpoint = {
            'epoch':                   epoch + 1,
            'model_state_dict':        model.state_dict(),
            'optimizer_state_dict':    optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'val_loss':                avg_val_noise_loss,
            'ee_state_dim':            dynamic_ee_dim,
            'action_min':              dataset_min,
            'action_max':              dataset_max,
        }
        torch.save(checkpoint, os.path.join(output_dir, "latest_model.pt"))
        if avg_val_noise_loss < best_val_loss:
            best_val_loss = avg_val_noise_loss
            torch.save(checkpoint, os.path.join(output_dir, "best_model.pt"))
            print("New best model saved!")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    train_qwen_vla(
        data_dir="./libero_no_noops_dataset/libero_object_no_noops/1.0.0",
        output_dir="checkpoints_with_agentview_dit_pretrained_full",
        resume_from="checkpoints_with_agentview_dit",   # ← loads latest_model.pt from here
    )