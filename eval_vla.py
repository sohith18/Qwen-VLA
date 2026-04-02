import os
import math
import warnings
from collections import deque

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ROBOSUITE_QUIET"] = "True"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
from timm.models.vision_transformer import Mlp

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


# ==============================================================================
# INLINED: ActionNormalizer  (from train_vla.py — frozen to match checkpoint)
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
# INLINED: policy_head_v3.py  — OLD arch that matches the checkpoint
#          (prefix-concat design: cond_seq_proj + single self.attn per block)
# ==============================================================================

def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half  = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args      = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        return self.mlp(
            self.timestep_embedding(t, self.frequency_embedding_size)
        )


def _get_1d_sincos_pos_embed(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega  = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega  = 1.0 / 10000 ** omega
    pos    = pos.reshape(-1)
    out    = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class _DiTBlock(nn.Module):
    """OLD DiTBlock: single self-attention over prefix-concat(vlm_tokens, action_tokens)."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=0.0, bias=True
        )
        self.norm2            = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim        = int(hidden_size * mlp_ratio)
        approx_gelu           = lambda: nn.GELU(approximate="tanh")
        self.mlp              = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, need_weights=False):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x_norm       = _modulate(self.norm1(x), shift_msa, scale_msa)
        x_sa, attn_w = self.attn(
            x_norm, x_norm, x_norm,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        x = x + gate_msa.unsqueeze(1) * x_sa
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x, attn_w


class _ActionFinalLayer(nn.Module):
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.norm_final       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear           = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


class ActionDiT(nn.Module):
    """OLD ActionDiT — prefix-concat design. Must match checkpoint exactly."""
    def __init__(
        self,
        action_dim=7,
        global_cond_dim=256,
        state_dim=16,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        action_horizon=16,
    ):
        super().__init__()
        self.action_dim     = action_dim
        self.action_horizon = action_horizon
        self.hidden_size    = hidden_size

        self.x_embedder   = nn.Linear(action_dim, hidden_size, bias=True)
        self.pos_embed     = nn.Parameter(
            torch.zeros(1, action_horizon, hidden_size), requires_grad=False
        )
        self.cond_seq_proj = nn.Linear(global_cond_dim, hidden_size)
        self.t_embedder    = _TimestepEmbedder(hidden_size)
        self.state_mlp     = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList([
            _DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = _ActionFinalLayer(hidden_size, action_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = _get_1d_sincos_pos_embed(
            self.hidden_size, np.arange(self.action_horizon)
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        for w_attr in [
            self.cond_seq_proj.weight, self.state_mlp[0].weight,
            self.state_mlp[2].weight,  self.t_embedder.mlp[0].weight,
            self.t_embedder.mlp[2].weight,
        ]:
            nn.init.normal_(w_attr, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias,   0)

    def forward(self, sample, timestep, global_cond=None, states=None,
                return_unet_attn=False):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])

        x_actions   = self.x_embedder(sample) + self.pos_embed
        x_vlm       = self.cond_seq_proj(global_cond)
        x_combined  = torch.cat([x_vlm, x_actions], dim=1)
        vlm_seq_len = x_vlm.shape[1]

        t = self.t_embedder(timestep)
        c = t
        if states is not None:
            c = c + self.state_mlp(states)

        all_attn_weights = [] if return_unet_attn else None
        for block in self.blocks:
            x_combined, attn_w = block(x_combined, c, need_weights=return_unet_attn)
            if return_unet_attn and attn_w is not None:
                all_attn_weights.append(attn_w)

        x_out = x_combined[:, -self.action_horizon:, :]
        x_out = self.final_layer(x_out, c)

        if return_unet_attn:
            return x_out, all_attn_weights, vlm_seq_len
        return x_out


# ==============================================================================
# INLINED: Qwen3_5_DiffusionPolicy  (from train_vla.py — frozen to match checkpoint)
# ==============================================================================

class Qwen3_5_DiffusionPolicy(nn.Module):
    def __init__(self, action_dim=7, action_horizon=16, qwen_hidden_dim=1024,
                 ee_state_dim=8):
        super().__init__()
        self.action_dim     = action_dim
        self.action_horizon = action_horizon

        print(f"Loading Qwen3.5-0.8B... (ee_state_dim={ee_state_dim})")
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

        # NOTE: vlm_projector lives in the checkpoint under "vlm_projector.*"
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

    def extract_qwen_context(self, text_instructions, agent_images):
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
                {"role": "system",
                 "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
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
            if key in ["input_ids", "attention_mask",
                       "mm_token_type_ids", "image_grid_thw"]:
                clean_inputs[key] = val.to(self.vlm.device).long()
            elif key in ["pixel_values", "image_features"]:
                clean_inputs[key] = val.to(self.vlm.device, dtype=self.vlm.dtype)
            else:
                clean_inputs[key] = val.to(self.vlm.device)

        outputs = self.vlm(**clean_inputs, output_hidden_states=True)

        fused   = torch.cat(
            [outputs.hidden_states[-1], outputs.hidden_states[-2]], dim=-1
        )
        return self.vlm_projector(fused.to(torch.float32))


# ==============================================================================
# UTILS
# ==============================================================================

def quat2axisangle(quat):
    if quat[3] > 1.0:   quat[3] =  1.0
    if quat[3] < -1.0:  quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ==============================================================================
# REVERSE DIFFUSION
# ==============================================================================

@torch.no_grad()
def get_action_chunk(
    model, normalizer, agent_imgs, instruction, ee_state, device,
    action_dim=7, horizon=16, num_steps=100, debug=False,
):
    model.eval()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2"
    )

    cond_embed = model.extract_qwen_context([instruction], agent_imgs)
    cond_embed = cond_embed.to(device)          # align VLM device → unet device

    if debug:
        print("\n" + "-" * 40)
        print("🔍 INFERENCE SANITY CHECK")
        print(f"  Instruction  : '{instruction}'")
        print(f"  EE state shape: {ee_state.shape}")
        print(f"  cond_embed   : {cond_embed.shape}  on {cond_embed.device}")
        print("-" * 40)

    actions   = torch.randn((1, horizon, action_dim), device=device)
    ee_tensor = torch.tensor(ee_state, dtype=torch.float32, device=device).unsqueeze(0)

    noise_scheduler.set_timesteps(num_steps)
    for t in noise_scheduler.timesteps:
        pred_noise = model.unet(
            sample=actions,
            timestep=t.unsqueeze(0).to(device),
            global_cond=cond_embed,
            states=ee_tensor,
        )
        actions = noise_scheduler.step(pred_noise, t, actions).prev_sample

    final_actions = normalizer.unnormalize(actions)
    final_actions[..., -1] = torch.where(
        final_actions[..., -1] > 0.0,
        torch.tensor(1.0,  device=device),
        torch.tensor(-1.0, device=device),
    )
    return final_actions.squeeze(0).cpu().numpy()


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_libero(
    checkpoint_path,
    benchmark_name="libero_object",
    video_dir="eval_videos",
):
    os.makedirs(video_dir, exist_ok=True)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    OBS_HORIZON = 1

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ee_state_dim = checkpoint.get('ee_state_dim', 8)
    action_min   = checkpoint.get('action_min', np.array([-1.0] * 7))
    action_max   = checkpoint.get('action_max', np.array([ 1.0] * 7))
    normalizer   = ActionNormalizer(min_val=action_min, max_val=action_max)

    print(f"  ee_state_dim = {ee_state_dim}")
    print(f"  action_min   = {np.round(action_min, 3)}")
    print(f"  action_max   = {np.round(action_max, 3)}")

    model = Qwen3_5_DiffusionPolicy(ee_state_dim=ee_state_dim).to(device)

    # strict=False handles LoRA adapter key name differences
    missing, unexpected = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False
    )
    # Only flag genuinely structural mismatches (ignore expected LoRA base keys)
    structural_missing   = [k for k in missing   if 'lora_' not in k]
    structural_unexpected = [k for k in unexpected if 'lora_' not in k]
    if structural_missing:
        print(f"  ⚠️  Structurally missing   : {structural_missing}")
    if structural_unexpected:
        print(f"  ⚠️  Structurally unexpected: {structural_unexpected}")

    # Move non-VLM modules to inference device explicitly
    # (VLM uses device_map="auto" so we don't touch it)
    model.vlm_projector = model.vlm_projector.to(device)
    model.unet          = model.unet.to(device)
    model.eval()
    print("✅ Model loaded successfully.\n")

    benchmark  = get_benchmark(benchmark_name)()
    num_tasks  = benchmark.get_num_tasks()
    total_success  = 0
    total_attempts = 0

    for task_id in range(num_tasks):
        task = benchmark.get_task(task_id)
        # Match training preprocessing exactly: .capitalize() + "."
        instruction    = task.language.capitalize() + "."
        
        # make instruction blank
        instruction = "Pick up the salad dressing and place it in the basket."
        initial_states = benchmark.get_task_init_states(task_id)
        num_episodes   = len(initial_states)

        print(f"\n--- Task {task_id+1}/{num_tasks}: '{instruction}' ({num_episodes} eps) ---")

        env_args = {
            "bddl_file_name": os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder, task.bddl_file,
            ),
            "camera_heights": 256,
            "camera_widths":  256,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(42)

        task_success = 0

        for ep in range(num_episodes):
            env.reset()
            obs = env.set_init_state(initial_states[ep])

            dummy = [0., 0., 0., 0., 0., 0., -1.0]
            for _ in range(15):
                obs, _, _, _ = env.step(dummy)

            done             = False
            step_count       = 0
            max_steps        = 280
            execution_length = 4
            action_buffer    = []
            video_frames     = []
            agent_buffer     = deque(maxlen=OBS_HORIZON)
            state_buffer     = deque(maxlen=OBS_HORIZON)

            pbar = tqdm(total=max_steps, desc=f"Ep {ep+1}/{num_episodes}")
            while step_count < max_steps and not done:
                raw   = obs.get("agentview_image")
                agent_np = raw[::-1, ::-1]
                if agent_np.dtype != np.uint8:
                    agent_np = (agent_np * 255).astype(np.uint8)
                video_frames.append(agent_np.copy())
                
                # To send black images for ablation
                # agent_np = np.zeros_like(agent_np, dtype=np.uint8)

                agent_pil     = Image.fromarray(agent_np).resize((256, 256))
                eef_pos       = obs["robot0_eef_pos"]
                eef_axisangle = quat2axisangle(obs["robot0_eef_quat"])
                gripper_qpos  = obs["robot0_gripper_qpos"]
                state_8d      = np.concatenate(
                    [eef_pos, eef_axisangle, gripper_qpos]
                ).astype(np.float32)

                if step_count == 0:
                    for _ in range(OBS_HORIZON):
                        agent_buffer.append(agent_pil)
                        state_buffer.append(state_8d)
                else:
                    agent_buffer.append(agent_pil)
                    state_buffer.append(state_8d)

                if len(action_buffer) == 0:
                    flat_ee = np.concatenate(list(state_buffer), axis=-1)
                    chunk   = get_action_chunk(
                        model=model, normalizer=normalizer,
                        agent_imgs=list(agent_buffer),
                        instruction=instruction,
                        ee_state=flat_ee,
                        device=device,
                        debug=(step_count == 0),
                    )
                    action_buffer = chunk[:execution_length].tolist()

                obs, _, done, _ = env.step(action_buffer.pop(0))
                step_count += 1
                pbar.update(1)
            pbar.close()

            success = env.check_success()
            task_success  += int(success)
            total_success += int(success)
            total_attempts += 1
            print(f"  {'✅' if success else '❌'} Episode {ep+1}")

            imageio.mimsave(
                os.path.join(
                    video_dir,
                    f"task{task_id}_ep{ep}_{'success' if success else 'fail'}.mp4"
                ),
                video_frames, fps=20,
            )

        env.close()
        print(f"🎯 Task {task_id+1} SR: "
              f"{task_success/num_episodes*100:.1f}% ({task_success}/{num_episodes})")

    print(f"\n{'='*45}")
    print(f"Overall SR: {total_success/total_attempts*100:.2f}% "
          f"({total_success}/{total_attempts})")
    print(f"Videos saved to ./{video_dir}/")
    print(f"{'='*45}")


if __name__ == "__main__":
    ckpt = "checkpoints_with_agentview_dit_pretrained_full/latest_model.pt"
    if os.path.exists(ckpt):
        evaluate_libero(ckpt, benchmark_name="libero_object", video_dir="eval_videos_diff_language_task")
    else:
        print(f"Checkpoint not found at '{ckpt}'")