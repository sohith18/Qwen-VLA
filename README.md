# Qwen-VLA

## Setup

### 1. Install dependencies

### Training 

Download and install the dependencies by seeing the files manually. It works with latest libraries.

### Evaluation

Along with the training dependencies, you also need to clone the LIBERO repository and install the dependencies of that repository. You can follow the instructions in the LIBERO repository to do that.

### 2. Dataset

Dataset can be downloaded from the [LIBERO](https://huggingface.co/datasets/openvla/modified_libero_rlds/tree/main).

After downloading the dataset, change the path in the `train_vla_v3.py` and do not remove the `1.0.0` from the libero folders. The path should be like this:

```path = "/path/to/libero_object/1.0.0"```

### 3. Training

To train the model, run the following command:

```bash
python train_vla_v3.py
```

### 4. Evaluation
Make sure to pull out the `libero` folder from `LIBERO` repository and place it in the same directory as `eval_vla_v3.py`.

To evaluate the model, run the following command:

```bash
python eval_vla_v3.py
```