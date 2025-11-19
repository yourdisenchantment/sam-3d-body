# Installation Guide for SAM 3D Body

## Step-by-Step Installation

### 1. Create and Activate Environment

```bash
conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body
```

### 2. Install PyTorch

Please install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/).

### 3. Install Python Dependencies

```bash
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub
```

### 4. Install Detectron2

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
```

### 5. Install MoGe (Optional)

```bash
pip install git+https://github.com/microsoft/MoGe.git
```
