![Logo](docs/logo_magpie1_4.png)

# MAGPIE2025

MAGPIE: MULTI-STAIN AND MULTI-ORGAN PATHOLOGY IMAGE RESTORATION CHALLENGE (2025)  
Challenge URL: https://www.codabench.org/competitions/4880/

# Environment Setup

### Conda environment

```bash
conda create -n magpie python=3.10
conda activate magpie
pip install -r requirements.txt
```

### Docker Image

```bash
sudo docker pull keunhobyeon/magpie2025:latest
```

* Note: Patch generation and evaluation will be executed within this environment.

# Dataset

The dataset is provided as .svs Whole Slide Images (WSIs).
* Note: The test set patches are generated using following commands.

### Extract Patches

Patch extraction methods differ based on stain types:

1. H&E and IHC Stains  
   The patch extraction code for these stains is implemented based on the following paper:  
   Lu, Ming Y., et al. "Data-efficient and weakly supervised computational pathology on whole-slide images." Nature Biomedical Engineering 5.6 (2021): 555-570.
2. Special Stains  
   For special stains, a custom patch extraction method is used.

Each slide will be cropped to 512x512 pixels(40x magnification), resized to 256x256 pixels(20x magnification).  
To extract image patches from Whole Slide Images (WSIs), run the following command:

```bash
cd make_patches
git clone https://github.com/mahmoodlab/CLAM
python make_patches.py --source_dir "SLIDE_DIR" --save_dir "PATCH_SAVE_DIR"  --gpus 1 2 3 4
cd ../
```

### Data Folder Structure

```
SLIDE_DIR/
├── train/
│   ├── gt/
│   │   ├── Slide_1_0um.svs
│   │   └── ...
│   └── blur/
│       ├── Slide_1_1um.svs
│       ├── Slide_1_2um.svs
│       └── ...
└── val/
    ├── gt/
    │   ├── Slide_2_0um.svs
    │   └── ...
    └── blur/
        ├── Slide_2_1um.svs
        ├── Slide_2_2um.svs
        └── ...
```

```
PATCH_DIR/
├── train/
│   ├── blur/
│   │   └── SLIDE_1/    # Subfolder for each slide ID
│   │       ├── 1um/
│   │       │   ├── Slide_1_4656_41216.png
│   │       │   ├── Slide_1_4656_41728.png
│   │       │   └── ...
│   │       ├── 2um/
│   │       │   ├── Slide_1_4656_41216.png
│   │       │   ├── Slide_1_4656_41728.png
│   │       │   └── ...
│   │       └── ...
│   │ 
│   ├── gt/
│   │   └── SLIDE_1/    # Subfolder for each slide ID
│   │       └── 0um/       # Original slide with 0μm offset
│   │           ├── Slide_1_4656_41216.png
│   │           ├── Slide_1_4656_41728.png
│   │           └── ...
│   │
│   ├── masks/             # (CLAM) Stores tissue segmentation masks
│   │   ├── Slide_1_0um.jpg
│   │   └── ...
│   │
│   ├── patches/           # (CLAM) Stores extracted coords
│   │   ├── Slide_1_0um.h5
│   │   └── ... 
│   │
│   └── stitches/          # (CLAM) Stores stitched heatmap images 
│       ├── Slide_1_0um.jpg 
│       └── ...
│
└── val/
    ├── blur/...           #same as train
    └── gt/...             #same as train
```

# Sample inference code

### Installation

This sample inference code is implemented based on the following paper:
Yue, Zongsheng, Jianyi Wang, and Chen Change Loy. "Resshift: Efficient diffusion model for image super-resolution by residual shifting." Advances in Neural Information Processing Systems 36 (2024).

```bash
git clone https://github.com/zsyOAOA/ResShift
```

#### Copy python scripts to run inference

```bash
cp -r inference_sample/code/* ResShift/
```

#### Download Model Weights

1. Download the pre-trained VQGAN model weight ("autoencoder_vq_f4.pth") from the [ResShift GitHub repository](https://github.com/zsyOAOA/ResShift) and save it in the inference_sample/weights directory.
2. Download "model_400000.pth" from [this link](https://github.com/KeunhoByeon/MAGPIE2025/releases/tag/v1.0) and save it in the inference_sample/weights directory.

### How to build your docker image

#### Install the required NVIDIA container toolkit to support GPU usage in Docker
```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

#### Set a name for your Docker image
```bash
export DOCKER_IMAGE_NAME="YOUR_DOCKER_IMAGE_NAME"
```

#### Build docker image
```bash
sudo docker build -t "$DOCKER_IMAGE_NAME" -f Dockerfile .
```

#### (Optional) Save docker image file
```bash
sudo docker save -o "$DOCKER_IMAGE_NAME".tat "$DOCKER_IMAGE_NAME"
```

#### (Optional) Change the file permissions to 777
```bash
sudo chmod 777 "$DOCKER_IMAGE_NAME".tar
```
* Note: The participants are asked to submit a docker image through our website (https://www.codabench.org/competitions/4880).

### Run inference

#### Set your data path

```bash
export DATA_PATH="YOUR_PATCH_DATA_DIR"
```

#### Run inference

```bash
sudo systemctl restart docker
sudo docker run --gpus device=0 --network=host --privileged -it \
-v "$DATA_PATH":/data \
--name magpie2025_sample_run \
"$DOCKER_IMAGE_NAME" python inference.py
```

#### Copy the result files from the container

```bash
sudo docker cp magpie2025_sample_run:/workspace/results ./results
```

# Evaluation code

### Run evaluation

```bash
python evaluation.py --gt_dir "YOUR_GT_DIR" --pred_dir ./results/test/blur
```

See "evaluation.py" for more detail.

```python
import lpips

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

lpips_loss = lpips.LPIPS(net='vgg')


def calcualte_psnr(pred, gt):
    psnr_score = peak_signal_noise_ratio(pred, gt)
    return psnr_score


def calcualte_ssim(pred, gt):
    ssim_score = structural_similarity(pred, gt, channel_axis=2)
    return ssim_score


def calculate_lpips(pred, gt):
    pred_tensor = lpips.im2tensor(pred)
    gt_tensor = lpips.im2tensor(gt)
    return lpips_loss(pred_tensor, gt_tensor).item()
```

