# MAGPIE2025
MAGPIE: MULTI-STAIN AND MULTI-ORGAN PATHOLOGY IMAGE RESTORATION CHALLENGE (2025)


# Challenge Docker Image
```bash
docker pull keunhobyeon/magpie2025:latest
```


# Dataset
### Extract Patches
Patch extraction methods differ based on stain types:

H&E and IHC Stains:
The patch extraction code for these stains is implemented based on the following paper:
Lu, Ming Y., et al. "Data-efficient and weakly supervised computational pathology on whole-slide images." Nature Biomedical Engineering 5.6 (2021): 555-570.

Special Stains:
For special stains, a custom patch extraction method was used.

To extract image patches from Whole Slide Images (WSIs), run the following command:
```bash
cd make_pathes
git clone https://github.com/mahmoodlab/CLAM
python make_pathes.py
cd ../
```

### Data Folder Structure
```
DATA_ROOT_DIR/  
    ├── train/  
    │   ├── blur/  
    │   │   └── Slide_1/  
    │   │       ├── 1um/  
    │   │       │   ├── Slide_1_4656_41216.png  
    │   │       │   ├── Slide_1_4656_41728.png  
    │   │       │   └── ...  
    │   │       ├── 2um/  
    │   │       │   └── ...  
    │   │       ├── 3um/  
    │   │       │   └── ...  
    │   │       └── 4um/  
    │   │           └── ...  
    │   └── gt/  
    │       └── Slide_1/  
    │           └── 0um/  
    │               ├── Slide_1_4656_41216.png  
    │               ├── Slide_1_4656_41728.png  
    │               └── ...
    └── val/
        ├── blur/
        └── gt/
```

# Sample inference code
### Installation
This sample inference code is implemented based on the following paper:
Yue, Zongsheng, Jianyi Wang, and Chen Change Loy. "Resshift: Efficient diffusion model for image super-resolution by residual shifting." Advances in Neural Information Processing Systems 36 (2024).
```bash
cd inference_sample
git clone https://github.com/zsyOAOA/ResShift
cd ../
```
Setup environment
```
/bin/bash setup.sh
```

### Run Docker Container
Before running the docker container, set your data path as follows:
```bash
export DATA_PATH=YOUR_DATA_PATH
```
Restart Docker and launch the container:
```bash
sudo systemctl restart docker
sudo docker run --gpus all --network=host --privileged \
-v .:/workspace \
-v "$DATA_PATH":/data \
-it magpie2025 \
/bin/bash
```

### Run Inference
Run inference inside the container:
```bash
python inference.py
```

### Retrieve Results
Exit the container:
```bash
exit
```
Copy the result files from the container:
```bash
docker cp magpie2025:/workspace/results ./results
```

# Evaluation code
See "evaluation.py" for more detail.

```python
import lpips

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

lpips_loss = lpips.LPIPS(net='vgg').cuda()


def calcualte_psnr(pred, gt):
    psnr_score = peak_signal_noise_ratio(pred, gt)
    return psnr_score


def calcualte_ssim(pred, gt):
    ssim_score = structural_similarity(pred, gt, channel_axis=2)
    return ssim_score


def calcualte_lpips(pred, gt):
    lpips_score = lpips_loss(lpips.im2tensor(pred).cuda(), lpips.im2tensor(gt).cuda()).item()
    return lpips_score
```
