# MAGPIE2025

MAGPIE: MULTI-STAIN AND MULTI-ORGAN PATHOLOGY IMAGE RESTORATION CHALLENGE (2025)

# Challenge Docker Image
```bash
docker pull keunhobyeon/magpie2025:latest
```

# Sample inference code
Set data path
```bash
export DATA_PATH=YOUR_DATA_PATH
```
Run docker container
```bash
sudo docker run --gpus all --network=host --privileged \
-v .:/workspace \
-v "$DATA_PATH":/data \
-it magpie2025 \
/bin/bash
```
Run inference
```bash
python inference.py
```
Copy result files from container
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

# Data Folder Structure

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