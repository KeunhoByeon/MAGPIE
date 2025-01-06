import argparse
import cv2
import lpips
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Initialize LPIPS loss with VGG backbone
lpips_loss = lpips.LPIPS(net='vgg').cuda()


# Evaluation functions
def calculate_psnr(pred, gt):
    return peak_signal_noise_ratio(pred, gt)


def calculate_ssim(pred, gt):
    return structural_similarity(pred, gt, channel_axis=2, multichannel=True)


def calculate_lpips(pred, gt):
    pred_tensor = lpips.im2tensor(pred).cuda()
    gt_tensor = lpips.im2tensor(gt).cuda()
    return lpips_loss(pred_tensor, gt_tensor).item()


def evaluate(gt_dir, pred_dir):
    psnr_list, ssim_list, lpips_list = [], [], []

    for slide in os.listdir(gt_dir):
        gt_slide_path = os.path.join(gt_dir, slide)
        pred_slide_path = os.path.join(pred_dir, slide)
        for layer in os.listdir(pred_slide_path):
            gt_layer_path = os.path.join(gt_slide_path, "0")
            pred_layer_path = os.path.join(pred_slide_path, layer) if os.path.exists(pred_slide_path) else None
            if not pred_layer_path:
                continue

            for filename in os.listdir(pred_layer_path):
                pred_path = os.path.join(pred_layer_path, filename)
                patch_name = "_".join(filename.split("_")[-2:])
                gt_path_list = os.listdir(gt_layer_path)
                for gt_filename in gt_path_list:
                    if patch_name in gt_filename:
                        gt_path = os.path.join(gt_layer_path, gt_filename)

                if not os.path.exists(pred_path):
                    continue

                pred = cv2.imread(pred_path)
                gt = cv2.imread(gt_path)

                # Compute metrics
                psnr = calculate_psnr(pred, gt)
                ssim = calculate_ssim(pred, gt)
                lpips_score = calculate_lpips(pred, gt)

                # Append results
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                lpips_list.append(lpips_score)

    # Calculate averages
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)

    return avg_psnr, avg_ssim, avg_lpips


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAGPIE evaluation')
    parser.add_argument('--gt_dir', default="/home/compu/jiwoo/challenge_data_sample/test/gt", type=str)
    parser.add_argument('--pred_dir', default="./results/test/blur", type=str)
    args = parser.parse_args()

    avg_psnr, avg_ssim, avg_lpips = evaluate(args.gt_dir, args.pred_dir)

    print(f'Average PSNR: {avg_psnr:.4f}')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Average LPIPS: {avg_lpips:.4f}')
