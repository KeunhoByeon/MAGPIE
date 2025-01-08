import argparse
import os

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from get_model import get_model
from magpie_dataset import MagpieDataset


def process_output(output):
    output = output.transpose(1, 2, 0)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def inference(args, model, infer_loader, mode):
    model.eval()

    eval_loop = tqdm(infer_loader, leave=False, total=len(infer_loader))
    with torch.no_grad():
        for input_paths, inputs in eval_loop:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                if mode != 'test':
                    targets = targets.cuda()

            outputs = model(inputs)

            outputs = outputs.detach()
            outputs = outputs.float()
            if torch.cuda.is_available():
                outputs = outputs.cpu()
            outputs = outputs * 0.5 + 0.5
            outputs = outputs.clamp(min=0.0, max=1.0)
            outputs = (outputs * 255.0).round()
            outputs = outputs.numpy()
            for output in outputs:
                output = process_output(output)
                save_path = input_paths.replace(args.data_dir, args.results_dir)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAGPIE inference')
    # Model Arguments
    parser.add_argument('--model', default='resshift', type=str)
    parser.add_argument('--config', default="./configs/deblur_gopro256.yaml", type=str)
    parser.add_argument('--ckpt', default="./weights/model_400000.pth", type=str)
    parser.add_argument('--encoder_ckpt', default="./weights/autoencoder_vq_f4.pth", type=str)
    # Data Arguments
    parser.add_argument('--data_dir', default="/data", type=str)
    parser.add_argument('--results_dir', default="./results", help='path to results')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')
    parser.add_argument('--mode', default='test')
    args = parser.parse_args()

    model = get_model(args.config, args.ckpt, args.encoder_ckpt)
    infer_dataset = MagpieDataset(args.data_dir, mode='test')
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    inference(args, model, infer_loader, mode='test')
