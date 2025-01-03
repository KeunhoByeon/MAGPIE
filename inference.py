import argparse
import cv2
import os
import torch
from tqdm import tqdm

from inference_sample import get_model
from inference_sample.magpie_dataset import MagpieDataset


def inference(args, model, infer_loader):
    model.eval()

    eval_loop = tqdm(infer_loader, leave=False, total=len(infer_loader))
    with torch.no_grad():
        for input_paths, inputs in eval_loop:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = model(inputs)

            if torch.cuda.is_available():
                outputs = outputs.cpu()
            outputs = outputs.detach()
            outputs = outputs.numpy()

            for input_path, output in zip(input_paths, outputs):
                save_path = input_path.replace(args.data_dir, args.results_dir)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(output)


def run(args):
    model = get_model()
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()

    infer_loader = MagpieDataset(args.data_dir, mode='test')

    inference(args, model, infer_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAGPIE inference')
    # Model Arguments
    parser.add_argument('--model', default='resshift', type=str)
    parser.add_argument('--checkpoint', default="./inference_sample/model_400000.pth", type=str)
    # Data Arguments
    parser.add_argument('--data_dir', default="/data", type=str)
    parser.add_argument('--results_dir', default="/results", help='path to results')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')
    parser.add_argument('--mode', default='test')
    args = parser.parse_args()

    run(args)
