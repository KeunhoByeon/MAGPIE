import argparse
import os
import sys
from tempfile import NamedTemporaryFile

import cv2
import h5py
import numpy as np
import openslide
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Importing from CLAM for segmentation and patching
sys.path.append(os.path.join(os.path.dirname(__file__), 'CLAM'))
from CLAM.create_patches_fp import seg_and_patch


# Dataset class for patch extraction
class PatchDataset(Dataset):
    def __init__(self, coordinates, wsi_path):
        self.coordinates = coordinates
        self.wsi_path = wsi_path

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        x, y = self.coordinates[idx]
        return x, y, self.wsi_path


# Function to generate h5 files and extract patches
def generate_h5_and_patches(source_dir, save_dir, tile_size, resize_size, batch_size, gpu_ids):
    print("Starting the process: Generating h5 files and extracting patches")

    # Create CSV file to track progress
    process_list_csv = os.path.join(save_dir, "process_list.csv")

    # Check if CSV file exists, if not create a new one
    if not os.path.exists(process_list_csv) or os.stat(process_list_csv).st_size == 0:
        print(f"Creating new CSV file at {process_list_csv}")
        process_list = pd.DataFrame(columns=["slide_id", "offset", "status"])
    else:
        print(f"Loading existing CSV file from {process_list_csv}")
        process_list = pd.read_csv(process_list_csv)

    world_size = len(gpu_ids)
    mp.spawn(worker, args=(world_size, gpu_ids, source_dir, save_dir, tile_size, resize_size, batch_size, process_list_csv), nprocs=world_size, join=True)


# Worker function for multiprocessing
def worker(rank, world_size, gpu_ids, source_dir, save_dir, tile_size, resize_size, batch_size, process_list_csv):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12358'

    gpu_id = gpu_ids[rank]
    torch.cuda.set_device(gpu_id)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print(f"[Rank {rank}] Using device: {torch.cuda.get_device_name(gpu_id)}")

    if os.path.exists(process_list_csv):
        process_list = pd.read_csv(process_list_csv)
    else:
        print(f"[Rank {rank}] CSV file not found. Exiting...")
        return

    for subset in ["train", "val"]:
        gt_dir = os.path.join(source_dir, subset, "gt")
        blur_dir = os.path.join(source_dir, subset, "blur")

        if not os.path.exists(gt_dir):
            print(f"[Rank {rank}] Directory not found: {gt_dir}")
            continue

        # Ensure necessary directories exist
        os.makedirs(os.path.join(save_dir, subset, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, subset, 'patches'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, subset, 'stitches'), exist_ok=True)

        seg_and_patch(
            source=gt_dir,
            save_dir=save_dir,
            patch_save_dir=os.path.join(save_dir, subset, 'patches'),
            mask_save_dir=os.path.join(save_dir, subset, 'masks'),
            stitch_save_dir=os.path.join(save_dir, subset, 'stitches'),
            patch_size=tile_size[0],
            step_size=tile_size[0],
            seg=True, patch=True, stitch=True,
            patch_level=0
        )

        h5_files_dir = os.path.join(save_dir, subset, "patches")
        if not os.path.exists(h5_files_dir):
            print(f"[Rank {rank}] Directory {h5_files_dir} not found. Skipping...")
            continue

        h5_files = [os.path.join(h5_files_dir, f) for f in os.listdir(h5_files_dir) if f.endswith(".h5")]

        for h5_file in h5_files:
            slide_id = os.path.basename(h5_file).replace("_OFS-0.h5", "")
            coordinates = load_coordinates_from_h5(h5_file)

            for offset in range(5):
                slide_path = os.path.join(blur_dir if offset > 0 else gt_dir, f"{slide_id}_OFS-{offset}.svs")
                if not os.path.exists(slide_path):
                    continue

                slide_filename = f"{slide_id}_OFS-{offset}.svs"
                if ((process_list["slide_id"] == slide_filename) & (process_list["offset"] == f"{offset}um") & (process_list["status"] == "processed")).any():
                    print(f"[Rank {rank}] {slide_filename} already processed. Skipping...")
                    continue

                print(f"[Rank {rank}] Processing {slide_filename}...")
                process_slide(slide_path, coordinates, save_dir, slide_id, f"{offset}um", tile_size, resize_size, rank, subset, batch_size)

                new_row = pd.DataFrame({
                    "slide_id": [slide_filename],
                    "offset": [f"{offset}um"],
                    "status": ["processed"]
                })
                process_list = pd.concat([process_list, new_row], ignore_index=True)
                process_list.to_csv(process_list_csv, index=False)

    dist.destroy_process_group()


# Process individual slide
def process_slide(slide_path, coordinates, save_dir, slide_id, offset, tile_size, resize_size, rank, subset, batch_size):
    slide_output_dir = os.path.join(save_dir, subset, "blur" if "blur" in slide_path else "gt", slide_id, offset)
    os.makedirs(slide_output_dir, exist_ok=True)

    if not os.path.exists(slide_path):
        print(f"[Rank {rank}] Slide not found: {slide_path}")
        return

    dataset = PatchDataset(coordinates, slide_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with tqdm(total=len(dataloader), desc=f"[GPU {rank}] Processing {slide_id} {offset}", position=rank) as pbar:
        for batch in dataloader:
            batch_x, batch_y, wsi_path = batch
            crop_and_save_batch(wsi_path[0], batch_x, batch_y, slide_output_dir, slide_id, offset, tile_size, resize_size)
            pbar.update(1)


# Crop and save batch
def crop_and_save_batch(wsi_path, batch_x, batch_y, output_dir, slide_id, offset, tile_size=(512, 512), resize_size=(256, 256)):
    slide = openslide.OpenSlide(wsi_path)
    for x, y in zip(batch_x, batch_y):
        with NamedTemporaryFile(delete=False, suffix=".png", dir=output_dir) as tmp_file:
            tile = slide.read_region((x.item(), y.item()), 0, tile_size)
            tile = np.array(tile)[:, :, :3]
            tile_resized = cv2.resize(tile, resize_size)
            tmp_file_path = tmp_file.name
            output_path = os.path.join(output_dir, f"{slide_id}_{x.item()}_{y.item()}.png")
            cv2.imwrite(tmp_file_path, cv2.cvtColor(tile_resized, cv2.COLOR_RGB2BGR))
            os.rename(tmp_file_path, output_path)


# Load coordinates from h5
def load_coordinates_from_h5(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        coordinates = np.array(h5_file['coords'])
    return coordinates


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAGPIE make patches')
    parser.add_argument('--source_dir', default="", type=str)
    parser.add_argument('--save_dir', default="", type=str)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--tile_size', default=(512, 512), type=tuple)
    parser.add_argument('--resize_size', default=(256, 256), type=tuple)
    args = parser.parse_args()

    generate_h5_and_patches(args.source_dir, args.save_dir, args.tile_size, args.resize_size, args.batch_size, args.gpus)
