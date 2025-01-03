import cv2
import os
import torchvision
from torch.utils.data import Dataset


class MagpieDataset(Dataset):
    def __init__(self, root_dir, mode='test'):
        self.root_dir = root_dir
        self.mode = mode
        assert mode in ['train', 'val', 'test']

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=256),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.5),
        ])

        # Load gt images (except for test mode)
        if mode != 'test':
            self.gt_paths = {}
            gt_dir = os.path.join(root_dir, mode, 'gt')
            for path, dir, files in os.walk(gt_dir):
                for filename in files:
                    ext = os.path.splitext(filename)[-1]
                    if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                        continue
                    self.gt_paths[filename] = os.path.join(path, filename)

        # Load blur images from all levels
        self.samples = []
        blur_dir = os.path.join(root_dir, mode, 'blur')
        for path, dir, files in os.walk(blur_dir):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                image_path = os.path.join(path, filename)
                if mode != 'test':
                    assert filename in self.gt_paths
                self.samples.append(image_path)

        print("Loaded {} samples.".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        if self.mode == 'test':
            return image_path, image
        else:
            filename = os.path.basename(image_path)
            gt_path = self.gt_paths[filename]
            gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gt = self.transform(gt)
            return image_path, gt_path, image, gt
