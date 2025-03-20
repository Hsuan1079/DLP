import os
import torch
import shutil
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}
        
        self.root = root
        self.mode = mode
        self.transform = transform
        self.gamma_threshold = 50 
        self.gamma_value = 1.5  

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        # apply gamma correction
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)

        if brightness < self.gamma_threshold:
            image = self.apply_gamma(image, gamma=self.gamma_value)
        elif brightness > 200:
            image = self.apply_gamma(image,gamma=0.6)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)
            sample["mask"] = sample["mask"].unsqueeze(0)
    
        return sample

    @staticmethod
    # Pixel Annotations: 1: Foreground 2:Background 3: Not classified
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames
    
    def apply_gamma(self, image, gamma=1.5):
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode, args):
    assert mode in {"train", "valid", "test"}

    # implement the load dataset function here
    train_t = A.Compose([
        A.Normalize(), 
        A.HorizontalFlip(p=0.25),  # apply horizontal flip to 50% of images
        A.VerticalFlip(p=0.25),  # apply vertical flip to 50% of images
        A.RandomRotate90(p=0.25),  # apply random rotation to 50% of images
        A.RandomBrightnessContrast(p=0.25),  # apply random brightness and contrast
        A.Resize(256, 256),
        ToTensorV2(),
    ], additional_targets={"trimap": "mask"}) 
    vaild_t = A.Compose([
        A.Normalize(), 
        A.Resize(256, 256),
        ToTensorV2(),
    ], additional_targets={"trimap": "mask"})

    transform = train_t if mode == "train" else vaild_t

    
    dataset = OxfordPetDataset(root=data_path, mode=mode, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True if mode == "train" else False,
        num_workers=2,
        pin_memory=True,
    )

    return dataloader

# if __name__ == "__main__":

#     data_path = "./dataset" 
#     OxfordPetDataset.download(data_path)
#     train_loader = load_dataset(data_path, "test")

#     cnt = 0
#     for batch in train_loader:
#         images, masks = batch["image"], batch["mask"]
#         print(f"Image shape: {images.shape}, Mask shape: {masks.shape}")
#         # 這邊檢查mask.shape是否為(1, 1, 256, 256) 如果不是 raise AssertionError
#         assert masks.shape == (1, 1, 256, 256)