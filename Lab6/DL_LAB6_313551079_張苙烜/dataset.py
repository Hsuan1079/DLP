# dataset/clevr_dataset.py

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class ICLEVRDataset(Dataset):
    def __init__(self, img_root, json_path, object_json_path, mode='train'):
        """
        img_root: iclevr 資料夾路徑
        json_path: train.json 或 test.json 或 new_test.json 的路徑
        object_json_path: objects.json 的路徑
        mode: 'train' 或 'test'
        """
        self.img_root = img_root
        self.mode = mode

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        with open(object_json_path, 'r') as f:
            self.object_dict = json.load(f)  # e.g., {"gray cube": 0, ...}

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        if mode == 'train':
            # self.filenames = sorted(list(self.data.keys()))  # 訓練時有圖片檔名
            self.filenames = list(self.data.keys())
        else:
            self.conditions = self.data  # 測試時只有條件組合（list of objects）

    def __len__(self):
        if self.mode == 'train':
            return len(self.filenames)
        else:
            return len(self.conditions)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_name = self.filenames[idx]
            img_path = os.path.join(self.img_root, img_name)
            img = Image.open(img_path).convert('RGB')

            label_list = self.data[img_name]  # ["red cube", "gray cylinder", ...]
            
        else:
            # test.json or new_test.json
            img = torch.randn(3, 64, 64)  # 測試時沒有圖片，先放一個假的（sampling時才生成）
            label_list = self.conditions[idx]  # ["red sphere", "cyan cylinder", ...]

        label_tensor = self._labels_to_onehot(label_list)

        if self.mode == 'train':
            img = self.transform(img)
            # print(f"[DEBUG] {img_name} → {[obj for obj in label_list]}")
            # print(" → one-hot idx:", label_tensor.nonzero().squeeze().tolist())

        return img, label_tensor

    # 根據object.json的key，將labels轉成one-hot vector
    def _labels_to_onehot(self, labels):
        onehot = torch.zeros(len(self.object_dict))
        for obj_name in labels:
            idx = self.object_dict[obj_name]
            onehot[idx] = 1
        return onehot
    
if __name__ == "__main__":
    # 測試 train.json
    dataset = ICLEVRDataset(
        img_root='./iclevr',
        json_path='./train.json',
        object_json_path='./objects.json',
        mode='train'
    )
    print(len(dataset))
    img, label = dataset[0]
    print(img.shape, label.shape)

    # 測試 test.json
    dataset = ICLEVRDataset(
        img_root='./iclevr',
        json_path='./test.json',
        object_json_path='./objects.json',
        mode='test'
    )
    print(len(dataset))
    img, label = dataset[0]
    print(img.shape, label.shape)
    for i in range(5):
        img, label = dataset[i]
        print(f"Sample {i} one-hot label idx:", label.nonzero().squeeze().tolist())
        img_np = ((img.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        plt.imshow(img_np)
        plt.title(f"Index {i}")
        plt.show()
