import os
from PIL import Image
import matplotlib.pyplot as plt
import re

def natural_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]


def make_image_grid(image_paths, grid_size=(4, 8), image_size=(64, 64), save_path='grid_output.png'):
    assert len(image_paths) >= grid_size[0] * grid_size[1], "圖片數量不足以填滿網格"

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1]*2, grid_size[0]*2))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            img_index = i * grid_size[1] + j
            img = Image.open(image_paths[img_index]).resize(image_size)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ✅ 使用方式：放入圖片路徑列表
image_folder = 'images/new_test'  # 替換成你的圖片資料夾路徑
# image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
image_paths = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder), key=natural_key) if f.endswith('.png') or f.endswith('.jpg')]

make_image_grid(image_paths, grid_size=(4, 8), image_size=(64, 64), save_path='new_test.png')
