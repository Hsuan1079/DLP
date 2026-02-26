import os
import argparse
import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from diffusers import UNet2DModel
from ddpm import ConditionalDDPM
from evaluator import evaluation_model
import json
from tqdm import tqdm

@torch.no_grad()
def generate_images(model, dataloader, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_images = []
    all_labels = []

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    for idx, (_, cond) in enumerate(tqdm(dataloader, desc="Generating images")):
        cond = cond.to(model.device)
        fake_img = model.p_sample_loop((1, 3, 64, 64), cond)

        img = fake_img[0].cpu()
        img_for_save = (img + 1) / 2
        img_for_save = torch.clamp(img_for_save, 0., 1.)
        save_image(img_for_save, os.path.join(save_dir, f"{idx}.png"))
        save_denoising_process(model, cond, save_dir+"/process", sample_id=idx)

        img_for_eval = normalize(img_for_save)
        all_images.append(img_for_eval.unsqueeze(0))
        all_labels.append(cond.cpu())

    all_images = torch.cat(all_images, dim=0).to(model.device)
    all_labels = torch.cat(all_labels, dim=0).to(model.device)
    return all_images, all_labels

def build_test_dataloader(json_path, object_json_path, batch_size=1):
    with open(json_path, "r") as f:
        test_data = json.load(f)
    with open(object_json_path, "r") as f:
        obj_list = json.load(f)

    conds = []
    for cond in test_data:
        vec = torch.zeros(len(obj_list))
        for obj in cond:
            if obj in obj_list:
                vec[obj_list[obj]] = 1.0
        conds.append(vec)

    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, conds):
            self.conds = conds

        def __len__(self):
            return len(self.conds)

        def __getitem__(self, idx):
            return torch.zeros(3, 64, 64), self.conds[idx]

    dataset = TestDataset(conds)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def save_denoising_process(model, condition, save_dir, sample_id, num_steps=10):
    os.makedirs(save_dir, exist_ok=True)
    shape = (1, 3, 64, 64)
    x = torch.randn(shape, device=model.device)
    timesteps = model.scheduler.config.num_train_timesteps
    interval = timesteps // num_steps
    snapshots = []

    for i in reversed(range(timesteps)):
        t = torch.full((shape[0],), i, device=model.device, dtype=torch.long)
        x = model.reverse_diffusion(x, t, condition)
        if i % interval == 0 or i == timesteps - 1:
            img = x[0].detach().cpu()
            img = (img + 1) / 2
            img = torch.clamp(img, 0., 1.)
            snapshots.append(img)

    grid = make_grid(snapshots, nrow=num_steps+1)
    save_path = os.path.join(save_dir, f"{sample_id}_denoise_grid.png")
    save_image(grid, save_path)
    print(f"Saved denoising grid to {save_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = UNet2DModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        class_embed_type="identity",
    )
    model = ConditionalDDPM(model=unet, cond_dim=24, timesteps=1000, device=device).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model = model.to(device)

    dataloader = build_test_dataloader(args.test_json, args.object_json_path, batch_size=1)

    image_tensor, label_tensor = generate_images(model, dataloader, args.output_dir)
    

    evaluator = evaluation_model()
    score = evaluator.eval(image_tensor, label_tensor)
    print(f"Evaluation Score: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--test_json', type=str, required=True, help="Path to test.json")
    parser.add_argument('--object_json_path', type=str, default='objects.json')
    parser.add_argument('--output_dir', type=str, default='test_outputs')
    args = parser.parse_args()

    main(args)
