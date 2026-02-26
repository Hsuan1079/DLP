
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import wandb
from diffusers import UNet2DModel
from dataset import ICLEVRDataset
from ddpm import ConditionalDDPM
from evaluator import evaluation_model


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset & dataloader
    dataset = ICLEVRDataset(
        img_root=args.img_root,
        json_path=args.json_path,
        object_json_path=args.object_json_path,
        mode='train'
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # model
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
    model = ConditionalDDPM(unet, cond_dim=24, timesteps=1000, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # initialize wandb
    wandb.init(project="lab6-ddpm", config=vars(args))
    best_loss = float('inf')
    best_model_path = os.path.join(args.save_dir, "best_model.pt")

    for epoch in range(args.epochs):

        model.train()
        pbar = tqdm(dataloader)
        total_loss = 0.0

        for step, (x0, cond) in enumerate(pbar):
            x0 = x0.to(device)
            cond = cond.to(device)

            t = torch.randint(0, model.timesteps, (x0.size(0),), device=device).long()
            x_t, noise = model.forward_diffusion(x0, t)

            pred_noise = model.predict_noise(x_t, t, cond)
        
            loss = criterion(pred_noise, noise)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch+1} Step {step+1} Loss: {loss.item():.4f}")
            wandb.log({"train/loss": loss.item()}, step=epoch)
        
        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch/loss_avg": total_loss / len(dataloader), "epoch": epoch + 1}, step=epoch)
        os.makedirs(args.save_dir, exist_ok=True)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1} with loss {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"ddpm_epoch{epoch+1}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, default='./iclevr')
    parser.add_argument('--json_path', type=str, default='./train.json')
    parser.add_argument('--object_json_path', type=str, default='./objects.json')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_new')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    train(args)
