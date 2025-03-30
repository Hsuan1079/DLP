import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.writer = SummaryWriter("logs_c/")
        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def prepare_training():
        os.makedirs("checkpoints_c", exist_ok=True)

    def train_one_epoch(self, train_loader, epoch, args):
        losses = []
        self.model.train()
        for i, data in enumerate(tqdm(train_loader)):
            data = data.to(args.device)
            logits,z_indices = self.model(data)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
            loss.backward()
            losses.append(loss.item())
            
            self.optim.step()
            self.optim.zero_grad()

        epoch_loss = np.mean(losses)
        self.scheduler.step()
        self.writer.add_scalar("Loss/train", epoch_loss, epoch) 
        if epoch % 10 == 0:
            train_transformer.save_loss_plot(epoch)
        return epoch_loss

        
    def eval_one_epoch(self, val_loader, epoch, args):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                data = data.to(args.device)
                logits,z_indices = self.model(data)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
                losses.append(loss.item())
        
        epoch_loss = np.mean(losses)
        self.writer.add_scalar("Loss/val", epoch_loss, epoch)
        return epoch_loss
    
    def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

        
        # scheduler = None

        return optimizer,scheduler
    

    def save_loss_plot(self, epoch=None):
        os.makedirs("report_assets", exist_ok=True)
        epochs = list(range(1, len(self.train_losses)+1))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label='Train Loss', marker='o')
        plt.plot(epochs, self.val_losses, label='Validation Loss', marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Cross Entropy Loss")
        plt.title("Training & Validation Loss Curve")
        plt.grid(True)
        plt.legend()
        
        if epoch is not None:
            plt.savefig(f"report_assets/cosine_epoch_{epoch}.png")
        else:
            plt.savefig("report_assets/cosine.png")
        plt.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab3_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab3_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:2", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=10, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5: 
# Implement the training loop for the transformer model
    best_train_loss = np.inf
    best_val_loss = np.inf
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader,epoch,args)
        val_loss = train_transformer.eval_one_epoch(val_loader,epoch,args)

        # save loss
        train_transformer.train_losses.append(train_loss)
        train_transformer.val_losses.append(val_loss)

        if epoch % args.save_per_epoch == 0:
            torch.save(train_transformer.model.transformer.state_dict(), f"checkpoints_c/epoch_{epoch}_cosine.pt")
            print(f"Saved checkpoint at epoch {epoch}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"checkpoints_c/best_train_loss_cosine.pt")
            print(f"Saved best train loss checkpoint at epoch {epoch}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"checkpoints_c/best_val_loss_cosine.pt")
            print(f"Saved best val loss checkpoint at epoch {epoch}")
    train_transformer.save_loss_plot()