import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os

from tqdm import tqdm
from models.unet import UNet
from models.resnet34_unet import ResNet34_Unet
from oxford_pet import load_dataset
from evaluate import evaluate
from utils import dice_score, dice_loss


def check_memory(threshold=90):
    total_memory = torch.cuda.get_device_properties(0).total_memory
    used_memory = torch.cuda.memory_allocated(0)
    usage_percentage = (used_memory / total_memory) * 100
    return usage_percentage > threshold

def train(args, model_name):
    # implement the training function here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs("saved_models", exist_ok=True)

    # Load dataset
    train_loader = load_dataset(args.data_path, "train", args)
    valid_loader = load_dataset(args.data_path, "valid", args)

    # Initialize model
    if model_name == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif model_name == 'resnet34_unet':
        model = ResNet34_Unet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Model {model_name} not available")

    model = model.to(device)

    # Loss function / optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_dice = 0.9274
    # training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        for batch in tqdm(train_loader, desc="Training", leave=False):
            images, masks = batch["image"].to(device, dtype=torch.float32), batch["mask"].to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = 0.3*criterion(outputs, masks) + 0.7*dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            with torch.no_grad():
                train_dice += dice_score(outputs, masks).item() 

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Dice Score: {avg_train_dice:.4f}")


        # validation loop
        avg_val_loss, avg_val_dice = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Validation Loss: {avg_val_loss:.4f}, Dice Score: {avg_val_dice:.4f}")
        # Save model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            save_path = f"saved_models/{model_name}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path} (Best Dice Score: {best_val_dice:.4f})")
        # Check memory usage
        if check_memory():
            print("Memory usage > 90%")
            break


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--model', type=str, choices=['unet', 'resnet34_unet'], required=True, help='Model to train')
    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args, args.model)