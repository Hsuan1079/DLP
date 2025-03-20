import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

from models.unet import UNet
from models.resnet34_unet import ResNet34_Unet
from oxford_pet import load_dataset
from utils import plot_comparison
from evaluate import evaluate

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images and save results')
    parser.add_argument('--model_path', required=True, help='Path to the trained model (.pth file)')
    parser.add_argument('--model_type', type=str, choices=['unet', 'resnet34_unet'], required=True, help='Model architecture')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input dataset')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default="output", help='Directory to save predicted masks and comparisons')
    return parser.parse_args()

def load_model(model_path, model_type, device):
    if model_type == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif model_type == "resnet34_unet":
        model = ResNet34_Unet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args.model_type, device)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    predictions_dir = os.path.join(args.output_dir, "predictions")
    comparisons_dir = os.path.join(args.output_dir, "comparisons_u")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # Load dataset (Test Mode)
    test_loader = load_dataset(args.data_path, "test", args)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Running Inference")):
            images = batch["image"].to(device, dtype=torch.float32)
            gt_masks = batch["mask"].to(device, dtype=torch.float32)

            outputs = model(images)

            # Sigmoid activation and thresholding
            outputs = torch.sigmoid(outputs)  
            preds = (outputs > 0.5).float()  # turn to 0 or 1

            # save predictions
            # pred_save_path = os.path.join(predictions_dir, f"prediction_{i}.png")
            # save_image(preds, pred_save_path)

            # save comparisons
            # comparison_save_path = os.path.join(comparisons_dir, f"comparison_{i}.png")
            # plot_comparison(images[0], gt_masks[0], preds[0], comparison_save_path)  
            # print(f"Comparison saved: {comparison_save_path}")
    # evaluate
    avg_loss, avg_dice = evaluate(model, test_loader, device)
    print(f"Test - Loss: {avg_loss:.4f}, Dice Score: {avg_dice:.4f}")

if __name__ == '__main__':
    args = get_args()
    predict(args)
