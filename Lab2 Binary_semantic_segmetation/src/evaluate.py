import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import dice_score, dice_loss

def evaluate(model, dataloader, device):
    model.eval()
    dice_total = 0
    loss_total = 0
    criterion = torch.nn.BCEWithLogitsLoss()  

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images, masks = batch["image"].to(device), batch["mask"].to(device)

            outputs = model(images)
            loss = 0.3*criterion(outputs, masks)+0.7*dice_loss(outputs, masks)  # BCE Loss + Dice Loss

            dice_total += dice_score(outputs, masks).item()
            loss_total += loss.item()

    avg_dice = dice_total / len(dataloader)
    avg_loss = loss_total / len(dataloader)

    # print(f"Validation - Loss: {avg_loss:.4f}, Dice Score: {avg_dice:.4f}")
    return avg_loss, avg_dice
