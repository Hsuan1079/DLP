import torch
import matplotlib.pyplot as plt


def dice_score(pred_mask, gt_mask, threshold=0.5):
    pred_mask = torch.sigmoid(pred_mask) 
    pred_mask = (pred_mask > threshold).float()  
    
    intersection = torch.sum(pred_mask * gt_mask, dim=[2, 3])
    union = torch.sum(pred_mask, dim=[2, 3]) + torch.sum(gt_mask, dim=[2, 3])

    dice = (2.0 * intersection + 1e-6) / (union + 1e-6) 
    return dice.mean()

def dice_loss(pred_mask, gt_mask, eps=1e-6):
    pred_mask = torch.sigmoid(pred_mask)  
    
    intersection = torch.sum(pred_mask * gt_mask, dim=[2, 3])  
    union = torch.sum(pred_mask, dim=[2, 3]) + torch.sum(gt_mask, dim=[2, 3])  
    dice = (2.0 * intersection + eps) / (union + eps) 
    dice_loss = 1 - dice  
    
    return dice_loss.mean()  

def denormalize_image(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    image = image.clone().detach().cpu()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)  # 反標準化
    return torch.clamp(image, 0, 1)  # 限制範圍 0~1

def plot_comparison(image, gt_mask, pred_mask, save_path):

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # tensor → numpy
    raw_image = denormalize_image(image)  
    image = raw_image.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
    gt_mask = gt_mask.cpu().numpy().squeeze()  # (1, H, W) → (H, W)
    pred_mask = pred_mask.cpu().numpy().squeeze()  # (1, H, W) → (H, W)

    # plot
    ax[0].imshow(image)
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    ax[1].imshow(gt_mask, cmap="gray")
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis("off")

    ax[2].imshow(pred_mask, cmap="gray")
    ax[2].set_title("Predicted Mask")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

