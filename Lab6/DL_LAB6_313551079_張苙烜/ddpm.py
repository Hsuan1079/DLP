# 參考 Hugging Face Diffusers 的 UNet2DModel
# https://huggingface.co/learn/diffusion-course/unit1/3

from diffusers import UNet2DModel, DDPMScheduler
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from evaluator import evaluation_model

class ConditionalDDPM(nn.Module):
    def __init__(self, model, cond_dim,timesteps=1000, device='cuda'):
        super().__init__()
        self.model = model
        self.scheduler = DDPMScheduler(timesteps)
        self.evaluator = evaluation_model()
        self.device = device
        self.timesteps = timesteps
        print(self.scheduler.config)

        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

        self.cond_proj = nn.Linear(cond_dim,256)

    def sample_noise(self, shape):
        return torch.randn(shape, device=self.device)

    def forward_diffusion(self, x0, t):

        noise = self.sample_noise(x0.shape)
        x_t = self.scheduler.add_noise(x0, noise, t)
        return x_t, noise

    def predict_noise(self, x_t, t, condition):

        # 條件嵌入
        emb = self.cond_proj(condition)  # (B, in_channels)
        out = self.model(x_t, t, emb).sample
        return out

    def reverse_diffusion(self, x_t, t, condition, guidance_scale=1.0):

        pred_noise = self.predict_noise(x_t, t, condition)
        
        if t[0].item() % 5 == 0:
            with torch.enable_grad():
                x_t.requires_grad_(True)  # 允許對 x_t 計算梯度
                logits = self.evaluator.resnet18(x_t)  # 使用 evaluator 的分類器
                target_class = condition.argmax(dim=-1)  # 假設條件是 One-Hot 編碼
                loss = -logits[range(len(target_class)), target_class].sum()  # 對目標類別的負損失
                grad = torch.autograd.grad(loss, x_t)[0]  # 計算梯度
                x_t = x_t - guidance_scale * grad  # 使用梯度調整 x_t
                x_t = x_t.detach()  # 分離計算圖

        x_prev = self.scheduler.step(pred_noise, t, x_t).prev_sample
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, shape, condition):

        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.scheduler.config.num_train_timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.reverse_diffusion(x, t, condition)
        return x

# if __name__ == "__main__":
#     # 初始化 UNet 和 Scheduler
#     unet = UNet2DModel(
#         sample_size=64,
#         in_channels=3,
#         out_channels=3,
#         layers_per_block=2,
#         block_out_channels=(64, 128, 256, 512),
#         down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
#         up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
#     )
#     scheduler = DDPMScheduler(num_train_timesteps=1000)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 建立 ConditionalDDPM
#     model = ConditionalDDPM(model=unet, cond_dim=24, scheduler=scheduler, device=device)
#     model.to(model.device)

#     # 測試 forward diffusion
#     x0 = torch.randn(1, 3, 64, 64).to(model.device)
#     t = torch.randint(0, scheduler.num_train_timesteps, (1,), device=model.device)
#     x_t, noise = model.forward_diffusion(x0, t)

#     # 測試生成一張圖片
#     condition = torch.randint(0, 2, (1, 24), dtype=torch.float32, device=model.device)  # 假設一個隨機條件
#     generated_img = model.p_sample_loop((1, 3, 64, 64), condition)
#     print(generated_img.shape)  # (1, 3, 64, 64)
#     print(torch.isnan(generated_img).any())  # 是否有 NaN？
#     print(torch.isinf(generated_img).any())  # 是否有 Inf？
#     # save image
#     img = generated_img[0].cpu()  # 拿出一張 (3, 64, 64)

#     # 步驟1: [-1,1] 轉 [0,1]
#     img = (img + 1) / 2

#     # 步驟2: clamp保護
#     img = torch.clamp(img, 0., 1.)

#     # 步驟3: 轉PIL
#     transform = transforms.ToPILImage()
#     img_pil = transform(img)

#     # 步驟4: 儲存
#     img_pil.save("generated_image.png")