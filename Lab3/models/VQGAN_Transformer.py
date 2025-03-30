import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, _ = self.vqgan.encode(x)
        return codebook_indices.reshape(codebook_mapping.shape[0], -1)
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda ratio: 1 - ratio
        elif mode == "cosine":
            return lambda ratio: np.cos(ratio * np.pi / 2)
        elif mode == "square":
            return lambda ratio: 1 - ratio ** 2
        else:
            raise NotImplementedError

#TODO2 step1-3:            
    def forward(self, x):
        # step1: encode the input image to latent space z
        z_indices = self.encode_to_z(x)

        B, N = z_indices.shape
        device = z_indices.device
        # step2: initialize the mask
        # r = torch.rand(1).item()
        # num_mask = int(r * N) # number of mask tokens

        # mask = torch.zeros_like(z_indices, dtype=torch.bool) # 0: not masked, 1: masked inital don't mask any token
        # for i in range(B):
        #     perm = torch.randperm(N)
        #     mask[i, perm[:num_mask]] = True
        r = torch.rand(B)
        num_mask = (r * N).long()

        mask = torch.zeros_like(z_indices, dtype=torch.bool)
        for i in range(B):
            perm = torch.randperm(N)
            mask[i, perm[:num_mask[i]]] = True

        # step3: replace the mask token with the mask token id
        z_masked = z_indices.clone() # copy the z_indices
        z_masked[mask] = self.mask_token_id # mask token id

        # step4: feed the masked z to transformer
        logits = self.transformer(z_masked) # fget the probability of tokens

        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_b, step, total_iter):
        """
        z_indices:   tensor of shape (B, 256) → 部分位置是 token，部分是 mask token
        mask_b:      tensor of shape (B, 256) → bool，True 表示這個位置被遮住
        step:        第幾輪 iterative decoding（從 0 開始）
        total_iter:  總共 decoding 幾次
        """ 
        B,N = z_indices.shape
        device = z_indices.device
        # predict the token probability
        z_indices_input = z_indices.clone()
        z_indices_input[mask_b] = self.mask_token_id
        logits = self.transformer(z_indices_input)

        print(f"Step {step}, logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")

        probs = torch.softmax(logits, dim=-1) # shape: (B, 256, 512)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = probs.max(dim=-1) 

        # confidence scheduling
        ratio= (step+1) / total_iter
        ratio = self.gamma(ratio)
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = - torch.empty_like(z_indices_predict_prob).exponential_().log() # gumbel noise(0,1)
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        print(confidence.mean())
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        ##At the end of the decoding process, add back the original(non-masked) token values
        
        mask_bc= mask_b.clone()
        confidence[~mask_bc] = float('inf') # dont change the non-masked token

        mask_counts = mask_bc.sum(dim=1).float()  # 每筆的 mask 總數
        num_remain = (ratio * mask_counts).long()   # 每筆應保留更新的數量

        _, rank = torch.topk(confidence,k=confidence.shape[1], dim=-1, largest=False)

        new_mask = torch.zeros_like(mask_bc)
        for i in range(B):
            k = num_remain[i].item()
            if k > 0:
                keep = rank[i, :k]
                new_mask[i, keep] = True
                
        new_z_indices = z_indices.clone()
        update_mask = ~(new_mask) & mask_bc
        new_z_indices[update_mask] = z_indices_predict[update_mask]

        return new_z_indices, new_mask
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
