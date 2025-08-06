import torch
import torch.nn as nn
from mamba_ssm.models.v_mamba.v_mamba_block import VMambaBlock

class V_MambaBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.block = VMambaBlock(dim=dim, fused_add_norm=False)

    def forward(self, x):
        # Entrada esperada: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        x = self.block(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x
