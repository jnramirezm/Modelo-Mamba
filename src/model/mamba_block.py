import torch
import torch.nn as nn
from config import MAMBA_VARIANT

# Intentar usar mamba-ssm, si no está disponible usar implementación alternativa
try:
    if MAMBA_VARIANT == "simple":
        from mamba_ssm.modules.mamba_simple import Mamba
    elif MAMBA_VARIANT == "full":
        from mamba_ssm.modules.mamba import Mamba
    MAMBA_SSM_AVAILABLE = True
    print("✅ Usando mamba-ssm oficial")
except ImportError:
    print("⚠️  mamba-ssm no disponible, usando implementación alternativa")
    from model.mamba_alternative import MambaBlock as AlternativeMambaBlock
    MAMBA_SSM_AVAILABLE = False

if MAMBA_SSM_AVAILABLE:
    if MAMBA_VARIANT == "simple":
        class MambaBlock(nn.Module):
            def __init__(self, d_model=64):
                super().__init__()
                self.norm = nn.LayerNorm(d_model)
                self.mamba = Mamba(d_model)

            def forward(self, x):
                B, C, H, W = x.shape
                x = x.view(B, C, -1).permute(0, 2, 1)
                x = self.mamba(self.norm(x))
                x = x.permute(0, 2, 1).view(B, C, H, W)
                return x

    elif MAMBA_VARIANT == "full":
        class MambaBlock(nn.Module):
            def __init__(self, d_model=64):
                super().__init__()
                self.norm = nn.LayerNorm(d_model)
                self.mamba = Mamba(d_model)

            def forward(self, x):
                B, C, H, W = x.shape
                x = x.view(B, C, -1).permute(0, 2, 1)
                x = self.mamba(self.norm(x))
                x = x.permute(0, 2, 1).view(B, C, H, W)
                return x

        def forward(self, x):
            B, C, H, W = x.shape
            x = x.view(B, C, -1).permute(0, 2, 1)
            def forward(self, x):
                B, C, H, W = x.shape
                x = x.view(B, C, -1).permute(0, 2, 1)
                x = self.mamba(self.norm(x))
                x = x.permute(0, 2, 1).view(B, C, H, W)
                return x
    
    elif MAMBA_VARIANT == "v":
        from model.v_mamba.v_mamba_block import V_MambaBlock as MambaBlock
    else:
        raise ValueError(f"❌ Variante de Mamba no válida: {MAMBA_VARIANT}")

else:
    # Usar implementación alternativa
    MambaBlock = AlternativeMambaBlock
    print(f"✅ Usando MambaBlock alternativo para variante: {MAMBA_VARIANT}")
