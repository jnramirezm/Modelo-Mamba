"""
Implementación alternativa de Mamba basada en conceptos de State Space Models
Esta implementación no depende de mamba-ssm y funciona con PyTorch estándar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleMamba(nn.Module):
    """
    Implementación simplificada de Mamba basada en State Space Models
    Sin dependencias externas, solo PyTorch
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        
        # Proyecciones de entrada (como en Mamba original)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolución 1D para dependencias locales
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # Activación (SiLU)
        self.activation = nn.SiLU()
        
        # Parámetros del State Space Model para selective scan
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Parámetros aprendibles del SSM
        A_log = torch.log(torch.rand(self.d_inner, self.d_state))
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Proyección de salida
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        """
        x: (B, L, D) donde B=batch, L=sequence length, D=d_model
        """
        B, L, D = x.shape
        
        # Proyección de entrada con gating
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # cada uno (B, L, d_inner)
        
        # Convolución 1D (necesitamos transponer para conv1d)
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # (B, d_inner, L)
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # Activación
        x = self.activation(x)
        
        # Selective scan (el corazón de Mamba)
        y = self.selective_scan(x)
        
        # Aplicar gate (z)
        y = y * self.activation(z)
        
        # Proyección de salida
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(self, x):
        """
        Implementación simplificada pero funcional del selective scan
        """
        B, L, D = x.shape
        
        # Calcular delta (dt) - parámetro de tiempo
        delta = F.softplus(self.dt_proj(x))  # (B, L, d_inner)
        
        # Calcular B y C del SSM
        x_dbl = self.x_proj(x)  # (B, L, d_state*2)
        B_ssm, C_ssm = x_dbl.chunk(2, dim=-1)  # cada uno (B, L, d_state)
        
        # Parámetro A (transiciones de estado)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Implementación del scan secuencial
        y = torch.zeros_like(x)  # (B, L, d_inner)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        for i in range(L):
            # Parámetros en el tiempo i
            dt_i = delta[:, i, :]  # (B, d_inner)
            B_i = B_ssm[:, i, :]  # (B, d_state)
            C_i = C_ssm[:, i, :]  # (B, d_state)
            x_i = x[:, i, :]  # (B, d_inner)
            
            # Discretización del sistema continuo: h_new = exp(A*dt)*h + B*x*dt
            # Simplificación: usar aproximación de primer orden
            dA = 1 + A.unsqueeze(0) * dt_i.unsqueeze(-1)  # (B, d_inner, d_state)
            dB = B_i.unsqueeze(1) * x_i.unsqueeze(-1) * dt_i.unsqueeze(-1)  # (B, d_inner, d_state)
            
            # Actualizar estado oculto
            h = h * dA + dB
            
            # Calcular salida: y = C*h + D*x
            y_i = torch.sum(C_i.unsqueeze(1) * h, dim=-1) + self.D * x_i  # (B, d_inner)
            y[:, i, :] = y_i
        
        return y


class MambaBlock(nn.Module):
    """
    Bloque Mamba que se puede usar en lugar del original
    """
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SimpleMamba(d_model, d_state, d_conv, expand)
        
    def forward(self, x):
        """
        x: (B, C, H, W) - formato de imagen
        """
        B, C, H, W = x.shape
        
        # Convertir a formato de secuencia (B, L, D)
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        
        # Aplicar normalización y Mamba
        x = self.mamba(self.norm(x))
        
        # Convertir de vuelta a formato de imagen
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        return x


# Para compatibilidad con el código existente
class Mamba(SimpleMamba):
    """Alias para compatibilidad"""
    pass
