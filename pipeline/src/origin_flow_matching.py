"""
===============================================================================
HuMob 2026: Per-Origin Destination Map Flow Matching Model
===============================================================================
Architecture:
  - Sample: (Origin O, Date d) -> Residual Map Z in R^(1, 70, 100)
  - Source distribution: N(0, I)
  - Flow: Linear interpolation x_t = (1-t)*noise + t*x_0, vector field u = x_0 - noise
  - Condition: 8D vector [sin(2pi*wd/7), cos(2pi*wd/7), is_holiday, sin(2pi*month/12), cos(2pi*month/12), progression, ox/70, oy/100]
  - Sampling: Euler ODE (15~20 steps)
===============================================================================
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device) / (half - 1)
        )
        emb = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)

class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.gn1   = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film  = nn.Linear(emb_dim, out_ch * 2)
        self.gn2   = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.gn1(x)))
        scale, shift = self.film(emb).chunk(2, dim=-1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.gn2(h)))
        return h + self.skip(x)

class SpatialSelfAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.gn  = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.gn(x)
        qkv = self.qkv(h).reshape(B, 3, self.n_heads, C // self.n_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * (C // self.n_heads) ** -0.5
        attn = F.softmax(attn, dim=-1)
        out  = torch.einsum('bhnm,bhdm->bhdn', attn, v).reshape(B, C, H, W)
        return x + self.proj(out)

class OriginDestFlowUNet(nn.Module):
    def __init__(self, cond_dim: int = 8, base_ch: int = 32, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.t_emb = FlowTimeEmbedding(time_dim)
        self.c_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        emb_dim = time_dim

        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4

        self.pad_h = (1, 1)
        self.pad_w = (2, 2)

        self.init_conv = nn.Conv2d(1, ch1, 3, padding=1)
        self.enc1 = ResBlock2D(ch1, ch1, emb_dim)
        self.down1 = nn.Conv2d(ch1, ch2, 3, stride=2, padding=1)

        self.enc2 = ResBlock2D(ch2, ch2, emb_dim)
        self.down2 = nn.Conv2d(ch2, ch3, 3, stride=2, padding=1)

        self.mid1    = ResBlock2D(ch3, ch3, emb_dim)
        self.mid_att = SpatialSelfAttention(ch3, n_heads=4)
        self.mid2    = ResBlock2D(ch3, ch3, emb_dim)

        self.up2  = nn.ConvTranspose2d(ch3, ch2, 4, stride=2, padding=1)
        self.dec2 = ResBlock2D(ch2 + ch2, ch2, emb_dim)

        self.up1  = nn.ConvTranspose2d(ch2, ch1, 4, stride=2, padding=1)
        self.dec1 = ResBlock2D(ch1 + ch1, ch1, emb_dim)

        self.out_gn   = nn.GroupNorm(min(8, ch1), ch1)
        self.out_conv = nn.Conv2d(ch1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[2], x.shape[3]
        x = F.pad(x, (self.pad_w[0], self.pad_w[1], self.pad_h[0], self.pad_h[1]))

        emb = self.t_emb(t) + self.c_proj(c)

        h0 = self.init_conv(x)
        h1 = self.enc1(h0, emb)
        d1 = self.down1(h1)

        h2 = self.enc2(d1, emb)
        d2 = self.down2(h2)

        m = self.mid1(d2, emb)
        m = self.mid_att(m)
        m = self.mid2(m, emb)

        u2 = self.up2(m)
        u2 = self.dec2(torch.cat([u2, h2], dim=1), emb)

        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, h1], dim=1), emb)

        out = self.out_conv(F.silu(self.out_gn(u1)))
        out = out[:, :, self.pad_h[0]: self.pad_h[0] + orig_h,
                        self.pad_w[0]: self.pad_w[0] + orig_w]
        return out

class OriginFlowMatching:
    @staticmethod
    def get_xt_and_target(x0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x0)
        t4 = t[:, None, None, None]
        x_t = (1.0 - t4) * noise + t4 * x0
        target = x0 - noise
        return x_t, target

    @staticmethod
    @torch.no_grad()
    def sample(model: nn.Module, cond: torch.Tensor,
               shape: tuple, device: str, n_steps: int = 20) -> torch.Tensor:
        model.eval()
        x = torch.randn(shape, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_val = i / n_steps
            t_tensor = torch.full((shape[0],), t_val, device=device, dtype=torch.float32)
            v = model(x, t_tensor, cond)
            x = x + dt * v

        return x
