"""Small conv VAE: 1-channel image → spatial latent → simple_image."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_STEM_CHANNELS = 16
_STAGE_CHANNELS = (32, 32, 48, 64, 64, 64)


def downsample_count(patch_size: int) -> int:
    if patch_size < 2 or patch_size & (patch_size - 1):
        raise ValueError(f"patch_size ({patch_size}) must be a power of 2 ≥ 2")
    return patch_size.bit_length() - 1


def latent_hw(image_size: int, patch_size: int) -> int:
    downsample_count(patch_size)
    if image_size < patch_size or image_size % patch_size != 0:
        raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")
    return image_size // patch_size


def stage_channels(n_down: int) -> tuple[int, ...]:
    if n_down < 1:
        raise ValueError(f"need at least one downsample, got {n_down}")
    if n_down <= len(_STAGE_CHANNELS):
        return _STAGE_CHANNELS[:n_down]
    return _STAGE_CHANNELS + (_STAGE_CHANNELS[-1],) * (n_down - len(_STAGE_CHANNELS))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.res = ResidualBlock(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.res(F.silu(self.norm(self.conv(x))))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.res = ResidualBlock(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.res(F.silu(self.norm(self.up(x))))


class FloorplanVAE(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int = 1,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        latent_hw(image_size, patch_size)
        downs_ch = stage_channels(downsample_count(patch_size))
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, _STEM_CHANNELS, 3, padding=1),
            nn.GroupNorm(8, _STEM_CHANNELS),
            nn.SiLU(),
            ResidualBlock(_STEM_CHANNELS),
        )
        downs: list[DownBlock] = []
        ch = _STEM_CHANNELS
        for out_ch in downs_ch:
            downs.append(DownBlock(ch, out_ch))
            ch = out_ch
        self.downs = nn.ModuleList(downs)
        self.to_mu = nn.Conv2d(ch, latent_channels, 1)
        self.to_logvar = nn.Conv2d(ch, latent_channels, 1)

        self.from_z = nn.Sequential(
            nn.Conv2d(latent_channels, ch, 3, padding=1),
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            ResidualBlock(ch),
        )
        ups: list[UpBlock] = []
        for out_ch in (*reversed(downs_ch[:-1]), _STEM_CHANNELS):
            ups.append(UpBlock(ch, out_ch))
            ch = out_ch
        self.ups = nn.ModuleList(ups)
        self.to_image = nn.Sequential(
            ResidualBlock(ch),
            nn.Conv2d(ch, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.stem(x)
        for down in self.downs:
            h = down(h)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        logvar = logvar.clamp(-30.0, 20.0)
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z: Tensor) -> Tensor:
        h = self.from_z(z)
        for up in self.ups:
            h = up(h)
        return self.to_image(h)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(
    recon: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    *,
    ink_weight: float,
    kl_beta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    weight = 1.0 + ink_weight * (1.0 - target)
    rec = (weight * (recon - target).abs()).mean()
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1, 2, 3)).mean()
    return rec + kl_beta * kl, rec, kl
