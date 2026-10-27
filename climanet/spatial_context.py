"""Hourly-level spatial context for SpatioTemporalModel.

The target is  monthly - mean_gapped  per pixel. Because the input anomalies are
relative to each pixel's own gapped mean, they sum to zero over observed hours, so

    mean_full - mean_gapped  ~=  (1 / n_valid) * sum_{missing hours t} fill_t

where fill_t is the neighbor-based anomaly estimate from NeighborFill. That is a
parameter-free, pixel-resolution estimate of the main (sampling-error) part of the
target. We feed it, plus per-pixel gap/diurnal statistics, straight into the
decoder at full resolution, and make the decoder predict a correction on top of it.
Patch tokens now only supply context, not pixel detail.

"""

import torch
from torch import nn


class NeighborFill(torch.nn.Module):
    """What do the neighbors say about this pixel at this hour.

    For each hour-frame: fill = G*(anom * obs) / G*(obs), conf = G*(obs).

    Land is never counted as observed (in the model, land is not "missing", and
    would otherwise inject zeros near coasts). Output: (B*M, 2*S, T, H, W),
    channel order [fill_s0, conf_s0, fill_s1, conf_s1, ...].
    """

    def __init__(self, sigmas=(2.0, 5.0), anom_scale=0.1, chunk=16384):
        """Initialize the NeighborFill module.

        Args:
            sigmas (tuple, optional): Standard deviations for the Gaussian kernels. Defaults to (2.0, 5.0).
            anom_scale (float, optional): Scaling factor for anomalies. Defaults
            to 0.1. This can be calculated from the input data using the
            `compute_anom_scale` function.
            chunk (int, optional): Chunk size for processing large inputs. Defaults to 16384.
        """
        super().__init__()

        self.anom_scale = anom_scale
        self.chunk = chunk
        self.radii = [int(3 * sigma) for sigma in sigmas]

        # Precompute Gaussian kernels for each sigma and radius
        for i, (sigma, rad) in enumerate(zip(sigmas, self.radii)):
            ax = torch.arange(-rad, rad + 1).float()
            gaussian_kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
            self.register_buffer(
                f"gaussian_kernel{i}", gaussian_kernel / gaussian_kernel.sum(), persistent=False
            )

        self.n_sigmas = len(sigmas)

    def _blur(self, input_tensor, g, r):
        """Separable depthwise blur with Gaussian kernel g and radius r."""
        kernel = g.view(1, 1, 1, -1).to(input_tensor.dtype).expand(2, 1, 1, -1)

        # Apply separable depthwise convolution with the Gaussian kernel
        input_tensor = torch.nn.functional.conv2d(input_tensor, kernel, padding=(0, r), groups=2)
        return torch.nn.functional.conv2d(input_tensor, kernel.transpose(2, 3), padding=(r, 0), groups=2)

    @torch.no_grad()
    def forward(self, x_norm, mask, land):
        """Forward pass for the NeighborFill module.

        Args:
            x_norm (torch.Tensor): Normalized input anomalies of shape (B, 1, M, T, H, W).
            mask (torch.Tensor): Boolean mask indicating missing values, shape (B, 1, M, T, H, W).
            land (torch.Tensor): Boolean mask indicating land pixels, shape (B, H, W).

        Returns:
            torch.Tensor: Filled and confidence maps of shape (B*M, 2*S, T, H, W).
        """
        B, _, M, T, H, W = x_norm.shape
        ocean = ~land
        obs = (~mask[:, 0]) & ocean[:, None, None] # (B,M,T,H,W)
        obs_mask = obs.reshape(B * M * T, 1, H, W).to(x_norm.dtype)
        valid = x_norm[:, 0].reshape(B * M * T, 1, H, W) * obs_mask
        features = torch.cat([valid, obs_mask], dim=1)  # (N,2,H,W)

        # Process features in chunks to save memory
        outs = []
        for chunk in features.split(self.chunk):
            per_sigma = []
            for i in range(self.n_sigmas):
                # Apply Gaussian blur for the current sigma
                blurred = self._blur(chunk, getattr(self, f"gaussian_kernel{i}"), self.radii[i])
                num, den = blurred[:, :1], blurred[:, 1:]  # Separate them into numerator and denominator

                # Compute the filled values for the current sigma using the normalized blur results
                fill = torch.where(den > 1e-3, num / den.clamp(min=1e-3), torch.zeros_like(num))
                per_sigma += [fill / self.anom_scale, den]
            outs.append(torch.cat(per_sigma, dim=1))
        out = torch.cat(outs).view(B * M, T, 2 * self.n_sigmas, H, W).permute(0, 2, 1, 3, 4)

        ocean_bm = ocean[:, None, None].to(out.dtype).repeat_interleave(M, dim=0)
        return (out * ocean_bm).contiguous()  # (B*M,1,1,H,W)


def token_validity(mask, land, patch_size):
    """Fraction of ocean pixels observed in each token: (B, M, Tp, Hp, Wp) in [0,1]."""
    B, _, M, T, H, W = mask.shape
    pt, ph, pw = patch_size

    ocean = (~land).float()

    obs = (~mask[:, 0]).float() * ocean[:, None, None]
    obs_p = torch.nn.functional.avg_pool3d(obs.reshape(B * M, 1, T, H, W), (pt, ph, pw), (pt, ph, pw))
    oc_p = torch.nn.functional.avg_pool2d(ocean[:, None], (ph, pw))
    oc_p = oc_p.repeat_interleave(M, dim=0)[:, :, None]
    valid = (obs_p / oc_p.clamp(min=1e-6)).clamp(max=1.0)
    return valid.squeeze(1).reshape(B, M, *valid.shape[2:])


class MaskedSpatialMix(torch.nn.Module):
    """Masked spatial mixing module for latent tokens."""
    def __init__(self, embed_dim=128, dilations=(1, 2), kernel_size=3):
        """Initialize the masked spatial mixing module.
        Args:
            embed_dim (int): Dimension of the embedding.
            dilations (tuple): Dilation rates for the depthwise convolutions.
            kernel_size (int): Kernel size for the depthwise convolutions.
        """
        super().__init__()
        self.kernel_size, self.dilations = kernel_size, dilations
        self.norm1 = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in dilations)
        self.dw = nn.ModuleList(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size,
                padding=d * (kernel_size // 2),
                dilation=d,
                groups=embed_dim,
                bias=False
            )
            for d in dilations
        )
        self.proj = nn.ModuleList(nn.Linear(embed_dim, embed_dim) for _ in dilations)
        zero_init = [self.proj]
        for layers in zero_init:
            for p in layers:
                nn.init.zeros_(p.weight)
                nn.init.zeros_(p.bias)

    def forward(self, latent, tok_valid):
        """Forward pass for the masked spatial mixing module.

        Args:
            latent (torch.Tensor): Input latent tokens of shape (B, M, Tp, Hp, Wp, E).
            tok_valid (torch.Tensor): Token validity mask of shape (B, M, Tp, Hp, Wp).

        Returns:
            torch.Tensor: Output latent tokens of shape (B, M, Tp, Hp, Wp, E).
        """
        B, M, Tp, Hp, Wp, E = latent.shape
        N = B * M * Tp

        features = latent.reshape(N, Hp, Wp, E)
        weight = tok_valid.reshape(N, 1, Hp, Wp).to(features.dtype)
        ones = weight.new_ones(1, 1, self.kernel_size, self.kernel_size)

        for i, d in enumerate(self.dilations):
            h = self.norm1[i](features).permute(0, 3, 1, 2)
            nb = self.dw[i](h * weight)
            cnt = torch.nn.functional.conv2d(weight, ones, padding=d * (self.kernel_size // 2), dilation=d)
            nb = nb * (self.kernel_size * self.kernel_size) / cnt.clamp(min=1.0)
            features = features + self.proj[i](nb.permute(0, 2, 3, 1))
        return features.view(B, M, Tp, Hp, Wp, E)


class PatchStats(torch.nn.Module):
    """Explicit gap/diurnal statistics per patch, computed from data + mask.

    Per pixel and hour-of-day bin (24 bins):
      - share of observations falling in that bin (the sampling distribution)
      - mean anomaly of that bin relative to the pixel's observed mean (diurnal shape)
    Plus the overall observed fraction. Then pooled to the patch grid (ocean only)
    and projected to embed_dim.
    """

    def __init__(self, embed_dim, patch_size, anom_scale=0.1):
        """
        Args:
            embed_dim (int): Dimension of the embedding.
            patch_size (tuple): Size of the spatial patch (C, H, W).
            anom_scale (float, optional): Scaling factor for anomalies. Defaults to 0.1.
                This is used to scale the anomalies computed from the input data
                using the `compute_anom_scale` function.
        """
        super().__init__()

        self.ph, self.pw = patch_size[1], patch_size[2]
        self.anom_scale = anom_scale

        in_dim = 24 + 24 + 1  # 24 bin anomalies + 24 bin shares + overall fraction
        hidden = 2 * embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x_norm, mask, land, padded_days_mask):
        """
        Args:
            x_norm (torch.Tensor): Normalized input data of shape (B, 1, M, T, H, W).
            mask (torch.Tensor): Mask indicating missing values, same shape as x_norm.
            land (torch.Tensor): Land mask of shape (B, 1, H, W), True for land pixels.
            padded_days_mask (torch.Tensor): Mask indicating padded days, shape (B, M, nd, 24, H, W).

        Returns:
            torch.Tensor: Patch-level statistics of shape (B, M, Hp*Wp, embed_dim).
        """
        # x_norm: (B, 1, M, T, H, W) anomaly input (gaps = 0); mask: same shape, True = missing
        B, C, M, T, H, W = x_norm.shape

        assert C == 1 and T % 24 == 0, "Input tensor must have 1 channel and T must be divisible by 24"

        nd = T // 24
        x, mask = x_norm[:, 0], mask[:, 0]

        # Compute observed values
        obs = (~mask).to(x.dtype)
        xv = x.reshape(B, M, nd, 24, H, W)  # gaps are already 0
        ov = obs.reshape(B, M, nd, 24, H, W)
        cnt = ov.sum(2)
        tot = cnt.sum(2)

        # Compute bin-level anomalies and shares
        bin_anom = xv.sum(2) / cnt.clamp(min=1) / self.anom_scale   # empty bins -> 0
        share = cnt / tot[:, :, None].clamp(min=1) * 24

        n_valid = (~padded_days_mask).sum(-1).clamp(min=1)
        frac = tot / n_valid[:, :, None, None]

        features = torch.cat([bin_anom, share, frac[:, :, None]], dim=2)
        F_ = features.shape[2]
        features = features.reshape(B * M, F_, H, W).permute(0, 2, 3, 1)

        # Apply MLP to features
        embeded_features = self.mlp(features).permute(0, 3, 1, 2)

        # Pool over ocean regions only
        ocean = (~land).to(x.dtype)[:, None].repeat_interleave(M, dim=0)
        num = torch.nn.functional.avg_pool2d(embeded_features * ocean, (self.ph, self.pw))
        den = torch.nn.functional.avg_pool2d(ocean, (self.ph, self.pw)).clamp(min=1e-6)
        pooled = num / den

        Hp, Wp = pooled.shape[-2:]
        return pooled.permute(0, 2, 3, 1).reshape(B, M, Hp * Wp, -1)


class PixelFeatures(nn.Module):
    """Returns (B*M, F, H, W) with channels:
       [24 hour-bin mean anomaly | 24 hour-bin share | observed fraction |
        S delta estimates | S confidences of those estimates]
    """
    def __init__(self, anom_scale=0.1, n_sigmas=2):
        super().__init__()
        self.anom_scale = anom_scale
        self.n_sigmas = n_sigmas
        self.delta_idx = 49
        self.n_feat = self.delta_idx + 2 * n_sigmas

    @torch.no_grad()
    def forward(self, x_norm, mask, land, padded_days_mask, ctx):
        """Compute pixel-level features for spatial context.

        Args:
            x_norm: (B,1,M,T,H,W) normalized anomalies with gaps filled as 0
            mask: (B,1,M,T,H,W) boolean mask, True = missing
            land: (B,H,W) boolean mask, True = land
            padded_days_mask: (B,M,T) boolean mask, True = padded
            ctx: (B*M, 2S, T, H, W) neighbor-based context from NeighborFill

        Returns:
            Tensor of shape (B*M, F, H, W) with pixel-level features.
        """
        B, C, M, T, H, W = x_norm.shape
        assert C == 1 and T % 24 == 0
        nd = T // 24

        ocean = (~land)[:, None, None]
        obs = (~mask[:, 0]) & ocean
        valid_t = ~padded_days_mask
        n_valid = valid_t.sum(-1).clamp(min=1)
        miss = (~obs) & valid_t[..., None, None] & ocean

        # diurnal statistics per pixel (same definition as ExplicitStats, un-pooled)
        ov = obs.reshape(B, M, nd, 24, H, W).to(x_norm.dtype)
        xv = x_norm[:, 0].reshape(B, M, nd, 24, H, W) * ov
        cnt = ov.sum(2)
        tot = cnt.sum(2)
        bin_anom = xv.sum(2) / cnt.clamp(min=1) / self.anom_scale
        share = cnt / tot[:, :, None].clamp(min=1) * 24
        frac = tot / n_valid[:, :, None, None]

        # neighbor-based estimate of the missing-hours contribution
        ctx = ctx.view(B, M, 2 * self.n_sigmas, T, H, W)
        n_miss = miss.sum(2).clamp(min=1)
        deltas, confs = [], []
        for s in range(self.n_sigmas):
            fill = ctx[:, :, 2 * s] * self.anom_scale
            conf = ctx[:, :, 2 * s + 1]
            deltas.append(((fill * miss).sum(2) / n_valid[:, :, None, None] / self.anom_scale)[:, :, None])
            confs.append(((conf * miss).sum(2) / n_miss)[:, :, None])

        feats = torch.cat([bin_anom, share, frac[:, :, None], *deltas, *confs], dim=2)
        feats = feats * ocean.to(feats.dtype)
        return feats.reshape(B * M, self.n_feat, H, W)
