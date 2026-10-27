"""
Spatio-Temporal encoder-decoder for Monthly Prediction.
The main model class is SpatioTemporalModel.
"""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class VideoEncoder(nn.Module):
    """Video Encoder with spatio-temporal patch embedding.

    This module converts an input video into a sequence of non-overlapping
    spatio-temporal patch embeddings using a 3D convolution.

    Masking is handled by:
    - zeroing out masked (missing) pixels
    - concatenating a validity mask as an additional input channel

    The convolution uses kernel size and stride equal to the patch size.
    The output is a sequence of patch embeddings, as used in VideoMAE:
    https://arxiv.org/abs/2203.12602
    """

    def __init__(self, in_chans=1, embed_dim=128, proj_dim=64, patch_size=(1, 4, 4), dropout=0.0):
        """
        Args:
            in_chans: Number of input channels (1 for SST)
            embed_dim: Dimension of the patch embedding. The default is 128.
                Many vision transformers use embedding dimensions that are multiples
                of 64 (e.g., 64, 128, 256). This can be tuned.
            proj_dim: Dimension of the intermediate projection before expanding to embed_dim.
                This allows control over the size of the patch embeddings before the final embedding.
            patch_size: Tuple of (T, H, W) patch size. Default is (1, 4, 4).
            dropout: Dropout rate applied after the patch embedding and expansion.
        """
        super().__init__()
        self.patch_size = patch_size

        # proj is a Conv3d with kernel and stride = patch_size to create non-overlapping patches
        # 2 * in_chans because we add a validity channel
        self.proj = nn.Conv3d(
            2 * in_chans, proj_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.expand = nn.Linear(proj_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        """Forward pass with masking support via an additional validity channel.
        Args:
            x: Input video tensor of shape (B, C, T, H, W)
            mask: Boolean mask tensor of shape (B, C, T, H, W), where True
            indicates masked pixels

        Returns:
            Embedded patches of shape (B, N_patches, embed_dim)
        Notes:
            - Masked pixels are zeroed out before patch embedding
            - A validity mask (1 = observed, 0 = missing) is concatenated
            as an additional input channel
        """
        valid = (~mask).to(dtype=x.dtype)

        if valid.shape != x.shape:
            valid = valid.expand_as(x)

        x = x.masked_fill(mask, 0.0)
        x = torch.cat((x, valid), dim=1)

        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.expand(x)
        x = self.norm(x)
        x = self.drop(x)
        return x


class RelTimeEmbedding(nn.Module):
    """Relative time embedding for temporal tokens within a month."""
    def __init__(self, embed_dim=64, n_harm=4):
        """
        Initialize the relative time embedding.

        Args:
            embed_dim: Dimension of the output embedding.
            n_harm: Number of harmonic functions for the Fourier features.
        """
        super().__init__()

        # harmonic indices for Fourier features
        self.register_buffer("h_indices", torch.arange(1, n_harm + 1).float())

        self.proj = nn.Linear(2 * n_harm + 1, embed_dim, bias=False)

    def forward(self, T, padded_days_mask):
        """
        Compute the relative time embedding for temporal tokens within a month.

        Args:
            T: Number of temporal tokens (days or hours) in the month.
            padded_days_mask: Boolean mask tensor of shape (B, M, T), where True indicates padded times.

        Returns:
            Relative time embeddings of shape (B, M, T, embed_dim)
        """
        n_valid = (~padded_days_mask).sum(-1).clamp(min=1)   # valid times per month

        # Compute the relative position of each valid time within the month.
        time_pos = torch.arange(T, device=padded_days_mask.device)[None, None]

        time_pos_norm = (time_pos / n_valid[..., None]).clamp(max=1.0)
        h_phases = math.pi * time_pos_norm[..., None] * self.h_indices  # harmonic phases
        rel_time_embed = self.proj(
            torch.cat([torch.sin(h_phases), torch.cos(h_phases), time_pos_norm[..., None] - 0.5], -1)
        )
        return rel_time_embed


class HourEmbedding(nn.Module):
    """Fourier embedding of hour-of-day phase only."""
    def __init__(self, embed_dim=128, n_harmonics=4):
        """Initialize the hour embedding module.
        Args:
            embed_dim: Dimension of the output embedding.
            n_harmonics: Number of harmonic functions for the Fourier features.
        """
        super().__init__()
        self.register_buffer("freq", torch.arange(1, n_harmonics + 1).float())
        self.proj = nn.Linear(2 * n_harmonics, embed_dim, bias=False)

    def forward(self, time_features):
        """Compute the hour-of-day embedding.

        Args:
            time_features: Tensor of shape (B, M, T, F) containing time features.

        Returns:
            Hour embeddings of shape (B, M, T, embed_dim)
        """
        x = time_features[..., 2].unsqueeze(-1) * self.freq # (B, M, T, F)
        return self.proj(torch.cat([torch.sin(x), torch.cos(x)], dim=-1))


class TemporalAttentionAggregator(nn.Module):
    """Temporal attention-based aggregator.

    This module aggregates temporal information for each spatial patch by
    applying attention across the temporal dimension.
    Steps:
        1. Compute hour and relative time embeddings.
        2. Apply token MLP to combine data and time embeddings.
        3. Compute day attention scores and aggregate temporal tokens.
        4. Return aggregated spatio-temporal tokens.
    """

    def __init__(self, embed_dim=128, dropout=0.0):
        """Initialize the temporal attention aggregator.

        Args:
            embed_dim: Dimension of the embedding. The default is 128.
                Many vision transformers use embedding dimensions that are multiples
                of 64 (e.g., 64, 128, 256). This can be tuned.
            dropout: Dropout rate for regularization in the day scorer.
                Default is 0.0. Increase it if there is overfitting.
        """
        super().__init__()

        self.hour_embed = HourEmbedding(embed_dim)
        self.rel_embed = RelTimeEmbedding(embed_dim)

        # token MLP: first layer split into data part and hour part
        self.mlp_ln_seq = nn.LayerNorm(embed_dim)
        self.mlp_ln_hour = nn.LayerNorm(embed_dim)
        self.mlp_seq = nn.Linear(embed_dim, embed_dim)
        self.mlp_hour = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mlp_out = nn.Linear(embed_dim, embed_dim)

        # day scorer: same split, smaller output
        self.sc_ln_seq = nn.LayerNorm(embed_dim)
        self.sc_ln_hour = nn.LayerNorm(embed_dim)
        self.sc_seq = nn.Linear(embed_dim, embed_dim // 2)
        self.sc_hour = nn.Linear(embed_dim, embed_dim // 2, bias=False)
        self.sc_drop = nn.Dropout(dropout)
        self.sc_out = nn.Linear(embed_dim // 2, 1)

        self.out_ln = nn.LayerNorm(embed_dim)

    def forward(self, x, time_features, padded_days_mask=None):
        """
        Args:
            x: (B, M, T, H, W, E) containing spatio-temporal tokens, where E is the embedding dimension.
                B: batch size
                M: number of months
                T: number of temporal tokens per month after temporal patching (Tp)
                H: spatial height after spatial patching
                W: spatial width after spatial patching
            time_features: (B,M,T,3) containing cyclically phase encoded MOY, DOY and HOD
            padded_days_mask: Optional boolean tensor of shape (B, M, T), bool,
                True indicating which day tokens are padded (because some months
                have fewer days). This is used to mask out padded tokens in attention computation.
        Returns:
            Tensor of shape (B, M, H*W, E) with one temporally aggregated, where E is the embedding dimension.
        """
        B, M, Tp, Hp, Wp, E = x.shape
        HW = Hp * Wp

        seq = x.reshape(B, M, Tp, HW, E).permute(0, 3, 1, 2, 4)

        hour_emb = (self.hour_embed(time_features) + self.rel_embed(Tp, padded_days_mask)).unsqueeze(1)

        # token MLP (hour term computed once, broadcast over HW)
        h = self.mlp_seq(self.mlp_ln_seq(seq)) + self.mlp_hour(self.mlp_ln_hour(hour_emb))
        seq = seq + self.mlp_out(torch.nn.functional.gelu(h))

        # day scorer
        s = self.sc_seq(self.sc_ln_seq(seq)) + self.sc_hour(self.sc_ln_hour(hour_emb))
        logits = self.sc_out(self.sc_drop(torch.nn.functional.gelu(s))).squeeze(-1)  # (B, HW, M, Tp)

        if padded_days_mask is not None:
            logits = logits.masked_fill(padded_days_mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(logits, dim=-1).unsqueeze(-1)

        month_tokens = (seq * w).sum(dim=3)  # (B, HW, M, E)
        out = self.out_ln(month_tokens).permute(0, 2, 1, 3)  # (B, M, HW, E)

        return out


class MonthlyConvDecoder(nn.Module):
    """Decoder to reconstruct 2D maps from patch tokens.

    The MonthlyConvDecoder converts latent patch tokens back to pixel space:
        - Applies a 1*1 convolution to mix features on the patch grid.
        - Uses a transposed convolution (deconvolution) to upsample tokens to the original spatial resolution.
        - Applies a convolutional refinement block to smooth patch boundaries.
        - Applies a small convolutional head to produce the final single-channel output.
        - Optionally masks out land regions using a boolean mask.
    """

    def __init__(
        self,
        embed_dim=128,
        patch_h=4,
        patch_w=4,
        hidden=128,
    ):
        """
        Args:
            embed_dim: Dimension of the patch embedding.The default is 128.
                Many vision transformers use embedding dimensions that are
                multiples of 64 (e.g., 64, 128, 256). This can be tuned.
            patch_h: Patch height
            patch_w: Patch width
            hidden: Hidden dimension in the decoder for mixing channel features.
                The default is 128, which can be tuned.
        """
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w

        # Mix channel features on the patch grid (Hp, Wp)
        # Input shape: (B, embed_dim, Hp, Wp) → Output shape: (B, hidden, Hp, Wp)
        # here kernel_size=1 means we are mixing features at each patch location
        # without spatial interaction
        in_channels, out_channels = embed_dim, hidden
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Spatial processing before upsampling
        self.spatial_mix = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)

        # Upsample to full resolution
        in_channels, out_channels = hidden, hidden // 2
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
            padding=0,
        )

        # Refinement block: a small conv layers to smooth patch boundaries
        in_channels, out_channels = hidden // 2, hidden // 2
        self.refine = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Final conv head to map to single-channel output
        self.head = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, latent, out_H, out_W, land_mask=None):
        """Reconstruct 2D maps from latent patch tokens.
        Args:
            latent: Tensor of shape (B, M*Hp*Wp, E) where E is the embedding dimension.
                M: Number of months (temporal patches)
            out_H: Target output height (must be divisible by patch_h)
            out_W: Target output width (must be divisible by patch_w)
            land_mask: Optional boolean tensor of shape (B, out_H, out_W). Values set to True
                will be masked out (set to 0) in the output (only ocean pixels exist).
        Returns:
            Tensor of shape (B, M, out_H, out_W) representing the monthly variable e.g. SST.
        """
        B, M, Np, E = latent.shape
        Hp = out_H // self.patch_h
        Wp = out_W // self.patch_w
        assert Np == Hp * Wp, f"Token mismatch: got {Np}, expected {Hp * Wp}"

        # transforms the latent tensor from sequence format to image format for
        # convolution operations;
        out = latent.reshape(B, M, Hp, Wp, E)
        out = out.permute(0, 1, 4, 2, 3).contiguous()
        out = out.reshape(B * M, E, Hp, Wp)

        # Apply 1x1 convolution to mix features
        out = self.proj(out)  # (B*M, hidden, Hp, Wp)

        # Spatial interaction before upsampling
        out = self.spatial_mix(out)

        # Use transposed convolution to upsample
        out = self.deconv(out)  # (B*M, hidden//2, H, W)

        # Refinement CNN to smooth boundaries
        refine_input = out
        out = self.refine(out)  # (B*M, hidden//2, H, W)
        out = out + refine_input  # Residual connection to preserve original features

        # Apply final conv head to get single channel output
        out = self.head(out)  # (B*M, 1, H, W)

        out = out.view(B, M, out_H, out_W)  # (B, M, H, W)

        # Mask out land areas if land_mask is provided
        if land_mask is not None:
            out = out.masked_fill(land_mask.bool()[:, None, :, :], 0.0)
        return out  # (B, M, out_H, out_W)


class SpatioTemporalModel(nn.Module):
    """Spatio-Temporal Model for Monthly Prediction.

    Processes daily data in a video-style format with shape (B, C, T, H, W):
        B: batch size
        C: number of channels (e.g., 1 for SST, can include additional channels like masks)
        T: temporal dimension (number of days, e.g., 31 for a month)
        H: spatial height
        W: spatial width

    The model pipeline:
        1. Encode spatio-temporal patches using VideoEncoder.
        2. Aggregate temporal information for each spatial patch via TemporalAttentionAggregator.
        3. Add patch statistics (e.g., mean, variance) to the tokens.
        4. Decode aggregated tokens into a full-resolution 2D map using MonthlyConvDecoder.

    Output:
        - Reconstructed monthly (SST or residuals) map of shape (B, M, H, W)
    """

    def __init__(
        self,
        in_chans=1,
        embed_dim=128,
        encoder_proj_dim=64,
        patch_size=(1, 4, 4),
        hidden=256,
        dropout=0.0,
        anom_scale=0.1,
        use_checkpoint=True,
    ):
        """Initialize the Spatio-Temporal Model.

        Args:
            in_chans: Number of input channels (e.g., 1 for SST, additional channels possible)
            embed_dim: Dimension of the patch embedding
            encoder_proj_dim: Dimension of the projection in the encoder
            patch_size: Tuple of (T, H, W) patch sizes for temporal and spatial patching
            hidden: Hidden dimension used in the decoder
            dropout: Dropout rate for regularization in various components. Increase it if there is overfitting.
            anom_scale: Scaling factor for anomaly computation in patch statistics.
            use_checkpoint: Whether to use gradient checkpointing to save memory during training.
        """
        super().__init__()

        # Store arguments to be used later for model saving/loading
        self.config = {
            k: v for k, v in locals().items() if k not in ("self", "__class__")
        }

        self.use_checkpoint = use_checkpoint

        self.encoder = VideoEncoder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            proj_dim=encoder_proj_dim,
            patch_size=patch_size,
            dropout=dropout,
        )
        self.temporal = TemporalAttentionAggregator(
            embed_dim=embed_dim,
            dropout=dropout,
        )

        self.decoder = MonthlyConvDecoder(
            embed_dim=embed_dim,
            patch_h=patch_size[1],
            patch_w=patch_size[2],
            hidden=hidden,
        )
        self.patch_size = patch_size
        self.stats = PatchStats(embed_dim=embed_dim, patch_size=patch_size, anom_scale=anom_scale)

    def forward(
        self,
        input_data,
        daily_mask,
        daily_timef,
        land_mask_patch,
        padded_days_mask=None,
    ):
        """Forward pass of the Spatio-Temporal model.

        Args:
            input_data: Tensor of shape (B, C, M, T, H, W) containing daily
                data, where C is the number of channels (e.g., 1 for SST)
            daily_mask: Boolean tensor of same shape as input_data indicating missing values
            daily_timef: Tensor of shape (B, M, T, 2) containing the cyclically phase encoded day-of-year
                and hour-of-day information for the daily data
            land_mask_patch: Boolean tensor of shape (B, H, W) to mask land areas in the output
            geo_pos_embedding_patch: Tensor of shape (B, Hp*Wp, sh_embed_dim) containing the geo-positional embeddings for each model patch within the dataset patch
            scale_feature_patch: Tensor of shape (B, Hp*Wp, scale_dim) containing the scale features for each model patch within the dataset patch
            padded_days_mask: Optional boolean tensor of shape (B, M, T) indicating which day tokens are padded
                 (True for padded tokens). Used to mask out padded tokens in temporal attention.
        Returns:
            monthly_pred: Tensor of shape (B, M, H, W) representing the reconstructed monthly map
        """
        B, C, M, T, H, W = input_data.shape

        Tp = T // self.patch_size[0]
        Hp = H // self.patch_size[1]
        Wp = W // self.patch_size[2]

        # check shape and patch compatibility
        assert daily_mask.shape == input_data.shape, (
            "daily_mask must have the same shape as input_data"
        )
        assert H % self.patch_size[1] == 0 and W % self.patch_size[2] == 0, (
            "H and W must be divisible by patch size"
        )
        assert T % self.patch_size[0] == 0, "T must be divisible by patch size"

        # input_data, daily_mask: (B, C, M, T, H, W), mask True = missing
        # TODO: move to dataset
        obs = ~daily_mask
        cnt = obs.sum(dim=3, keepdim=True).clamp(min=1)
        mean_obs = (input_data * obs).sum(dim=3, keepdim=True) / cnt
        input_data_norm = (input_data - mean_obs) * obs

        # Step 1: Encode spatio-temporal patches
        # each month independently by folding M into batch
        input_data_reshaped = input_data_norm.reshape(B * M, C, T, H, W)
        daily_mask_reshaped = daily_mask.reshape(B * M, C, T, H, W)

        if self.use_checkpoint:
            latent = checkpoint(
                self.encoder,
                input_data_reshaped,
                daily_mask_reshaped,
                use_reentrant=False,
            )
        else:
            latent = self.encoder(
                input_data_reshaped, daily_mask_reshaped,
            )

        # Step 2: Aggregate temporal information for each spatial patch
        embed_dim = latent.shape[-1]
        latent = latent.view(B, M, Tp, Hp, Wp, embed_dim)

        if self.use_checkpoint:
            x_averaged = checkpoint(
                self.temporal,
                latent,
                daily_timef,
                padded_days_mask,
                use_reentrant=False,
            )
        else:
            x_averaged = self.temporal(
                latent, daily_timef, padded_days_mask
            )

        # # Step 3: Add explicit statistics per patch
        stat_embed = self.stats(input_data_norm, daily_mask, land_mask_patch, padded_days_mask)
        x = x_averaged + stat_embed

        # Step 4: Decode to full-resolution 2D map
        if self.use_checkpoint:
            monthly_res_pred = checkpoint(
                self.decoder, x, H, W, land_mask_patch, use_reentrant=False
            )
        else:
            monthly_res_pred = self.decoder(x, H, W, land_mask_patch)

        return monthly_res_pred   # (B, M, H, W)


class PatchStats(nn.Module):
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
