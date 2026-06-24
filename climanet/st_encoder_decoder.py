"""
Spatio-Temporal encoder-decoder for Monthly Prediction.
The main model class is SpatioTemporalModel.
"""

import math
import torch
import torch.nn as nn


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

    def __init__(self, in_chans=1, embed_dim=128, patch_size=(1, 4, 4)):
        """
        Args:
            in_chans: Number of input channels (1 for SST)
            embed_dim: Dimension of the patch embedding. The default is 128.
                Many vision transformers use embedding dimensions that are multiples
                of 64 (e.g., 64, 128, 256). This can be tuned.
            patch_size: Tuple of (T, H, W) patch size. Default is (1, 4, 4).
        """
        super().__init__()
        self.patch_size = patch_size

        # proj is a Conv3d with kernel and stride = patch_size to create non-overlapping patches
        # 2 * in_chans because we add a validity channel
        self.proj = nn.Conv3d(
            2 * in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # norm is LayerNorm over the embedding dimension to normalize patch embeddings
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
        # x: (B,1,T,H,W), mask: (B,1,T,H,W) where True means missing
        valid = mask.logical_not()
        x = x * valid  # zero-out missing values
        x = torch.cat([x, valid], dim=1)  # add validity as a channel

        x = self.proj(x)  # (B, C, T', H', W')

        B, C, Tp, Hp, Wp = x.shape
        x = x.reshape(B, C, Tp * Hp * Wp)
        x = x.transpose(1, 2)  # (B, N, C)

        x = self.norm(x)
        return x  # (B, N_patches, embed_dim)


class CyclicTimeEmbedding(nn.Module):
    """Cyclical Temporal encoding using day-of-year and hour-of-day values in
    combination sine and cosine functions

    This module generates fixed (non-learnable) trigonometric temporal encodings
    for the temporal dimension using the cyclcial phase encoded day-of-year and
    hour-of-day values extracted from the datetime associated with the input.
    This represents a natural positional encoding on the temporal cycle related
    to the solar (tropical) year and the diurnal cycle.

    The module uses fixed Fourier frequencies and mixed doy-hod terms to expand
    the cyclic encoding to the embedding dimension and capture time of day and
    day of year interactions. The returned encodings are intended to be added to
    embeddings of the input data by the caller. The module does not perform the
    additon.
    """

    def __init__(self, embed_dim=128, include_cross=True):
        """
        Initialize temporal encodings

        Args:
            embed_dim: Dimension of the embedding.The default is 128.
                Many vision transformers use embedding dimensions that are multiples
                of 64 (e.g., 64, 128, 256). This can be tuned.
            include_cross: bool, default True. Also Create phase_doy +/- phase_hod
                cross term emeddings
        """

        super().__init__()

        self.include_cross = include_cross

        num_base_phase = 2
        num_cross = 2 if include_cross else 0
        num_phase_terms = num_base_phase + num_cross

        # Determine number of frequencies for Fourier expansion in line with embedding dimension

        if embed_dim % (2 * num_phase_terms) == 0:
            num_frequencies = int(embed_dim / (2 * num_phase_terms))
            self.num_freqencies = num_frequencies
            freqs = torch.linspace(1.0, num_frequencies, num_frequencies)
            self.register_buffer("freqs", freqs)
        else:
            raise ValueError(
                f"embed_dim must be an even multiple of num_phase_terms for fixed encoding."
                f"Got embed_dim: {embed_dim} and num_phase_terms: {num_phase_terms}."
            )

    def forward(self, time_features):
        """
        create encodings in of size embedding dimension

        Args:
        time_features: (B, M, T, D) ; D is base_dim

        Returns:
        emb_encode : (B,M,T, embed_dim)
        """
        B, M, T, D = time_features.shape

        # extract individual phases from features
        phase_doy = time_features[..., 0]
        phase_hod = time_features[..., 1]
        phases = [phase_doy, phase_hod]

        # construct cross terms
        if self.include_cross:
            phases.append(phase_doy + phase_hod)
            phases.append(phase_doy - phase_hod)

        # stack these to get (B,M,T,num_terms)
        x = torch.stack(phases, dim=-1)

        # (B, M, T, num_terms, 1)
        x = x.unsqueeze(-1)

        # (1,1,1,1,F)
        freqs = self.freqs.view(1, 1, 1, 1, -1)

        # apply frequencies
        x = x * freqs  # (B, M, T, num_phase_terms, F)

        sinx = torch.sin(x)
        cosx = torch.cos(x)

        emb_encode = torch.cat([sinx, cosx], dim=-1)  # (B,M,T,num_phase_terms, 2F)

        emb_encode = emb_encode.view(B, M, T, -1)  # flatten

        return emb_encode


class TemporalPositionalEncoding(nn.Module):
    """Temporal Positional Encoding using sine and cosine functions.

    This module generates fixed (non-learnable) sinusoidal positional encodings
    for the temporal dimension, following the formulation in
    "Attention Is All You Need" (Vaswani et al., 2017).

    The returned positional encodings are intended to be added to temporal
    embeddings by the caller, but this module itself does not perform the addition.
    """

    def __init__(self, embed_dim=128, max_len=31):
        """Initialize the temporal positional encoding.
        Args:
            embed_dim: Dimension of the embedding.The default is 128.
                Many vision transformers use embedding dimensions that are multiples
                of 64 (e.g., 64, 128, 256). This can be tuned.
            max_len: Maximum length of the temporal dimension to precompute
            encodings for. Default is 31, which is sufficient for a month of
            daily data.
        """
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, embeddim)

    def forward(self, T):
        """Return positional encodings for a temporal sequence.
        Args:
            T: Temporal length (must be <= max_len)
        Returns:
            Tensor of shape (T, embed_dim) containing sinusoidal positional encodings
        """
        return self.pe[:T]  # (T, embed_dim)


class TemporalAttentionAggregator(nn.Module):
    """Temporal attention-based aggregator.

    This module aggregates temporal information for each spatial patch by
    applying attention across the temporal dimension. It consists of two main
    steps:
    1. Day attention: For each month, it computes attention weights across the
    temporal tokens (days) and performs a weighted sum to get one token per
    spatial location for each month.
    2. Cross-month mixing: After temporal aggregation, it applies a Transformer
    encoder layer to mix information across months at each spatial location.

    For each spatial location, the day attention allows the model to learn which
    days are most important for predicting the monthly average, while the
    cross-month mixing allows the model to learn interactions between different
    months.
    """

    def __init__(self, embed_dim=128, max_months=12, dropout=0.0):
        """Initialize the temporal attention aggregator.

        Args:
            embed_dim: Dimension of the embedding. The default is 128.
                Many vision transformers use embedding dimensions that are multiples
                of 64 (e.g., 64, 128, 256). This can be tuned.
            max_months: Maximum number of months (temporal patches) to precompute
            encodings for. Default is 12, which is sufficient for a year of monthly data.
            dropout: Dropout rate for regularization in the day scorer and
            cross-month mixing. Default is 0.0. Increase it if there is overfitting.
        """
        super().__init__()

        self.time_embed = CyclicTimeEmbedding(embed_dim=embed_dim)

        # Positional encodings for days and months
        self.pos_months = TemporalPositionalEncoding(embed_dim, max_len=max_months)

        # Day scorer (within each month)
        self.day_scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),  # normalizing features
            nn.Linear(embed_dim, embed_dim),  # learns temporal feature transformation
            nn.GELU(),  # adds non-linearity to capture complex temporal patterns
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),  # project to a single score
        )

        # Cross month mixing
        self.month_ln = nn.LayerNorm(embed_dim)
        self.month_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.month_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(
                embed_dim, 4 * embed_dim
            ),  # 4 is a common factor in transformer feedforward layers
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        # Pre-compute and register as buffer — auto-moves with .to(device/dtype)
        pe = self.pos_months(max_months)  # (max_months, C)
        self.register_buffer("pe_months_cache", pe)  # tracks device/dtype automatically

    def forward(self, x, M, T, H, W, time_features, padded_days_mask=None):
        """
        Args:
            x: (B, M, T, H, W, C) containing spatio-temporal tokens, where C is the embedding dimension.
            M: number of months
            T: number of temporal tokens per month after temporal patching (Tp)
            H: spatial height after spatial patching
            W: spatial width after spatial patching
            time_features: (B,M,T,2) containing cyclically phase encoded DOY and HOD
            padded_days_mask: Optional boolean tensor of shape (B, M, T), bool,
                True indicating which day tokens are padded (because some months
                have fewer days). This is used to mask out padded tokens in attention computation.
        Returns:
            Tensor of shape (B, M, H*W, C) with one temporally aggregated, where C is the embedding dimension.
        """
        B, M, Tp, Hp, Wp, C = x.shape
        HW = Hp * Wp

        # Reshape to (B, Hp*Wp, M, Tp, C) for temporal processing
        x = x.reshape(B, Hp, Wp, M, Tp, C)
        x = x.permute(0, 3, 4, 1, 2, 5).contiguous()  # (B, M, Tp, Hp, Wp, C)
        seq = x.reshape(B, M, Tp, HW, C).permute(0, 3, 1, 2, 4)

        temp_emb = self.time_embed(time_features)
        pe_months = self.pe_months_cache[:M]
        token_emb = temp_emb + pe_months[None, :, None, :]  # (B, M, T, C)

        day_logits = self.day_scorer(token_emb).squeeze(-1)  # (B, M, T)

        if padded_days_mask is not None:
            day_logits = day_logits.masked_fill(padded_days_mask, float("-inf"))

        day_w = torch.softmax(day_logits, dim=-1)            # (B, M, T)
        day_w = day_w.unsqueeze(1).unsqueeze(-1)

        month_tokens = (seq * day_w).sum(dim=3)
        month_emb = (token_emb * day_w.squeeze(1)).sum(dim=2)

        month_tokens = month_tokens + month_emb[:, None, :, :]

        z = month_tokens.reshape(B * HW, M, C)
        z= self.month_ln(z)
        attn_out, _ = self.month_attn(z, z, z, need_weights=False)

        z = z + attn_out
        z = z + self.month_ffn(z)

        z = z.reshape(B, HW, M, C)
        out = z.permute(0, 2, 1, 3).contiguous()

        return out # (B, M, H*W, C)  C: embedding dimension


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
        overlap=1,
        num_months=12,
        dropout=0.0,
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
            overlap: Overlap size for deconvolution. It creates smooth blending
                between adjacent upsampled patches. Default is 1, no overlap at edges.
            num_months: Number of months. Default is 12.
            dropout: Dropout rate for regularization in the refinement block. Default is 0.0.
        """
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.overlap = overlap

        # Mix channel features on the patch grid (Hp, Wp)
        # Input shape: (B, embed_dim, Hp, Wp) → Output shape: (B, hidden, Hp, Wp)
        # here kernel_size=1 means we are mixing features at each patch location
        # without spatial interaction
        in_channels, out_channels = embed_dim, hidden
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Upsample to full resolution
        # With kernel = stride + 2*overlap and padding=overlap,
        # output size is exact: H = Hp*patch_h, W = Wp*patch_w (no output_padding needed).
        k_h = patch_h + 2 * overlap
        k_w = patch_w + 2 * overlap
        # As spatial size increases, channel count decreases to keep computation
        # manageable; here  hidden // 2 is a design choice.
        in_channels, out_channels = hidden, hidden // 2
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(k_h, k_w),
            stride=(patch_h, patch_w),
            padding=overlap,
            output_padding=0,
            bias=True,
        )

        # Final conv head to get single channel output kernel_size=3 is the most
        # common choice for spatial convolutions; it's the smallest kernel that
        # captures spatial context in all directions
        in_channels, out_channels = hidden // 2, hidden // 2

        # Refinement block: a small conv layers to smooth patch boundaries
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
        )

        # Final conv head to map to single-channel output
        self.head = nn.Conv2d(out_channels, 1, kernel_size=1)

        # Learnable scale and bias (mean and std) to improve predictions
        self.scale = nn.Parameter(torch.ones(num_months))
        self.bias = nn.Parameter(torch.zeros(num_months))

    def forward(self, latent, M, out_H, out_W, land_mask=None):
        """Reconstruct 2D maps from latent patch tokens.
        Args:
            latent: Tensor of shape (B, M*Hp*Wp, C) where C is the embedding dimension.
            M: Number of months (temporal patches)
            out_H: Target output height (must be divisible by patch_h)
            out_W: Target output width (must be divisible by patch_w)
            land_mask: Optional boolean tensor of shape (B, out_H, out_W). Values set to True
                will be masked out (set to 0) in the output (only ocean pixels exist).
        Returns:
            Tensor of shape (B, M, out_H, out_W) representing the monthly variable e.g. SST.
        """
        B, M, Np, C = latent.shape
        Hp = out_H // self.patch_h
        Wp = out_W // self.patch_w
        assert Np == Hp * Wp, f"Token mismatch: got {Np}, expected {Hp * Wp}"

        # transforms the latent tensor from sequence format to image format for
        # convolution operations;
        out = latent.reshape(B, M, Hp, Wp, C)
        out = out.permute(0, 1, 4, 2, 3).contiguous()
        out = out.reshape(B * M, C, Hp, Wp)

        # Apply 1x1 convolution to mix features
        out = self.proj(out)  # (B*M, hidden, Hp, Wp)

        # Use transposed convolution to upsample
        out = self.deconv(out)  # (B*M, hidden//2, H, W)

        # Refinement CNN to smooth boundaries
        out = self.refine(out)  # (B*M, hidden//2, H, W)

        # Apply final conv head to get single channel output
        out = self.head(out)  # (B*M, 1, H, W)

        # Apply scale and bias per month to improve predictions; reshape to (B*M, 1, 1, 1) for broadcasting
        scale = self.scale[:M].unsqueeze(0).expand(B, M).reshape(B * M, 1, 1, 1)
        bias = self.bias[:M].unsqueeze(0).expand(B, M).reshape(B * M, 1, 1, 1)
        out = out * scale + bias
        out = out.view(B, M, out_H, out_W)  # (B, M, H, W)

        # Mask out land areas if land_mask is provided
        if land_mask is not None:
            out = out.masked_fill(land_mask.bool()[:, None, :, :], 0.0)
        return out  # (B, M, out_H, out_W)


class GeoPositionScaleEmbedding(nn.Module):
    """Sphere aware encoding of geographical position and patch (resolution) scales.

    This module uses static precomputed spherical-harmonic-based geoposition encodings and
    scale encodings at the patch level to generate learned positonal embedding for patches.
    The static, precomputed geo position and scale features are created at the dataset
    level and passed to the model, together with patch data.

    Geo position uses a sphere-aware patch area average of the PCA projection of real-valued
    spherical harmonics functions up to and including order L ( with dim PCA < (L+1)**2 ) at
    the resolution of the input data.

    Patch scale embedding encodes, patch, scale, anisotropy, linear resolution, and
    effective harmonic cut-off.

    These embeddings are concatenated with learnable vector valued gains and then projected
    to the required embedding dimension using a simple dense NN.
    """

    def __init__(
        self,
        sh_dim=96,
        scale_dim=10,
        embed_dim=128,
    ):
        """
        initialize geo-position and scale embeddings and projection

        Args:
            sh_dim: int, Dimension of pca of spherical harmonics for embedding. defaults to 96
            scale_dim: int, Dimension of patch scale feature embedding. default 10
            embed_dim: int, Dimension of embeddings to be created. default 128
        """

        super().__init__()

        # 0.1 is an initialization scale factor
        self.sh_gain = nn.Parameter(0.1 * torch.ones(sh_dim))

        # 0.1 is an initialization scale factor
        self.scale_gain = nn.Parameter(0.1 * torch.ones(scale_dim))
        self.scale_norm = nn.LayerNorm(scale_dim)

        in_dim = sh_dim + scale_dim
        hidden_dim = 2 * embed_dim

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        sh_geo_pos,
        geo_scale_feat,
    ):
        """
        Create learned geo-position-and-scale embedding of desired
        dimension from pre-calculated patch level geo-position and
        patch scale embeddings

        Args:
            sh_geo_pos: Tensor of dimension sh_dim. Patch level geo-position
                embedding using pca of spherical harmonics
            geo_scale_feat: Tensor of dimension scale_dim. Patch level
                patch-scale features
        Returns:
            geo_emb: Tensor of dimesnion embed_dim. Learned geo-position-and-scale
                embedding
        """
        sh_geo_pos = self.sh_gain * sh_geo_pos
        geo_scale_feat = self.scale_norm(geo_scale_feat)
        geo_scale_feat = self.scale_gain * geo_scale_feat

        x = torch.cat([sh_geo_pos, geo_scale_feat], dim=-1)

        geo_emb = self.proj(x)

        return geo_emb


class SpatialTransformer(nn.Module):
    """Spatial Transformer for spatial feature mixing.

    This module applies a standard Transformer encoder to a sequence of spatial tokens
    (patch embeddings), allowing information to be mixed across all spatial locations.

    Key points:
        - Uses multi-head self-attention and feedforward layers.
        - Designed to operate on flattened spatial tokens.
    """

    def __init__(self, embed_dim=128, depth=2, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        """Initialize the spatial transformer.
        Args:
            embed_dim: Dimension of the embedding. Default is 128.
                The embedding dimensions are multiples of 64 (e.g., 64, 128,
                256). This can be tuned.
            depth: Number of transformer encoder layers. Default is 2. This can be
                increased for more complex spatial mixing.
            num_heads: Number of attention heads in each layer. Default is 4.
                When embed_dim is 128, 4 heads is a common choice.
            mlp_ratio: Ratio of feedforward hidden dimension to embed_dim. Default is 4.0.
            dropout: Dropout rate applied to attention and feedforward layers. Default is 0.0.
        """
        super().__init__()

        # a single Transformer encoder block that
        # performs self-attention and feedforward processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        # stack multiple layers to form the full encoder
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        """Forward pass of the spatial transformer.
        Args:
            x: Input tensor of shape (B, N, C), where N = number of spatial tokens (H'*W') and
                C = embedding dimension
        Returns:
            Tensor of shape (B, N, C) with spatially mixed features across patches
        """
        return self.enc(x)


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
        3. Add 2D spatial positional encodings and mix spatial features with SpatialTransformer.
        4. Decode aggregated tokens into a full-resolution 2D map using MonthlyConvDecoder.

    Output:
        - Reconstructed monthly (SST) map of shape (B, M, H, W)
    """

    def __init__(
        self,
        in_chans=1,
        embed_dim=128,
        patch_size=(1, 4, 4),
        max_months=12,
        num_months=12,
        hidden=256,
        overlap=1,
        spatial_depth=2,
        spatial_heads=4,
        dropout=0.0,
        sh_dim=96,
        scale_dim=10,
    ):
        """Initialize the Spatio-Temporal Model.

        Args:
            in_chans: Number of input channels (e.g., 1 for SST, additional channels possible)
            embed_dim: Dimension of the patch embedding
            patch_size: Tuple of (T, H, W) patch sizes for temporal and spatial patching
            max_months: Maximum number of months for temporal positional encoding
            num_months: Number of months to predict (output channels in decoder)
            hidden: Hidden dimension used in the decoder
            overlap: Overlap for deconvolution in the decoder
            max_H: Maximum spatial height for 2D positional encoding
            max_W: Maximum spatial width for 2D positional encoding
            spatial_depth: Number of layers in the spatial Transformer
            spatial_heads: Number of attention heads in the spatial Transformer
            dropout: Dropout rate for regularization in various components. Increase it if there is overfitting.
            sh_dim: Dimension of spherical harmonics based pca of geo-position
            scale_dim: Dimension of patch-level patch-scale features
        """
        super().__init__()

        # Store arguments to be used later for model saving/loading
        self.config = {
            k: v for k, v in locals().items() if k not in ("self", "__class__")
        }

        self.encoder = VideoEncoder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.temporal = TemporalAttentionAggregator(
            embed_dim=embed_dim,
            max_months=max_months,
            dropout=dropout,
        )
        self.geo_embedding = GeoPositionScaleEmbedding(
            sh_dim=sh_dim,
            scale_dim=scale_dim,
            embed_dim=embed_dim,
        )
        self.spatial_tr = SpatialTransformer(
            embed_dim=embed_dim,
            depth=spatial_depth,
            num_heads=spatial_heads,
            dropout=dropout,
        )
        self.decoder = MonthlyConvDecoder(
            embed_dim=embed_dim,
            patch_h=patch_size[1],
            patch_w=patch_size[2],
            hidden=hidden,
            overlap=overlap,
            num_months=num_months,
            dropout=dropout,
        )
        self.patch_size = patch_size

    def forward(
        self,
        daily_data,
        daily_mask,
        daily_timef,
        land_mask_patch,
        geo_pos_embedding_patch,
        scale_feature_patch,
        padded_days_mask=None,
    ):
        """Forward pass of the Spatio-Temporal model.

        Args:
            daily_data: Tensor of shape (B, C, M, T, H, W) containing daily
                data, where C is the number of channels (e.g., 1 for SST)
            daily_mask: Boolean tensor of same shape as daily_data indicating missing values
            daily_timef: Tensor of shape (B, M, T, 2) containing the cyclically phase encoded day-of-year
                and hour-of-day information for the daily data
            land_mask_patch: Boolean tensor of shape (B, H, W) to mask land areas in the output
            padded_days_mask: Optional boolean tensor of shape (B, M, T) indicating which day tokens are padded
                 (True for padded tokens). Used to mask out padded tokens in temporal attention.
        Returns:
            monthly_pred: Tensor of shape (B, M, H, W) representing the reconstructed monthly map
        """
        B, C, M, T, H, W = daily_data.shape

        Tp = T // self.patch_size[0]
        Hp = H // self.patch_size[1]
        Wp = W // self.patch_size[2]

        # check shape and patch compatibility
        assert daily_mask.shape == daily_data.shape, (
            "daily_mask must have the same shape as daily_data"
        )
        assert H % self.patch_size[1] == 0 and W % self.patch_size[2] == 0, (
            "H and W must be divisible by patch size"
        )
        assert T % self.patch_size[0] == 0, "T must be divisible by patch size"

        # Step 1: Encode spatio-temporal patches
        # each month independently by folding M into batch
        # encoder input shape = (B, C, T, H, W) where C is channel.
        # encoder output shape = (B, N_patches, embed_dim)
        # so M is folded into B, and T, H, W are the spatio-temporal dimensions to be patched.
        daily_data_reshaped = daily_data.reshape(B * M, C, T, H, W)
        daily_mask_reshaped = daily_mask.reshape(B * M, C, T, H, W)

        latent = self.encoder(
            daily_data_reshaped, daily_mask_reshaped
        )  # (B*M, N_patches, embed_dim)

        # Step 2: Aggregate temporal information for each spatial patch
        # temporal input shape = (B, M*T*H*W, C),  C: embedding dimension
        # temporal output shape = (B, M, H*W, C)  C: embedding dimension
        embed_dim = latent.shape[-1]
        latent = latent.view(B, M, Tp, Hp, Wp, embed_dim)

        agg_latent = self.temporal(
            latent, M, Tp, Hp, Wp, daily_timef, padded_days_mask=padded_days_mask
        )  # (B, M, Hp*Wp, embed_dim)

        # Step 3: Add geo position and scale encodings
        geo_emb = self.geo_embedding(
            geo_pos_embedding_patch,
            scale_feature_patch,
        )  # (B, embed_dim)

        # Broadcasting: same geo embedding for all M months at each Hp*Wp location
        # we use weighted mean patch embedding, see `geo_embedding_utils.py`
        geo_emb = geo_emb[:, None, None, :]  # (B,1,1,E)
        x = agg_latent + geo_emb  # (B, M, Hp*Wp, E)

        # Step 4: Spatial mixing with Transformer
        # spatial transformer input shape = (B, N, C), output shape = (B, N, C) C: embedding dimension
        # M is folded in B.

        C = x.shape[-1]
        x = x.reshape(B * M, Hp * Wp, C)
        x = self.spatial_tr(x)
        x = x.view(B, M, Hp * Wp, C)

        # Step 5: Decode to full-resolution 2D map
        # decoder input shape is (B, M*Hp*Wp, C), C: embedding dimension
        # decoder output shape is (B, M, H, W)
        monthly_pred = self.decoder(x, M, H, W, land_mask_patch)  # (B, M, H, W)
        return monthly_pred
