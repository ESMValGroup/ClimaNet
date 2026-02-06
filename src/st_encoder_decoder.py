"""
Spatio-Temporal encoder-decoder for Monthly Prediction.
The main model class is SpatioTemporalModel.
"""

import math
import torch
import torch.nn as nn
from einops import rearrange


device = "cuda" if torch.cuda.is_available() else "cpu"

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
    def __init__(self, in_chans=1, embed_dim=128, patch_size=(1,4,4), drop=0.0):
        """
        Args:
            in_chans: Number of input channels (1 for SST)
            embed_dim: Dimension of the patch embedding. The default is 128.
                Many vision transformers use embedding dimensions that are multiples
                of 64 (e.g., 64, 128, 256). This can be tuned.
            patch_size: Tuple of (T, H, W) patch size. Default is (1, 4, 4).
            drop: Probability of an element to be zeroed. Default is 0.0.
                Increase it if there is overfitting.
        """
        super().__init__()
        self.patch_size = patch_size

        # proj is a Conv3d with kernel and stride = patch_size to create non-overlapping patches
        # 2 * in_chans because we add a validity channel
        self.proj = nn.Conv3d(2*in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # norm is LayerNorm over the embedding dimension to normalize patch embeddings
        self.norm = nn.LayerNorm(embed_dim)

        # dropout for regularization
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        """ Forward pass with masking support via an additional validity channel.
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
        valid = (~mask).float()
        x = x * valid  # zero-out missing values
        x = torch.cat([x, valid], dim=1)  # add validity as a channel
        x = self.proj(x)  # (B, C, T', H', W')
        x = rearrange(x, "b c t h w -> b (t h w) c")
        x = self.norm(x)
        x = self.drop(x)
        return x  # (B, N_patches, embed_dim)


class TemporalPositionalEncoding(nn.Module):
    """ Temporal Positional Encoding using sine and cosine functions.

    This module generates fixed (non-learnable) sinusoidal positional encodings
    for the temporal dimension, following the formulation in
    "Attention Is All You Need" (Vaswani et al., 2017).

    The returned positional encodings are intended to be added to temporal
    embeddings by the caller, but this module itself does not perform the addition.
    """
    def __init__(self, embed_dim=128, max_len=31):
        """ Initialize the temporal positional encoding.
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
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
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

    This module aggregates temporal tokens into a single token per spatial patch
    by computing learned attention weights over the temporal dimension. Temporal
    positional encodings are added before computing the attention weights. Then,
    temporal attention weights are calculated by applying `nn.Sequential` to get
    a scalar score.

    For each spatial location, a weighted sum over time is performed to
    produce one aggregated token.
    """
    def __init__(self, embed_dim=128, max_T=31):
        """Initialize the temporal attention aggregator.

        Args:
            embed_dim: Dimension of the embedding. The default is 128.
                Many vision transformers use embedding dimensions that are multiples
                of 64 (e.g., 64, 128, 256). This can be tuned.
            max_len: Maximum length of the temporal dimension to precompute
            encodings for. Default is 31, which is sufficient for a month of
            daily data.
        """
        super().__init__()
        self.pos = TemporalPositionalEncoding(embed_dim, max_len=max_T)
        self.scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),  # normalizing features
            nn.Linear(embed_dim, embed_dim),  # learns temporal feature transformation
            nn.GELU(),  # adds non-linearity to capture complex temporal patterns
            nn.Linear(embed_dim, 1),  # project to a single score
        )

    def forward(self, x, T, H, W):
        """
        Args:
            x: (B, T*H*W, C) containing spatio-temporal tokens
            T: number of temporal tokens per spatial location (here days)
            H: spatial height after patching
            W: spatial width after patching
        Returns:
            Tensor of shape (B, H*W, C) with one temporally aggregated token
            per spatial location
        """
        seq = rearrange(x, "b (t h w) c -> b (h w) t c", t=T, h=H, w=W)
        pe = self.pos(T).to(seq.device).to(seq.dtype)
        seq = seq + pe.unsqueeze(0).unsqueeze(0)
        weights = torch.softmax(self.scorer(seq).squeeze(-1), dim=-1)
        out = (seq * weights.unsqueeze(-1)).sum(dim=2)
        return out


class MonthlyConvDecoder(nn.Module):
    """Decoder to reconstruct 2D maps from patch tokens.

    The MonthlyConvDecoder converts latent patch tokens back to pixel space:
        - Applies a 1*1 convolution to mix features on the patch grid.
        - Uses a transposed convolution (deconvolution) to upsample tokens to the original spatial resolution.
        - Applies a small convolutional head to produce the final single-channel output.
        - Optionally masks out land regions using a boolean mask.
    """
    def __init__(self, embed_dim=128, patch_h=4, patch_w=4, hidden=128, overlap=1):
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
            bias=True
        )

        # Final conv head to get single channel output kernel_size=3 is the most
        # common choice for spatial convolutions; it's the smallest kernel that
        # captures spatial context in all directions
        in_channels, out_channels = hidden // 2, hidden // 2
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, 1, kernel_size=1),
        )

        # Learnable scale and bias (mean and std) to improve predictions
        self.scale = nn.Parameter(torch.ones(1))
        self.bias  = nn.Parameter(torch.zeros(1))

    def forward(self, latent, out_H, out_W, land_mask=None):
        """Reconstruct 2D maps from latent patch tokens.
        Args:
            latent: Tensor of shape (B, H'*W', C), where H'*W' = out_H//patch_h * out_W//patch_w
            out_H: Target output height (must be divisible by patch_h)
            out_W: Target output width (must be divisible by patch_w)
            land_mask: Optional boolean tensor of shape (out_H, out_W). Values set to True
                will be masked out (set to 0) in the output (only ocean pixels exist).
        Returns:
            Tensor of shape (B, out_H, out_W) representing the monthly SST.
        """
        B, HW, C = latent.shape
        Hp = out_H // self.patch_h
        Wp = out_W // self.patch_w
        assert Hp * Wp == HW, f"Token count mismatch: HW={HW}, Hp={Hp}, Wp={Wp}"

        #transforms the latent tensor from sequence format to image format for
        #convolution operations
        out = latent.view(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()  # (B, C, Hp, Wp)

        # Apply 1x1 convolution to mix features
        out = self.proj(out)        # (B, hidden, Hp, Wp)

        # Use transposed convolution to upsample
        out = self.deconv(out)      # (B, hidden//2, H, W)

        # Apply final conv head to get single channel output
        out = self.head(out)        # (B, 1, H, W)

        # Apply scale and bias
        out = out * self.scale + self.bias
        out = out.squeeze(1)        # (B, H, W)

        # Mask out land areas if land_mask is provided
        if land_mask is not None:
            out = out.masked_fill(land_mask.bool().unsqueeze(0), 0.0)
        return out


class SpatialPositionalEncoding2D(nn.Module):
    """2D Spatial Positional Encoding using sine and cosine functions.

    This module generates fixed sinusoidal positional encodings for a 2D spatial
    grid, following the formulation in "Attention Is All You Need" (Vaswani et al., 2017).

    The returned positional encodings are intended to be added to spatial tokens
    by the caller. The encodings are **not learnable**.
    """
    def __init__(self, embed_dim=128, max_H=1024, max_W=1024):
        """ Initialize the positional encoding.
        Args:
            embed_dim: Dimension of the embedding, it must be even. The default is 128.
                Embedding dimensions are usually multiples of 64 (e.g., 64, 128,
                256). This can be tuned.
            max_H: Maximum height. Default is 1024, which should be sufficient.
            max_W: Maximum width. Default is 1024, which should be sufficient.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_H = max_H
        self.max_W = max_W
        self.register_buffer("pe", self.build_pe(max_H, max_W, embed_dim), persistent=False)

    @staticmethod
    def build_pe(H, W, embed_dim):
        """ Build the 2D positional encoding encoding tensor.
        Args:
            H: Height of the grid
            W: Width of the grid
            embed_dim: Dimension of the embedding (must be even)
        Returns:
            Tensor of shape (H, W, embed_dim) containing fixed positional encodings.
            Encodings are constructed by combining sine/cosine encodings along
            height and width. Not learnable.
        """
        assert embed_dim % 2 == 0, "embed_dim must be even"
        pe_h = torch.zeros(H, embed_dim // 2)
        pe_w = torch.zeros(W, embed_dim // 2)
        pos_h = torch.arange(H).unsqueeze(1)
        pos_w = torch.arange(W).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim // 2, 2) * (-math.log(10000.0) / (embed_dim // 2)))
        pe_h[:, 0::2] = torch.sin(pos_h * div)
        pe_h[:, 1::2] = torch.cos(pos_h * div)
        pe_w[:, 0::2] = torch.sin(pos_w * div)
        pe_w[:, 1::2] = torch.cos(pos_w * div)
        pe_2d = (pe_h.unsqueeze(1) + pe_w.unsqueeze(0))  # (H, W, embed_dim/2)
        # concatenate to reach embed_dim
        pe = torch.cat([pe_2d, pe_2d], dim=-1)  # (H, W, embed_dim)
        return pe  # not learned

    def forward(self, Hp, Wp):
        """ Get positional encoding for size (Hp, Wp).
        Args:
            Hp: Height after patching (≤ max_H)
            Wp: Width after patching (≤ max_W)
        Returns:
            Tensor of shape (Hp*Wp, dim) containing positional encodings
            flattened in row-major order (height * width).
        """
        # returns (Hp*Wp, dim)
        pe_hw = self.pe[:Hp, :Wp, :].reshape(Hp * Wp, -1)
        return pe_hw


class SpatialTransformer(nn.Module):
    """Spatial Transformer for spatial feature mixing.

    This module applies a standard Transformer encoder to a sequence of spatial tokens
    (patch embeddings), allowing information to be mixed across all spatial locations.

    Key points:
        - Uses multi-head self-attention and feedforward layers.
        - Designed to operate on flattened spatial tokens.
    """
    def __init__(self, embed_dim=128, depth=2, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        """ Initialize the spatial transformer.
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
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True, dropout=dropout, activation="gelu"
        )
        # stack multiple layers to form the full encoder
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        """ Forward pass of the spatial transformer.
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
        T: temporal dimension (number of days)
        H: spatial height
        W: spatial width

    The model pipeline:
        1. Encode spatio-temporal patches using VideoEncoder.
        2. Aggregate temporal information for each spatial patch via TemporalAttentionAggregator.
        3. Add 2D spatial positional encodings and mix spatial features with SpatialTransformer.
        4. Decode aggregated tokens into a full-resolution 2D map using MonthlyConvDecoder.

    Output:
        - Reconstructed monthly SST map of shape (B, H, W)
    """
    def __init__(
            self,
            in_chans=1,
            embed_dim=128,
            patch_size=(1,4,4),
            max_T=64, hidden=128,
            overlap=1,
            max_H=1024,
            max_W=1024,
            spatial_depth=2,
            spatial_heads=4,
        ):
        """Initialize the Spatio-Temporal Model.

        Args:
            in_chans: Number of input channels (e.g., 1 for SST, additional channels possible)
            embed_dim: Dimension of the patch embedding
            patch_size: Tuple of (T, H, W) patch sizes for temporal and spatial patching
            max_T: Maximum temporal length for temporal positional encoding
            hidden: Hidden dimension used in the decoder
            overlap: Overlap for deconvolution in the decoder
            max_H: Maximum spatial height for 2D positional encoding
            max_W: Maximum spatial width for 2D positional encoding
            spatial_depth: Number of layers in the spatial Transformer
            spatial_heads: Number of attention heads in the spatial Transformer

        """
        super().__init__()
        self.encoder = VideoEncoder(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        self.temporal = TemporalAttentionAggregator(embed_dim=embed_dim, max_T=max_T)
        self.spatial_pe = SpatialPositionalEncoding2D(dim=embed_dim, max_H=max_H, max_W=max_W)
        self.spatial_tr = SpatialTransformer(embed_dim=embed_dim, depth=spatial_depth, num_heads=spatial_heads)
        self.decoder = MonthlyConvDecoder(
            embed_dim=embed_dim, patch_h=patch_size[1], patch_w=patch_size[2], hidden=hidden, overlap=overlap
        )
        self.patch_size = patch_size

    def forward(self, daily_data, daily_mask, land_mask_patch):
        """
        Forward pass of the Spatio-Temporal model.

        Args:
            daily_data: Tensor of shape (B, C, T, H, W) containing daily data
            daily_mask: Boolean tensor of same shape as daily_data indicating missing values
            land_mask_patch: Boolean tensor of shape (H, W) to mask land areas in the output

        Returns:
            monthly_pred: Tensor of shape (B, H, W) representing the reconstructed monthly map
        """
        B, C, T, H, W = daily_data.shape

        # Step 1: Encode spatio-temporal patches
        latent = self.encoder(daily_data, daily_mask)  # (B, T'*H'*W', C)
        Tp = T // self.patch_size[0]
        Hp = H // self.patch_size[1]
        Wp = W // self.patch_size[2]

        # Step 2: Aggregate temporal information for each spatial patch
        agg_latent = self.temporal(latent, Tp, Hp, Wp)

        # Step 3: Add spatial positional encodings and mix spatial features
        pe = self.spatial_pe(Hp, Wp).to(agg_latent.device).to(agg_latent.dtype)  # (Hp*Wp, C)
        x = agg_latent + pe.unsqueeze(0)

        # Step 4: Spatial mixing with Transformer
        x = self.spatial_tr(x)  # (B, Hp*Wp, C)

        # Step 5: Decode to full-resolution 2D map
        monthly_pred = self.decoder(x, H, W, land_mask_patch)
        return monthly_pred


def make_daily_mask(daily_ts, land_mask):
    """ mask True only for missing ocean pixels

    daily_ts: (T,H,W) float tensor
    land_mask: (H,W) or (1,H,W) bool, True=land
    """
    isnan = torch.isnan(daily_ts)

    # normalize land_mask to (H,W) and broadcast to (T,H,W)
    land2d = land_mask.squeeze(0) if land_mask.dim() == 3 else land_mask
    land2d = land2d.bool()
    land3d = land2d.unsqueeze(0).expand(daily_ts.size(0), -1, -1)

    mask = isnan & (~land3d)    # True only at ocean-missing
    return mask


def prepare_spatiotemporal_batch(
    daily_ts,      # (T,H,W) float32, may contain NaNs
    monthly_ts,    # (1,H,W) or (H,W) float32
    land_mask,     # (1,H,W) or (H,W) bool, True=land
    patch_size=(1,4,4),
):

    # convert to tensors
    daily_ts = torch.tensor(daily_ts, dtype=torch.float32, device=device)  # shape: (31, H, W)
    monthly_ts = torch.tensor(monthly_ts, dtype=torch.float32, device=device)  # shape: (1, H, W)
    land_mask = torch.tensor(land_mask, dtype=torch.bool, device=device) # shape (1, H, W), True = land

    # sanitize inputs
    assert daily_ts.dim() == 3, f"daily_ts must be (T,H,W), got {daily_ts.shape}"

    daily_mask = make_daily_mask(daily_ts, land_mask)  # (T,H,W) bool

    monthly = monthly_ts.squeeze(0) if monthly_ts.dim() == 3 else monthly_ts
    land = land_mask.squeeze(0) if land_mask.dim() == 3 else land_mask

    # replace NaNs in daily data; keep mask as source of truth
    daily_ts = torch.nan_to_num(daily_ts, nan=0.0)

    T, H, W = daily_ts.shape
    pt, ph, pw = patch_size
    assert T % pt == 0, f"T={T} not divisible by pt={pt}"
    assert H % ph == 0, f"H={H} not divisible by ph={ph}"
    assert W % pw == 0, f"W={W} not divisible by pw={pw}"

    return {
        "daily_data": daily_ts.unsqueeze(0).unsqueeze(0),   # (B=1,C=1,T,H,W),
        "daily_mask": daily_mask.unsqueeze(0).unsqueeze(0),  # (B=1,C=1,T,H,W),
        "monthly_target": monthly.unsqueeze(0),
        "land_mask": land,
    }


def pred_to_numpy(pred, orig_H=None, orig_W=None, land_mask=None):
    """
    pred: (B,H_pad,W_pad) or (B, H, W) torch tensor
    orig_H/W: original sizes before padding (optional)
    land_mask: (H_pad,W_pad) or (H,W) bool; if given, land will be set to NaN
    returns: (H,W) numpy array
    """
    # crop to original size if provided
    if orig_H is not None and orig_W is not None:
        pred = pred[..., :orig_H, :orig_W]
        if land_mask is not None:
            land_mask = land_mask[..., :orig_H, :orig_W]

    # set land to NaN (broadcast mask across batch)
    if land_mask is not None:
        pred = pred.clone().to(torch.float32)
        pred[:, land_mask.bool()] = float('nan')

    return pred.detach().cpu().numpy()