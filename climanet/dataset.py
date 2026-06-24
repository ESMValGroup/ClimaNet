import warnings

import numpy as np
from .utils import add_month_day_dims, calc_stats
from .geo_embedding_utils import (
    calculate_sh_geo_pos_embeddings,
    compute_patch_geo_pos_embedding,
)
from .geo_embedding_utils import compute_patch_scale_features
import xarray as xr
import torch
from torch.utils.data import Dataset
from typing import Tuple


class STDataset(Dataset):
    """Dataset for spatiotemporal patches."""

    def __init__(
        self,
        daily_da: xr.DataArray,
        monthly_da: xr.DataArray,
        land_mask: xr.DataArray = None,
        time_dim: str = "time",
        spatial_dims: Tuple[str, str] = ("lat", "lon"),
        patch_size: Tuple[int, int] = (16, 16),  # (lat, lon)
        stride: Tuple[int, int] = None,
        sh_pos_table: str = None,  # Optional; str formatted path to precomputed table of sh
        sh_embed_dim: int = 96,  # sh_embed_dim should <= (sh_order_L + 1)**2
        sh_order_L: int = 10,
    ):
        """Initialize the dataset with daily and monthly data, land mask, and patching parameters.

        Parameters
        ----------
        daily_da : xr.DataArray
            Daily data array.
        monthly_da : xr.DataArray
            Monthly data array.
        land_mask : xr.DataArray, optional
            Land mask array, by default None
        time_dim : str, optional
            Name of the time dimension, by default "time"
        spatial_dims : Tuple[str, str], optional
            Names of the spatial dimensions, by default ("lat", "lon")
        patch_size : Tuple[int, int], optional
            Size of the patches, by default (16, 16)
        """
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.daily_da = daily_da
        self.monthly_da = monthly_da
        self.stride = stride if stride is not None else patch_size

        self.sh_embed_dim = sh_embed_dim
        self.sh_order_L = sh_order_L

        # Check that the input data has the expected dimensions
        if time_dim not in daily_da.dims or time_dim not in monthly_da.dims:
            raise ValueError(f"Time dimension '{time_dim}' not found in input data")
        for dim in spatial_dims:
            if dim not in daily_da.dims or dim not in monthly_da.dims:
                raise ValueError(f"Spatial dimension '{dim}' not found in input data")

        if (
            patch_size[0] > daily_da.sizes[spatial_dims[0]]
            or patch_size[1] > daily_da.sizes[spatial_dims[1]]
        ):
            raise ValueError(
                f"Patch size {patch_size} is larger than data dimensions {daily_da.sizes}"
            )

        # Reshape daily → (M, T=31, H, W), monthly → (M, H, W),
        # and get padded_days_mask → (M, T=31)
        daily_mt, monthly_m, padded_days_mask, daily_timef = add_month_day_dims(
            daily_da, monthly_da, time_dim=time_dim
        )

        # Convert to numpy once — all __getitem__ calls use these
        self.daily_np = daily_mt.to_numpy().copy().astype(np.float32)  # (M, T=31, H, W) float
        self.monthly_np = monthly_m.to_numpy().copy().astype(np.float32)  # (M, H, W) float
        self.padded_mask_np = padded_days_mask.to_numpy().copy()  # (M, T=31) bool
        self.daily_timef_np = daily_timef.to_numpy().copy().astype(np.float32) # (M,T=31, 4)

        # Store coordinate arrays
        self.lat_coords = daily_da[spatial_dims[0]].to_numpy().copy()
        self.lon_coords = daily_da[spatial_dims[1]].to_numpy().copy()

        if land_mask is not None:
            lm = land_mask.to_numpy().copy()
            if lm.ndim == 3:
                lm = lm.squeeze(0)  # (1, H, W) → (H, W)
            self.land_mask_np = lm
        else:
            self.land_mask_np = None

        # Precompute the NaN mask before filling NaNs
        # daily_mask: True where NaN (i.e. missing ocean data, not land)
        self.daily_nan_mask = np.isnan(self.daily_np)  # (M, T=31, H, W)

        # NaNs will be filled with 0 in-place
        np.nan_to_num(self.daily_np, copy=False, nan=0.0)

        # Stats will be set later via set_stats() for train/test datasets
        self.daily_mean = None
        self.daily_std = None

        # Precompute padded_days_mask as a tensor (same for all patches)
        self.padded_days_tensor = torch.from_numpy(self.padded_mask_np).bool()

        # Precompute lazy index mapping for patches
        H, W = self.daily_np.shape[2], self.daily_np.shape[3]
        self.patch_indices = self._compute_patch_indices(H, W)

        # Precompute geoposition and scale embeddings for patches
        self.sh_geo_pos = None
        self.geo_pos_table = self._get_geo_pos(sh_pos_table)
        self.patch_geo_embeddings, self.patch_scale_features = (
            self._compute_geoscalepatch_embeddings()
        )

    def _get_geo_pos(self, sh_pos_table: str):
        """Calculate or retrieve spherical harmonics based geo position embeddings."""
        if sh_pos_table is None:
            self.sh_geo_pos = calculate_sh_geo_pos_embeddings(
                self.lat_coords, self.lon_coords, self.sh_order_L, self.sh_embed_dim
            )
        else:
            # load then set embed dim and sh order L from here
            raise (RuntimeError("load method not implemented"))
            # TODO implement load functionality. loaded tensor should
            # be placed in self.sh_geo_pos. return sh_pos_table to
            # preserve provenance in dataset. IMPORTANT check
            # compatability of L and sh_dim between requested
            # and loaded. Raise error if not consistent

    def _compute_patch_indices(self, H: int, W: int) -> list:
        """Generate patch start indices with coverage warning (overlap support)."""
        ph, pw = self.patch_size
        sh, sw = self.stride

        # Validate stride
        if sh > ph or sw > pw:
            warnings.warn(
                f"Stride {self.stride} is larger than patch size {self.patch_size}. "
                f"This will leave gaps between patches.",
                UserWarning,
            )

        # Compute patch start indices using stride
        # Ensure we don't go out of bounds
        i_starts = list(range(0, H - ph + 1, sh))
        j_starts = list(range(0, W - pw + 1, sw))

        # Warn if there's incomplete coverage at the edges
        if not i_starts or not j_starts:
            raise ValueError(
                f"No valid patches can be extracted. Image size ({H}, {W}) "
                f"is smaller than patch size {self.patch_size}."
            )

        # Check edge coverage
        last_i = i_starts[-1] + ph
        last_j = j_starts[-1] + pw
        if last_i < H or last_j < W:
            warnings.warn(
                f"Patches do not fully cover the image. "
                f"Uncovered pixels: {H - last_i} in height, {W - last_j} in width. "
                f"Consider adjusting stride or adding edge patches.",
                UserWarning,
            )

        overlap_h = ph - sh if sh < ph else 0
        overlap_w = pw - sw if sw < pw else 0
        print(
            f"Patch grid: {len(i_starts)} x {len(j_starts)} = {len(i_starts) * len(j_starts)} patches"
        )
        print(f"Overlap: {overlap_h} pixels (height), {overlap_w} pixels (width)")

        return [(i, j) for i in i_starts for j in j_starts]

    def _compute_geoscalepatch_embeddings(self):
        patch_geo_embeddings = []
        patch_scale_features = []

        for i, j in self.patch_indices:
            ph, pw = self.patch_size
            geo_pos_tensor = self.sh_geo_pos[
                i : i + ph,
                j : j + pw,
            ]
            lat_patch = self.lat_coords[i : i + ph]
            lon_patch = self.lon_coords[j : j + pw]

            geo_emb = compute_patch_geo_pos_embedding(
                geo_pos_tensor,
                lat_patch,
            )
            scale_feat = compute_patch_scale_features(
                lat_patch,
                lon_patch,
            )

            patch_geo_embeddings.append(geo_emb)
            patch_scale_features.append(scale_feat)

        patch_geo_embeddings = torch.stack(patch_geo_embeddings)
        patch_scale_features = torch.stack(patch_scale_features)

        return patch_geo_embeddings, patch_scale_features

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        """Get a spatiotemporal patch sample based on the index."""

        if idx < 0 or idx >= len(self.patch_indices):
            raise IndexError("Index out of range")

        i, j = self.patch_indices[idx]
        ph, pw = self.patch_size

        # Extract spatial patch via numpy slicing — faster than xarray indexing
        daily_patch = self.daily_np[
            :, :, i : i + ph, j : j + pw
        ]  # (M, T, H, W) -> (M,T,pH, pW)
        monthly_patch = self.monthly_np[
            :, i : i + ph, j : j + pw
        ]  # (M, H, W) -> (M, pH, pW)
        daily_nan_mask = self.daily_nan_mask[
            :, :, i : i + ph, j : j + pw
        ]  # (M, T, H, W) -> (M, T, pH, pW)

        if self.land_mask_np is not None:
            land_patch = self.land_mask_np[i : i + ph, j : j + pw]  # (H, W)
            land_tensor = torch.from_numpy(np.ascontiguousarray(land_patch)).bool()
        else:
            land_tensor = torch.zeros(ph, pw, dtype=torch.bool)

        # geo_pos_tensor = self.sh_geo_pos[i: i + ph, j: j + pw] # (H,W, sh_emb_dim) -> (pH, pW, sh_embed_dim)

        # Convert to tensors (from_numpy is zero-copy on contiguous arrays)
        # (1, M, T, H, W)
        daily_tensor = torch.from_numpy(daily_patch).unsqueeze(0)
        # (M, H, W)
        monthly_tensor = torch.from_numpy(monthly_patch)
        # (1, M, T, H, W)
        daily_nan_mask = torch.from_numpy(daily_nan_mask).unsqueeze(0)
        # ( M, T, 2)
        daily_timef_tensor = torch.from_numpy(self.daily_timef_np)

        # daily_mask: NaN locations that are NOT land
        # Reshape land_tensor for broadcasting: (pH, pW) → (1, 1, 1, pH, pW)
        daily_mask_tensor = daily_nan_mask & (
            ~land_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )

        # Extract lat/lon coordinates for this patch
        lat_patch = self.lat_coords[i : i + ph]  # (H,) -> (pH,)
        lon_patch = self.lon_coords[j : j + pw]  # (W,) -> (pW,)

        # get patch geo pos embedding
        geo_pos_embedding_tensor = self.patch_geo_embeddings[idx]  # (sh_dim,)

        # get scale feature for patch
        scale_feature_tensor = self.patch_scale_features[idx]  # (10,)

        # create tensors to pass sh embedding dimension, harmonic order, and scale feature dim
        sh_embed_dim = torch.tensor(self.sh_embed_dim)
        harmonic_order = torch.tensor(self.sh_order_L)
        scale_f_dim = torch.tensor(len(scale_feature_tensor))

        # Convert to tensors
        return {
            "daily_patch": daily_tensor,  # (C=1, M, T=31, pH, pW)
            "monthly_patch": monthly_tensor,  # (M, pH, pW)
            "daily_mask_patch": daily_mask_tensor,  # (C=1, M, T=31, pH, pW)
            "land_mask_patch": land_tensor,  # (pH,pW) True=Land
            "daily_timef_patch": daily_timef_tensor,  # (M, T=31, 2)
            "padded_days_mask": self.padded_days_tensor,  # (M, T=31) True=padded
            "scale_feature_patch": scale_feature_tensor,  # (10,)
            "geo_pos_embedding_patch": geo_pos_embedding_tensor,  # (sh_embed_dim,)
            "sh_embed_dim": sh_embed_dim,
            "harmonic_order": harmonic_order,
            "scale_f_dim": scale_f_dim,
            "coords": (i, j),
            "lat_patch": lat_patch,  # (pH,)
            "lon_patch": lon_patch,  # (pW,)
        }

    def compute_stats(self, indices: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std from specified indices (or all data if None).

        Args:
            indices: List of patch indices to compute stats from. If None, use all.

        Returns:
            Tuple of (mean, std) arrays
        """
        if indices is None:
            data = self.monthly_np  # (M, H, W)
        else:
            # Stack selected spatial patches
            ph, pw = self.patch_size
            patches = []
            for idx in indices:
                i, j = self.patch_indices[idx]
                patch = self.monthly_np[:, i : i + ph, j : j + pw]
                patches.append(patch)
            data = np.concatenate(patches, axis=-1)

        mean, std = calc_stats(data)  # (M,)

        self.daily_mean = mean
        self.daily_std = std

        return mean, std
