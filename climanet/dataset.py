import numpy as np
from .utils import add_month_day_dims
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
        overlap: int = 0,
    ):
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.overlap = overlap

        # Check that the input data has the expected dimensions
        if time_dim not in daily_da.dims or time_dim not in monthly_da.dims:
            raise ValueError(f"Time dimension '{time_dim}' not found in input data")
        for dim in spatial_dims:
            if dim not in daily_da.dims or dim not in monthly_da.dims:
                raise ValueError(f"Spatial dimension '{dim}' not found in input data")

        if (
            patch_size[0] > daily_da.sizes[spatial_dims[0]] or patch_size[1] > daily_da.sizes[spatial_dims[1]]
        ):
            raise ValueError(
                f"Patch size {patch_size} is larger than data dimensions {daily_da.sizes[spatial_dims]}"
            )

        # Reshape daily → (M, T=31, H, W), monthly → (M, H, W),
        # and get padded_days_mask → (M, T=31)
        daily_mt, monthly_m, padded_days_mask = add_month_day_dims(
            daily_da, monthly_da, time_dim=time_dim
        )

        # Convert to numpy once — all __getitem__ calls use these
        self.daily_np = daily_mt.to_numpy().copy()  # (M, T=31, H, W) float
        self.monthly_np = monthly_m.to_numpy().copy()  # (M, H, W) float
        self.padded_mask_np = padded_days_mask.to_numpy().copy()  # (M, T=31) bool

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

        # Fill NaNs with 0 in-place
        np.nan_to_num(self.daily_np, copy=False, nan=0.0)

        # Precompute padded_days_mask as a tensor (same for all patches)
        self.padded_days_tensor = torch.from_numpy(self.padded_mask_np).bool()

        # Precompute lazy index mapping for patches
        self.stride = (patch_size[0] - overlap, patch_size[1] - overlap)
        H, W = self.daily_np.shape[2], self.daily_np.shape[3]
        self.patch_indices = self._compute_patch_indices(H, W)

    def _compute_patch_indices(self, H: int, W: int) -> list:
        """Generate patch start indices ensuring full coverage."""
        def get_starts(size, patch_len, stride):
            starts = list(range(0, size - patch_len + 1, stride))
            if not starts or starts[-1] + patch_len < size:
                starts.append(size - patch_len)
            return sorted(set(starts))

        i_starts = get_starts(H, self.patch_size[0], self.stride[0])
        j_starts = get_starts(W, self.patch_size[1], self.stride[1])
        return [(i, j) for i in i_starts for j in j_starts]


    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        """Get a spatiotemporal patch sample based on the index."""
        if idx < 0 or idx >= len(self.patch_indices):
            raise IndexError("Index out of range")

        i, j = self.patch_indices[idx]
        ph, pw = self.patch_size

        # Extract spatial patch via numpy slicing — faster than xarray indexing
        daily_patch = self.daily_np[:, :, i : i + ph, j : j + pw]  # (M, T, H, W)
        monthly_patch = self.monthly_np[:, i : i + ph, j : j + pw]  # (M, H, W)
        daily_nan_mask = self.daily_nan_mask[
            :, :, i : i + ph, j : j + pw
        ]  # (M, T, H, W)

        if self.land_mask_np is not None:
            land_patch = self.land_mask_np[i : i + ph, j : j + pw]  # (H, W)
            land_tensor = torch.from_numpy(land_patch.copy()).bool()
        else:
            land_tensor = torch.zeros(ph, pw, dtype=torch.bool)

        # Convert to tensors (from_numpy is zero-copy on contiguous arrays)
        # (1, M, T, H, W)
        daily_tensor = torch.from_numpy(daily_patch).float().unsqueeze(0)
        # (M, H, W)
        monthly_tensor = torch.from_numpy(monthly_patch).float()
        # (1, M, T, H, W)
        daily_nan_mask = torch.from_numpy(daily_nan_mask).unsqueeze(0)

        # daily_mask: NaN locations that are NOT land
        # Reshape land_tensor for broadcasting: (H, W) → (1, 1, 1, H, W)
        daily_mask_tensor = daily_nan_mask & (
            ~land_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )

        # Extract lat/lon coordinates for this patch
        lat_patch = self.lat_coords[i : i + ph]
        lon_patch = self.lon_coords[j : j + pw]

        # Convert to tensors
        return {
            "daily_patch": daily_tensor,  # (C=1, M, T=31, H, W)
            "monthly_patch": monthly_tensor,  # (M, H, W)
            "daily_mask_patch": daily_mask_tensor,  # (C=1, M, T=31, H, W)
            "land_mask_patch": land_tensor,  # (H,W) True=Land
            "padded_days_mask": self.padded_days_tensor,  # (M, T=31) True=padded
            "coords": (i, j),
            "lat_patch": lat_patch,  # (H,)
            "lon_patch": lon_patch,  # (W,)
        }
