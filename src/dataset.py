import numpy as np
from utils import add_month_day_dims
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
        patch_size: Tuple[int, int] = (16, 16),
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

        # Reshape daily → (M, T=31, H, W), monthly → (M, H, W),
        # and get padded_days_mask → (M, T=31)
        daily_mt, monthly_m, padded_days_mask = add_month_day_dims(
            daily_da, monthly_da, time_dim=time_dim
        )

        # Convert to numpy once — all __getitem__ calls use these
        self.daily_np = daily_mt.to_numpy().copy()  # (M, T=31, H, W) float
        self.monthly_np = monthly_m.to_numpy().copy()  # (M, H, W) float
        self.padded_mask_np = padded_days_mask.to_numpy().copy()  # (M, T=31) bool

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
        self.stride = self.patch_size[0] - self.overlap
        H, W = self.daily_np.shape[2], self.daily_np.shape[3]
        self.n_i = (H - self.patch_size[0]) // self.stride + 1
        self.n_j = (W - self.patch_size[1]) // self.stride + 1

        # Total length is only spatial patches (all months included in each sample)
        self.total_len = self.n_i * self.n_j

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        """Get a spatiotemporal patch sample based on the index."""
        if idx < 0 or idx >= self.total_len:
            raise IndexError("Index out of range")

        i_idx, j_idx = divmod(idx, self.n_j)
        i = i_idx * self.stride
        j = j_idx * self.stride
        ph, pw = self.patch_size

        # Extract spatial patch via numpy slicing — faster than xarray indexing
        daily_patch = self.daily_np[:, :, i:i+ph, j:j+pw]  # (M, T, H, W)
        monthly_patch = self.monthly_np[:, i:i+ph, j:j+pw]  # (M, H, W)
        daily_nan_mask = self.daily_nan_mask[:, :, i:i+ph, j:j+pw]  # (M, T, H, W)

        if self.land_mask_np is not None:
            land_patch = self.land_mask_np[i:i+ph, j:j+pw]  # (H, W)
            land_tensor = torch.from_numpy(land_patch.copy()).bool()
        else:
            land_tensor = torch.zeros(ph, pw, dtype=torch.bool)

        # Convert to tensors (from_numpy is zero-copy on contiguous arrays)
        daily_tensor = torch.from_numpy(daily_patch).float().unsqueeze(0)   # (1, M, T, H, W)
        monthly_tensor = torch.from_numpy(monthly_patch).float()            # (M, H, W)
        daily_nan_mask = torch.from_numpy(daily_nan_mask).unsqueeze(0)     # (1, M, T, H, W)

        # daily_mask: NaN locations that are NOT land
        # Reshape land_tensor for broadcasting: (H, W) → (1, 1, 1, H, W)
        daily_mask_tensor = daily_nan_mask & (~land_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0))

        # Convert to tensors
        return {
            "daily_patch": daily_tensor,  # (C=1, M, T=31, H, W)
            "monthly_patch": monthly_tensor,  # (M, H, W)
            "daily_mask_patch": daily_mask_tensor,  # (C=1, M, T=31, H, W)
            "land_mask_patch": land_tensor,  # (H,W) True=Land
            "padded_days_mask": self.padded_days_tensor, # (M, T=31) True=padded
            "coords": (i, j),
        }
