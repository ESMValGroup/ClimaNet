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
        self.land_mask = land_mask
        self.time_dim = time_dim
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.overlap = overlap

        # Reshape daily → (M, T=31, H, W), monthly → (M, H, W),
        # and get padded_days_mask → (M, T=31)
        self.daily_mt, self.monthly_m, self.padded_days_mask = add_month_day_dims(
            daily_da, monthly_da, time_dim=time_dim
        )

        # Precompute lazy index mapping for patches
        dim_y, dim_x = self.spatial_dims
        self.stride = self.patch_size[0] - self.overlap
        self.n_i = (
            self.daily_mt.sizes[dim_y] - self.patch_size[0]
        ) // self.stride + 1  # number of horizontal patches
        self.n_j = (
            self.daily_mt.sizes[dim_x] - self.patch_size[1]
        ) // self.stride + 1  # number of vertical patches

        # Total length is only spatial patches (all months included in each sample)
        self.total_len = self.n_i * self.n_j

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        """Get a spatiotemporal patch sample based on the index."""
        if idx < 0 or idx >= self.total_len:
            raise IndexError("Index out of range")

        dim_y, dim_x = self.spatial_dims
        per_t = self.n_i * self.n_j
        i_idx, j_idx = divmod(idx, self.n_j)
        i = i_idx * self.stride
        j = j_idx * self.stride

        # Extract spatial patch
        y_slice = slice(i, i + self.patch_size[0])
        x_slice = slice(j, j + self.patch_size[1])

        # Get daily data (all days in month)
        # Assuming monthly timestamp corresponds to days in that month
        daily_patch = self.daily_mt.isel(
            {dim_y: y_slice,dim_x: x_slice,}
        ).to_numpy()  # (M, T=31, H, W)

        # Add channel dim → (C=1, M, T=31, H, W)
        daily_patch = torch.from_numpy(daily_patch.copy()).float().unsqueeze(0)

        # Get monthly target
        monthly_patch = self.monthly_m.isel(
            {dim_y: y_slice,dim_x: x_slice,}
        ).to_numpy()  # (M, H, W)
        monthly_patch = torch.from_numpy(monthly_patch.copy()).float()

        if self.land_mask is not None:
            land_mask_patch = self.land_mask.isel(
                {dim_y: y_slice, dim_x: x_slice}
            ).to_numpy()
            land_mask_patch = torch.from_numpy(land_mask_patch.copy()).bool()  # (H,W)
        else:
            # No land mask → all ocean (False)
            land_mask_patch = torch.zeros(
                self.patch_size[0], self.patch_size[1], dtype=torch.bool
            )

        daily_mask_patch = torch.isnan(daily_patch) & (~land_mask_patch)

        # Replace NaNs in daily data with zeros (after creating mask)
        daily_patch = torch.nan_to_num(daily_patch, nan=0.0)

        # Padded days mask — same for all spatial patches
        padded_days_mask = torch.from_numpy(
            self.padded_days_mask.to_numpy().copy()
        ).bool()  # (M, T=31)

        # Convert to tensors
        sample = {
            "daily_patch": daily_patch,  # (C=1, M, T=31, H, W)
            "monthly_patch": monthly_patch,  # (M, H, W)
            "daily_mask_patch": daily_mask_patch,  # (C=1, M, T=31, H, W)
            "land_mask_patch": land_mask_patch.squeeze(),  # (H,W)
            "padded_days_mask": padded_days_mask, # (M, T) True=padded
            "coords": (i, j),
        }

        return sample
