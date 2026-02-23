import xarray as xr
import torch
from torch.utils.data import Dataset
from typing import Tuple


class SSTDataset(Dataset):
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
        self.daily_da = daily_da
        self.monthly_da = monthly_da
        self.land_mask = land_mask
        self.time_dim = time_dim
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.overlap = overlap

        # Group daily data
        # Create "YYYY-MM" string labels
        daily_labels = self.daily_da[time_dim].dt.strftime("%Y-%m")
        monthly_labels = self.monthly_da[time_dim].dt.strftime("%Y-%m")

        # Group daily indices by month label
        daily_groups = daily_labels.groupby(daily_labels).groups

        self.month_to_days = {}
        for month_idx, period in enumerate(monthly_labels.values):
            self.month_to_days[month_idx] = daily_groups.get(period, [])
            if len(self.month_to_days[month_idx]) == 0:
                raise ValueError(f"No daily data found for month index {month_idx}")

        # Precompute lazy index mapping for patches
        dim_y, dim_x = self.spatial_dims
        self.stride = self.patch_size[0] - self.overlap
        self.n_i = (
            self.daily_da.sizes[dim_y] - self.patch_size[0]
        ) // self.stride + 1  # number of horizontal patches
        self.n_j = (
            self.daily_da.sizes[dim_x] - self.patch_size[1]
        ) // self.stride + 1  # number of vertical patches
        self.total_len = len(self.monthly_da[time_dim]) * self.n_i * self.n_j

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        """Get a spatiotemporal patch sample based on the index."""
        if idx < 0 or idx >= self.total_len:
            raise IndexError("Index out of range")

        dim_y, dim_x = self.spatial_dims
        per_t = self.n_i * self.n_j
        t, rem = divmod(idx, per_t)
        i_idx, j_idx = divmod(rem, self.n_j)
        i = i_idx * self.stride
        j = j_idx * self.stride

        # Extract spatial patch
        y_slice = slice(i, i + self.patch_size[0])
        x_slice = slice(j, j + self.patch_size[1])

        # Get daily data (all days in month)
        # Assuming monthly timestamp corresponds to days in that month
        daily_patch = self.daily_da.isel(
            {
                self.time_dim: self.month_to_days[t],
                dim_y: y_slice,
                dim_x: x_slice,
            }
        ).to_numpy()  # shape: (T, H, W)

        # Add channel dim → (C=1, T, H, W)
        daily_patch = torch.from_numpy(daily_patch).float().unsqueeze(0)

        # Get monthly target
        monthly_patch = self.monthly_da.isel(
            {
                self.time_dim: t,
                dim_y: y_slice,
                dim_x: x_slice,
            }
        ).to_numpy()
        monthly_patch = torch.from_numpy(monthly_patch).float()

        if self.land_mask is not None:
            land_mask_patch = self.land_mask.isel(
                {dim_y: y_slice, dim_x: x_slice}
            ).to_numpy()
            land_mask_patch = torch.from_numpy(land_mask_patch).bool()  # (H,W)
        else:
            # No land mask → all ocean (False)
            land_mask_patch = torch.zeros(
                self.patch_size[0], self.patch_size[1], dtype=torch.bool
            )

        daily_mask_patch = torch.isnan(daily_patch) & (~land_mask_patch)

        # Replace NaNs in daily data with zeros (after creating mask)
        daily_patch = torch.nan_to_num(daily_patch, nan=0.0)

        # Convert to tensors
        sample = {
            "daily_patch": daily_patch,  # (C=1, T, H, W)
            "monthly_patch": monthly_patch,  # (H, W)
            "daily_mask_patch": daily_mask_patch,  # (C=1, T, H, W)
            "land_mask_patch": land_mask_patch.squeeze(),  # (H,W)
            "coords": (t, i, j),
        }

        return sample
