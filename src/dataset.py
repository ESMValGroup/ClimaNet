import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict


class SSTDataset(Dataset):
    """Dataset for spatiotemporal patches for Sea Surface Temperature (SST)."""

    def __init__(
        self,
        daily_ds: xr.Dataset,
        monthly_ds: xr.Dataset,
        mask_ds: xr.Dataset = None,
        daily_var: str = "ts",
        monthly_var: str = "ts",
        mask_var: str = "lsm",
        patch_size: Tuple[int, int] = (16, 16),
        overlap: int = 0,
        spatial_subset: Optional[Dict[str, slice]] = None,
        transform=None,
    ):
        self.daily_ds = daily_ds
        self.monthly_ds = monthly_ds
        self.mask_ds = mask_ds
        self.daily_var = daily_var
        self.monthly_var = monthly_var
        self.mask_var = mask_var
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform

        # Precompute lazy index mapping for patches
        lat_dim = self.daily_ds.sizes["lat"]
        lon_dim = self.daily_ds.sizes["lon"]
        self.time_dim = self.monthly_ds.sizes["time"]  # Use monthly for samples

        self.stride = self.patch_size[0] - self.overlap
        self.n_i = (lat_dim - self.patch_size[0]) // self.stride + 1
        self.n_j = (lon_dim - self.patch_size[1]) // self.stride + 1
        self.total_len = self.time_dim * self.n_i * self.n_j

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_len:
            raise IndexError("Index out of range")

        per_t = self.n_i * self.n_j
        t, rem = divmod(idx, per_t)
        i_idx, j_idx = divmod(rem, self.n_j)
        i = i_idx * self.stride
        j = j_idx * self.stride

        # Extract spatial patch
        lat_slice = slice(i, i + self.patch_size[0])
        lon_slice = slice(j, j + self.patch_size[1])

        # Get daily data (all days in month)
        # Assuming monthly timestamp corresponds to days in that month
        daily_patch = (
            self.daily_ds[self.daily_var].isel(lat=lat_slice, lon=lon_slice).to_numpy()
        )

        # Get monthly target
        monthly_patch = (
            self.monthly_ds[self.monthly_var]
            .isel(time=t, lat=lat_slice, lon=lon_slice)
            .to_numpy()
        )

        # Get mask patch if available
        if self.mask_ds is not None:
            mask_patch = (
                self.mask_ds[self.mask_var]
                .isel(lat=lat_slice, lon=lon_slice)
                .to_numpy()
            )
        else:
            mask_patch = np.ones_like(monthly_patch.isel(time=0), dtype=bool)

        # Convert to tensors
        sample = {
            "daily": torch.from_numpy(daily_patch).float(),
            "monthly": torch.from_numpy(monthly_patch).float(),
            "mask": torch.from_numpy(mask_patch).bool(),
            "coords": (t, i, j),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
