import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict


class SSTDataset(Dataset):
    """Dataset for spatiotemporal patches from NetCDF files."""

    def __init__(
        self,
        daily_files: List[Path],
        monthly_files: List[Path],
        mask_file: Optional[Path] = None,
        daily_var: str = "ts",
        monthly_var: str = "ts",
        mask_var: str = "lsm",
        patch_size: Tuple[int, int] = (16, 16),
        overlap: int = 0,
        spatial_subset: Optional[Dict[str, slice]] = None,
        transform=None,
    ):
        self.daily_files = daily_files
        self.monthly_files = monthly_files
        self.mask_file = mask_file
        self.daily_var = daily_var
        self.monthly_var = monthly_var
        self.mask_var = mask_var
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform

        # Open datasets
        self.daily_ds = xr.open_mfdataset(
            daily_files,
            chunks={"time": 1, "lat": self.patch_size[0], "lon": self.patch_size[1]},
        )

        self.monthly_ds = xr.open_mfdataset(
            monthly_files,
            chunks={"time": 1, "lat": self.patch_size[0], "lon": self.patch_size[1]},
        )

        if mask_file:
            self.mask_ds = xr.open_dataset(mask_file)
            # Mask should only have spatial dimensions, discard time if present
            if "time" in self.mask_ds.dims:
                self.mask_ds = self.mask_ds.isel(time=0)
        else:
            self.mask_ds = None

        # Apply spatial subset if provided
        if spatial_subset:
            self.daily_ds = self.daily_ds.sel(**spatial_subset)
            self.monthly_ds = self.monthly_ds.sel(**spatial_subset)
            if mask_file:
                self.mask_ds = self.mask_ds.sel(**spatial_subset)

        # Expand dimensions to include 'bands'
        self.daily_ds = self.daily_ds.expand_dims("bands", axis=0)
        self.monthly_ds = self.monthly_ds.expand_dims("bands", axis=0)

        # Calculate patch indices
        self.patches = self._compute_patch_indices()

    def _compute_patch_indices(self):
        """Compute starting indices for all patches."""
        # Get spatial dimensions
        lat_dim = self.daily_ds.sizes["lat"]
        lon_dim = self.daily_ds.sizes["lon"]
        time_dim = self.monthly_ds.sizes["time"]  # Use monthly for samples

        patches = []
        stride = self.patch_size[0] - self.overlap

        for t in range(time_dim):
            for i in range(0, lat_dim - self.patch_size[0] + 1, stride):
                for j in range(0, lon_dim - self.patch_size[1] + 1, stride):
                    patches.append((t, i, j))

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        t, i, j = self.patches[idx]

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

    def close(self):
        self.daily_ds.close()
        self.monthly_ds.close()
        if self.mask_file:
            self.mask_ds.close()
