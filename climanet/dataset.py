import warnings
from dataclasses import dataclass

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from .geo_embedding_utils import (
    calculate_sh_geo_pos_embeddings,
    compute_patch_geo_pos_embedding,
    compute_patch_scale_features,
)


@dataclass
class DataLoaderConfig:
    """Configuration for the data loader."""

    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = False
    persistent_workers: bool = True  # True when num_workers > 0
    device: str = "cpu"  # or "cuda"
    multiprocessing_context: str = "spawn"


class STDataset(Dataset):
    """Dataset for spatiotemporal data crops.

    This class provides a PyTorch Dataset interface for spatiotemporal data,
    allowing for the extraction of data crops from daily/hourly and monthly data
    arrays. The `input_da` is expected to be a daily or hourly data array, while
    the `monthly_da` is a monthly data array. To extract monthly data crops, the
    `input_da` and `monthly_da` are reshaped internally to have a month
    dimension, and the data crops are extracted accordingly.
    """

    def __init__(
        self,
        input_da: xr.DataArray,
        input_da_nan_mask: xr.DataArray,
        monthly_da: xr.DataArray,
        padded_days_mask: xr.DataArray,
        time_features: xr.DataArray,
        land_mask: xr.DataArray = None,
        spatial_dims: tuple[str, str] = ("lat", "lon"),
        crop_size: tuple[int, int, int] = (1, 16, 16),  # (Month, lat, lon)
        stride: tuple[int, int] = None,
        sh_embed_dim: int = 96,  # sh_embed_dim should <= (sh_order_L + 1)**2
        sh_order_L: int = 10,
        verbose: bool = False,
        load_lazy: bool = False,
    ):
        """Initialize the dataset with daily and monthly data, and optional land mask.

        Args:
            input_da: xarray DataArray with daily data (time, H, W) or hourly data (time, H, W)
            input_da_nan_mask: xarray DataArray with NaN mask for input_da (time, H, W)
            monthly_da: xarray DataArray with monthly data (M, H, W)
            padded_days_mask: xarray DataArray with padded days mask for input_da (time, H, W)
            land_mask: Optional xarray DataArray with land mask (H, W) or (1, H, W)
            spatial_dims: Tuple of (lat_dim, lon_dim) names in the input data
            crop_size: Tuple of (crop_time, crop_height, crop_width) in time
                unit and pixels in monthly data. For example, (1, 16, 16) means
                1 month, 16 pixels height, 16 pixels width. For this, the
                spatial resolution of `input_da` and `monthly_da` must match. To
                extract monthly data crops, the `input_da` and `monthly_da` are
                reshaped internally to have a month dimension, and the data crops are
                extracted accordingly.
            stride: Tuple of (stride_height, stride_width) in pixels. If None, defaults to crop_size (non-overlapping data crops).
            sh_pos_table: Optional path to precomputed spherical harmonics position embeddings.
            sh_embed_dim: Dimension of the spherical harmonics embedding.
            sh_order_L: Order of the spherical harmonics.
            verbose: If True, print dataset creation details.
            load_lazy: If True, use data lazily with zarr backend. This may slow down getitem but saves memory.
        """
        self.spatial_dims = spatial_dims
        self.crop_size = crop_size
        self.input_da = input_da
        self.input_da_nan_mask = input_da_nan_mask
        self.monthly_da = monthly_da
        self.padded_days_mask = padded_days_mask
        self.time_features = time_features
        self.land_mask = land_mask

        self.stride = stride if stride is not None else (crop_size[1], crop_size[2])

        self.sh_embed_dim = sh_embed_dim
        self.sh_order_L = sh_order_L
        self.verbose = verbose
        self.load_lazy = load_lazy

        # Check that the input data has the expected dimensions
        for dim in spatial_dims:
            if dim not in input_da.dims or dim not in monthly_da.dims:
                raise ValueError(f"Spatial dimension '{dim}' not found in input data")

        if (
            crop_size[1] > input_da.sizes[spatial_dims[0]]
            or crop_size[2] > input_da.sizes[spatial_dims[1]]
        ):
            raise ValueError(
                f"Crop size {crop_size} is larger than data dimensions {input_da.sizes}"
            )

        # Materialize data arrays to contiguous tensors for efficient access
        # Note: This may consume significant memory for large datasets.
        # Note: with load_lazy getitem becomes slower
        if self.load_lazy:
            self.daily_data_t = None
            self.daily_nan_mask_t = None
            self.monthly_data_t = None
            self.land_mask_t = None
            self.padded_days_t = None
            self.daily_timef_t = None
        else:
            self.daily_data_t = torch.from_numpy(self.input_da.to_numpy()).contiguous()
            self.daily_nan_mask_t = torch.from_numpy(
                self.input_da_nan_mask.to_numpy()
            ).contiguous()
            self.monthly_data_t = torch.from_numpy(
                self.monthly_da.to_numpy()
            ).contiguous()
            self.land_mask_t = self._prepare_land_mask(self.land_mask)
            self.padded_days_t = torch.from_numpy(
                self.padded_days_mask.to_numpy()
            ).bool()
            self.daily_timef_t = torch.from_numpy(
                self.time_features.to_numpy().astype(np.float32, copy=False)
            ).contiguous()

        # Store coordinate arrays
        self.lat_coords = torch.from_numpy(input_da[spatial_dims[0]].to_numpy().copy())
        self.lon_coords = torch.from_numpy(input_da[spatial_dims[1]].to_numpy().copy())

        # Pre-build zero land tensor for the no-mask case
        _, ph, pw = self.crop_size
        self._zero_land = torch.zeros(ph, pw, dtype=torch.bool)

        # Precompute lazy index mapping for data crops
        M, H, W = self.input_da.shape[0], self.input_da.shape[2], self.input_da.shape[3]
        self.crop_indices = self._compute_crop_indices(M, H, W)

        self.sh_embed_dim_t = torch.tensor(self.sh_embed_dim)
        self.harmonic_order_t = torch.tensor(self.sh_order_L)

        self.geo_pos_t = calculate_sh_geo_pos_embeddings(
            self.lat_coords, self.lon_coords, self.sh_order_L, self.sh_embed_dim
        )

    def _compute_crop_indices(self, M: int, H: int, W: int) -> list:
        """Generate crop start indices with coverage warning (overlap support)."""
        pm, ph, pw = self.crop_size
        sh, sw = self.stride

        # validate temporal crop size
        if pm > M:
            raise ValueError(
                f"Temporal crop size {pm} is larger than available months {M}."
            )
        if pm < 1:
            raise ValueError(f"Temporal crop size {pm} must be at least 1.")

        # Validate stride
        if sh > ph or sw > pw:
            warnings.warn(
                f"Stride {self.stride} is larger than crop size {self.crop_size}. "
                f"This will leave gaps between data crops.",
                UserWarning,
            )

        # Compute crop start indices using stride
        # Ensure we don't go out of bounds
        m_starts = list(
            range(0, M - pm + 1, pm)
        )  # Temporal data crops are non-overlapping
        i_starts = list(range(0, H - ph + 1, sh))
        j_starts = list(range(0, W - pw + 1, sw))

        # Warn if there's incomplete coverage at the edges
        if not i_starts or not j_starts or not m_starts:
            raise ValueError(
                f"No valid data crops can be extracted. Image size ({M}, {H}, {W}) "
                f"is smaller than crop size {self.crop_size}."
            )

        # Check edge coverage
        last_m = m_starts[-1] + pm
        last_i = i_starts[-1] + ph
        last_j = j_starts[-1] + pw
        if last_m < M or last_i < H or last_j < W:
            warnings.warn(
                f"Data crops do not fully cover the image. "
                f"Uncovered pixels: {M - last_m} in month, {H - last_i} in height, {W - last_j} in width. "
                f"Consider adjusting stride or adding edge data crops.",
                UserWarning,
            )

        overlap_h = ph - sh if sh < ph else 0
        overlap_w = pw - sw if sw < pw else 0

        len_m = len(m_starts)
        len_i = len(i_starts)
        len_j = len(j_starts)
        if self.verbose:
            print("Creating dataset:")
            print(
                f"Crop grid (m x i x j): {len_m} x {len_i} x {len_j} = {len_m * len_i * len_j} data crops"
            )
            print(f"Overlap: {overlap_h} pixels (height), {overlap_w} pixels (width)")

        return [(m, i, j) for m in m_starts for i in i_starts for j in j_starts]

    def _prepare_land_mask(self, land_mask):
        """Convert land mask to tensor."""
        if land_mask is None:
            return None

        lm = torch.as_tensor(land_mask.to_numpy(), dtype=torch.bool)
        if lm.ndim == 3:
            lm = lm.squeeze(0)  # (1, H, W) → (H, W)
        return lm

    def __len__(self):
        return len(self.crop_indices)

    def __getitem__(self, idx):
        """Get a spatiotemporal crop sample based on the index."""

        if idx < 0 or idx >= len(self.crop_indices):
            raise IndexError("Index out of range")

        m, i, j = self.crop_indices[idx]
        pm, ph, pw = self.crop_size

        if self.load_lazy:
            daily_t_crop = self.input_da.isel(
                M=slice(m, m + pm),
                **{
                    self.spatial_dims[0]: slice(i, i + ph),
                    self.spatial_dims[1]: slice(j, j + pw),
                },
            )
            daily_t_crop = (
                torch.from_numpy(daily_t_crop.to_numpy()).contiguous().unsqueeze(0)
            )

            daily_nan_mask_t_crop = self.input_da_nan_mask.isel(
                M=slice(m, m + pm),
                **{
                    self.spatial_dims[0]: slice(i, i + ph),
                    self.spatial_dims[1]: slice(j, j + pw),
                },
            )
            daily_nan_mask_t_crop = (
                torch.from_numpy(daily_nan_mask_t_crop.to_numpy())
                .contiguous()
                .unsqueeze(0)
            )

            monthly_t_crop = self.monthly_da.isel(
                M=slice(m, m + pm),
                **{
                    self.spatial_dims[0]: slice(i, i + ph),
                    self.spatial_dims[1]: slice(j, j + pw),
                },
            )
            monthly_t_crop = torch.from_numpy(monthly_t_crop.to_numpy()).contiguous()

            if self.land_mask is not None:
                land_t_crop = self.land_mask.isel(
                    **{
                        self.spatial_dims[0]: slice(i, i + ph),
                        self.spatial_dims[1]: slice(j, j + pw),
                    },
                )
                land_t_crop = self._prepare_land_mask(land_t_crop)

            daily_timef_crop = self.time_features.isel(M=slice(m, m + pm))
            daily_timef_crop = torch.from_numpy(
                daily_timef_crop.to_numpy().astype(np.float32, copy=False)
            ).contiguous()

            padded_days_mask_crop = self.padded_days_mask.isel(M=slice(m, m + pm))
            padded_days_mask_crop = torch.from_numpy(
                padded_days_mask_crop.to_numpy()
            ).bool()
        else:
            # Extract the crop data
            daily_t_crop = self.daily_data_t[
                m : m + pm, :, i : i + ph, j : j + pw
            ].unsqueeze(0)  # (1, pm, T, pH, pW)

            daily_nan_mask_t_crop = self.daily_nan_mask_t[
                m : m + pm, :, i : i + ph, j : j + pw
            ].unsqueeze(0)  # (1, pm, T, pH, pW)

            monthly_t_crop = self.monthly_data_t[m : m + pm, i : i + ph, j : j + pw]

            if self.land_mask_t is not None:
                land_t_crop = self.land_mask_t[i : i + ph, j : j + pw]
            else:
                land_t_crop = self._zero_land

            daily_timef_crop = self.daily_timef_t[m : m + pm]
            padded_days_mask_crop = self.padded_days_t[m : m + pm]

        # daily_mask: NaN locations that are NOT land
        # Reshape land_tensor for broadcasting: (pH, pW) → (1, 1, 1, pH, pW)
        daily_mask_t_crop = daily_nan_mask_t_crop & (
            ~land_t_crop.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )

        # Extract lat/lon coordinates for this crop
        lat_crop = self.lat_coords[i : i + ph]  # (H,) -> (pH,)
        lon_crop = self.lon_coords[j : j + pw]  # (W,) -> (pW,)

        geo_pos_embedding_t = compute_patch_geo_pos_embedding(
            self.geo_pos_t[i : i + ph, j : j + pw],
            lat_crop,
        )

        scale_feature_t = compute_patch_scale_features(
            lat_crop,
            lon_crop,
        )

        # Convert to dictionary
        return {
            "daily_patch": daily_t_crop,  # (C=1, pm, T=31, pH, pW)
            "monthly_patch": monthly_t_crop,  # (pm, pH, pW)
            "daily_mask_patch": daily_mask_t_crop,  # (C=1, pm, T=31, pH, pW)
            "land_mask_patch": land_t_crop,  # (pH,pW) True=Land
            "daily_timef_patch": daily_timef_crop,  # (pm, T=31, 3)
            "padded_days_mask": padded_days_mask_crop,  # (pm, T=31) True=padded
            "scale_feature_patch": scale_feature_t,  # (10,)
            "geo_pos_embedding_patch": geo_pos_embedding_t,  # (sh_embed_dim,)
            "sh_embed_dim": self.sh_embed_dim_t,
            "harmonic_order": self.harmonic_order_t,
            "coords": torch.tensor([m, i, j]),
            "lat_patch": lat_crop,  # (pH,)
            "lon_patch": lon_crop,  # (pW,)
        }

    def __getitems__(self, indices):
        return [self.__getitem__(i) for i in indices]
