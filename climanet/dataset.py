import warnings

import numpy as np
from .utils import add_month_day_dims, calc_stats, add_month_hour_dims
from .geo_embedding_utils import (
    calculate_sh_geo_pos_embeddings,
    compute_patch_geo_pos_embedding,
)
from .geo_embedding_utils import compute_patch_scale_features
import xarray as xr
import torch
from torch.utils.data import Dataset
from typing import Tuple


def _as_numpy(array_like, dtype=None, copy=False):
    """Convert numpy/dask/xarray-backed arrays to numpy, computing lazily when needed."""
    data = np.asarray(array_like)
    if dtype is not None:
        data = data.astype(dtype, copy=False)
    if copy:
        data = data.copy()
    return data


class STDataset(Dataset):
    """Dataset for spatiotemporal patches.

    This class provides a PyTorch Dataset interface for spatiotemporal data,
    allowing for the extraction of patches from daily/hourly and monthly data
    arrays. The `input_da` is expected to be a daily or hourly data array, while
    the `monthly_da` is a monthly data array. To extract monthly patches, the
    `input_da` and `monthly_da` are reshaped internally to have a month
    dimension, and the patches are extracted accordingly.
    """

    def __init__(
        self,
        input_da: xr.DataArray,
        monthly_da: xr.DataArray,
        land_mask: xr.DataArray = None,
        time_dim: str = "time",
        spatial_dims: Tuple[str, str] = ("lat", "lon"),
        patch_size: Tuple[int, int, int] = (1, 16, 16),  # (Month, lat, lon)
        stride: Tuple[int, int] = None,
        sh_pos_table: str = None,  # Optional; str formatted path to precomputed table of sh
        sh_embed_dim: int = 96,  # sh_embed_dim should <= (sh_order_L + 1)**2
        sh_order_L: int = 10,
        is_hourly: bool = False,
    ):
        """Initialize the dataset with daily and monthly data, and optional land mask.

        Args:
            input_da: xarray DataArray with daily data (time, H, W) or hourly data (time, H, W)
            monthly_da: xarray DataArray with monthly data (M, H, W)
            land_mask: Optional xarray DataArray with land mask (H, W) or (1, H, W)
            time_dim: Name of the time dimension in the input data
            spatial_dims: Tuple of (lat_dim, lon_dim) names in the input data
            patch_size: Tuple of (patch_time, patch_height, patch_width) in time
                unit and pixels in monthly data. For example, (1, 16, 16) means
                1 month, 16 pixels height, 16 pixels width. For this, the
                spatial resolution of `input_da` and `monthly_da` must match. To
                extract monthly patches, the `input_da` and `monthly_da` are
                reshaped internally to have a month dimension, and the patches are
                extracted accordingly.
            stride: Tuple of (stride_height, stride_width) in pixels. If None, defaults to patch_size (non-overlapping patches).
            is_hourly: Whether the daily data is hourly (T=31*24) or daily (T=31).

        """
        self.spatial_dims = spatial_dims
        self.patch_size = patch_size
        self.input_da = input_da
        self.monthly_da = monthly_da
        self.stride = stride if stride is not None else (patch_size[1], patch_size[2])

        self.sh_embed_dim = sh_embed_dim
        self.sh_order_L = sh_order_L

        # Check that the input data has the expected dimensions
        if time_dim not in input_da.dims or time_dim not in monthly_da.dims:
            raise ValueError(f"Time dimension '{time_dim}' not found in input data")
        for dim in spatial_dims:
            if dim not in input_da.dims or dim not in monthly_da.dims:
                raise ValueError(f"Spatial dimension '{dim}' not found in input data")

        if (
            patch_size[1] > input_da.sizes[spatial_dims[0]]
            or patch_size[2] > input_da.sizes[spatial_dims[1]]
        ):
            raise ValueError(
                f"Patch size {patch_size} is larger than data dimensions {input_da.sizes}"
            )

        if is_hourly:
            # hours_per_day == 24
            # Reshape daily → (M, T=31*24, H, W), monthly → (M, H, W),
            # and get padded_days_mask → (M, T=31*24)
            daily_mt, monthly_m, padded_days_mask, daily_timef = add_month_hour_dims(
                input_da, monthly_da, time_dim=time_dim
            )
        else:
            # Reshape daily → (M, T=31, H, W), monthly → (M, H, W),
            # and get padded_days_mask → (M, T=31)
            daily_mt, monthly_m, padded_days_mask, daily_timef = add_month_day_dims(
                input_da, monthly_da, time_dim=time_dim
            )

        # Keep xarray objects (potentially dask-backed) lazy; materialize per patch in __getitem__
        self.daily_mt = daily_mt  # (M, T=31, H, W) or (M, T=31*24, H, W)
        self.monthly_m = monthly_m  # (M, H, W)
        self.padded_days_mask = padded_days_mask  # (M, T)
        self.daily_timef = daily_timef  # (M, T, 3)

        # Store coordinate arrays
        self.lat_coords = input_da[spatial_dims[0]].to_numpy().copy()
        self.lon_coords = input_da[spatial_dims[1]].to_numpy().copy()

        if land_mask is not None:
            lm = torch.from_numpy(land_mask.values.copy()).bool()
            if lm.ndim == 3:
                lm = lm.squeeze(0)  # (1, H, W) → (H, W)
            self.land_mask_t = lm
        else:
            self.land_mask_t = None

        # Stats will be set later via set_stats() for train/test datasets
        self.daily_mean = None
        self.daily_std = None

        # Pre-build zero land tensor for the no-mask case
        _, ph, pw = self.patch_size
        self._zero_land = torch.zeros(ph, pw, dtype=torch.bool)

        # Precompute lazy index mapping for patches from data shape metadata
        M = int(self.daily_mt.sizes[self.daily_mt.dims[0]])
        H = int(self.daily_mt.sizes[spatial_dims[0]])
        W = int(self.daily_mt.sizes[spatial_dims[1]])
        self.patch_indices = self._compute_patch_indices(M, H, W)

        # Precompute geoposition and scale embeddings for patches
        self.sh_geo_pos = None
        self.geo_pos_table = self._get_geo_pos(sh_pos_table)
        self.patch_geo_embeddings, self.patch_scale_features = (
            self._compute_geoscalepatch_embeddings()
        )
        self.scale_f_dim = torch.tensor(self.patch_scale_features.shape[-1])
        self.sh_embed_dim_t = torch.tensor(self.sh_embed_dim)
        self.harmonic_order_t = torch.tensor(self.sh_order_L)

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

    def _compute_patch_indices(self, M: int, H: int, W: int) -> list:
        """Generate patch start indices with coverage warning (overlap support)."""
        pm, ph, pw = self.patch_size
        sh, sw = self.stride

        # validate temporal patch size
        if pm > M:
            raise ValueError(
                f"Temporal patch size {pm} is larger than available months {M}."
            )
        if pm < 1:
            raise ValueError(f"Temporal patch size {pm} must be at least 1.")

        # Validate stride
        if sh > ph or sw > pw:
            warnings.warn(
                f"Stride {self.stride} is larger than patch size {self.patch_size}. "
                f"This will leave gaps between patches.",
                UserWarning,
            )

        # Compute patch start indices using stride
        # Ensure we don't go out of bounds
        m_starts = list(
            range(0, M - pm + 1, pm)
        )  # Temporal patches are non-overlapping
        i_starts = list(range(0, H - ph + 1, sh))
        j_starts = list(range(0, W - pw + 1, sw))

        # Warn if there's incomplete coverage at the edges
        if not i_starts or not j_starts or not m_starts:
            raise ValueError(
                f"No valid patches can be extracted. Image size ({M}, {H}, {W}) "
                f"is smaller than patch size {self.patch_size}."
            )

        # Check edge coverage
        last_m = m_starts[-1] + pm
        last_i = i_starts[-1] + ph
        last_j = j_starts[-1] + pw
        if last_m < M or last_i < H or last_j < W:
            warnings.warn(
                f"Patches do not fully cover the image. "
                f"Uncovered pixels: {M - last_m} in time, {H - last_i} in height, {W - last_j} in width. "
                f"Consider adjusting stride or adding edge patches.",
                UserWarning,
            )

        overlap_h = ph - sh if sh < ph else 0
        overlap_w = pw - sw if sw < pw else 0

        len_m = len(m_starts)
        len_i = len(i_starts)
        len_j = len(j_starts)
        print(
            f"Patch grid (m x i x j): {len_m} x {len_i} x {len_j} = {len_m * len_i * len_j} patches"
        )
        print(f"Overlap: {overlap_h} pixels (height), {overlap_w} pixels (width)")

        return [(m, i, j) for m in m_starts for i in i_starts for j in j_starts]

    def _compute_geoscalepatch_embeddings(self):
        patch_geo_embeddings = []
        patch_scale_features = []

        for _, i, j in self.patch_indices:
            _, ph, pw = self.patch_size
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

        m, i, j = self.patch_indices[idx]
        pm, ph, pw = self.patch_size

        # Compute only the requested patch from potentially dask-backed arrays.
        daily_np_patch = _as_numpy(
            self.daily_mt.isel(
                {
                    self.daily_mt.dims[0]: slice(m, m + pm),
                    self.daily_mt.dims[2]: slice(i, i + ph),
                    self.daily_mt.dims[3]: slice(j, j + pw),
                }
            ).data,
            dtype=np.float32,
            copy=False,
        )

        # Build missing-data mask before NaN fill; True where NaN.
        daily_nan_mask_t_patch = torch.from_numpy(np.isnan(daily_np_patch)).unsqueeze(0)

        # Fill NaNs for model input.
        np.nan_to_num(daily_np_patch, copy=False, nan=0.0)
        daily_t_patch = torch.from_numpy(daily_np_patch).unsqueeze(0)

        monthly_np_patch = _as_numpy(
            self.monthly_m.isel(
                {
                    self.monthly_m.dims[0]: slice(m, m + pm),
                    self.monthly_m.dims[1]: slice(i, i + ph),
                    self.monthly_m.dims[2]: slice(j, j + pw),
                }
            ).data,
            dtype=np.float32,
            copy=False,
        )
        monthly_t_patch = torch.from_numpy(monthly_np_patch)

        if self.land_mask_t is not None:
            land_t_patch = self.land_mask_t[i : i + ph, j : j + pw]  # (H, W)
        else:
            land_t_patch = self._zero_land

        # daily_mask: NaN locations that are NOT land
        # Reshape land_tensor for broadcasting: (pH, pW) → (1, 1, 1, pH, pW)
        daily_mask_t_patch = daily_nan_mask_t_patch & (
            ~land_t_patch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )

        # Extract lat/lon coordinates for this patch
        lat_patch = self.lat_coords[i : i + ph]  # (H,) -> (pH,)
        lon_patch = self.lon_coords[j : j + pw]  # (W,) -> (pW,)

        # get patch geo pos embedding
        geo_pos_embedding_t = self.patch_geo_embeddings[idx]  # (sh_dim,)

        # get scale feature for patch
        scale_feature_t = self.patch_scale_features[idx]  # (10,)

        # Convert to tensors
        return {
            "daily_patch": daily_t_patch,  # (C=1, pm, T=31, pH, pW)
            "monthly_patch": monthly_t_patch,  # (pm, pH, pW)
            "daily_mask_patch": daily_mask_t_patch,  # (C=1, pm, T=31, pH, pW)
            "land_mask_patch": land_t_patch,  # (pH,pW) True=Land
            "daily_timef_patch": torch.from_numpy(
                _as_numpy(
                    self.daily_timef.isel(
                        {self.daily_timef.dims[0]: slice(m, m + pm)}
                    ).data,
                    dtype=np.float32,
                    copy=False,
                )
            ),  # (pm, T=31, 3)
            "padded_days_mask": torch.from_numpy(
                _as_numpy(
                    self.padded_days_mask.isel(
                        {self.padded_days_mask.dims[0]: slice(m, m + pm)}
                    ).data,
                    copy=True,
                )
            ).bool(),  # (pm, T=31) True=padded
            "scale_feature_patch": scale_feature_t,  # (10,)
            "geo_pos_embedding_patch": geo_pos_embedding_t,  # (sh_embed_dim,)
            "sh_embed_dim": self.sh_embed_dim_t,
            "harmonic_order": self.harmonic_order_t,
            "scale_f_dim": self.scale_f_dim,
            "coords": torch.tensor([m, i, j]),
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
            data = _as_numpy(
                self.monthly_m.data, dtype=np.float32, copy=False
            )  # (M, H, W)
        else:
            # Stack selected spatial patches
            pm, ph, pw = self.patch_size
            patches = []
            for idx in indices:
                m, i, j = self.patch_indices[idx]
                patch = _as_numpy(
                    self.monthly_m.isel(
                        {
                            self.monthly_m.dims[0]: slice(m, m + pm),
                            self.monthly_m.dims[1]: slice(i, i + ph),
                            self.monthly_m.dims[2]: slice(j, j + pw),
                        }
                    ).data,
                    dtype=np.float32,
                    copy=False,
                )
                patches.append(patch)
            data = np.concatenate(patches, axis=-1)

        mean, std = calc_stats(data)  # (pm,)

        self.daily_mean = mean
        self.daily_std = std

        return mean, std
