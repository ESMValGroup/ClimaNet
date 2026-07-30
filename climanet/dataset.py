import warnings
from collections import OrderedDict

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from .geo_embedding_utils import (
    calculate_sh_geo_pos_embeddings,
    compute_patch_geo_pos_embedding,
    compute_patch_scale_features,
)
from .utils import add_month_day_dims, add_month_hour_dims


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
        spatial_dims: tuple[str, str] = ("lat", "lon"),
        patch_size: tuple[int, int, int] = (1, 16, 16),  # (Month, lat, lon)
        stride: tuple[int, int] = None,
        sh_pos_table: str = None,  # Optional; str formatted path to precomputed table of sh
        sh_embed_dim: int = 96,  # sh_embed_dim should <= (sh_order_L + 1)**2
        sh_order_L: int = 10,
        is_hourly: bool = False,
        lazy_load: bool = False,
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
        self.land_mask = land_mask
        self.stride = stride if stride is not None else (patch_size[1], patch_size[2])

        self.sh_embed_dim = sh_embed_dim
        self.sh_order_L = sh_order_L
        self.lazy_load = lazy_load

        self._cache = OrderedDict()
        self._cache_size = 6

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
            self.daily_data, self.monthly_data, self.padded_days_mask, self.daily_timef = add_month_hour_dims(
                input_da, monthly_da, time_dim=time_dim
            )
        else:
            # Reshape daily → (M, T=31, H, W), monthly → (M, H, W),
            # and get padded_days_mask → (M, T=31)
            self.daily_data, self.monthly_data, self.padded_days_mask, self.daily_timef = add_month_day_dims(
                input_da, monthly_da, time_dim=time_dim
            )

        # keep data as xarray for lazy loading, but convert to tensor if not lazy
        if self.lazy_load:
            self.daily_data_t = None
            self.daily_nan_mask_t = None
            self.monthly_data_t = None
            self.land_mask_t = None
        else:
            self.daily_data_t, self.daily_nan_mask_t = self._prepare_daily_data(self.daily_data)
            self.monthly_data_t = self._prepare_monthly_data(self.monthly_data)
            self.land_mask_t = self._prepare_land_mask(self.land_mask)

        # other tensors that are always needed, even in lazy mode; not expensive to keep in memory
        self.padded_days_t = torch.from_numpy(self.padded_days_mask.to_numpy()).bool()
        self.daily_timef_t = torch.from_numpy(
            self.daily_timef.to_numpy().astype(np.float32, copy=False)
        ).contiguous()

        # Store coordinate arrays
        self.lat_coords = torch.from_numpy(input_da[spatial_dims[0]].to_numpy().copy())
        self.lon_coords = torch.from_numpy(input_da[spatial_dims[1]].to_numpy().copy())

        # Pre-build zero land tensor for the no-mask case
        _, ph, pw = self.patch_size
        self._zero_land = torch.zeros(ph, pw, dtype=torch.bool)

        # Precompute lazy index mapping for patches
        M, H, W = self.daily_data.shape[0], self.daily_data.shape[2], self.daily_data.shape[3]
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

        patch_geo_embeddings = torch.stack(patch_geo_embeddings).contiguous().clone()
        patch_scale_features = torch.stack(patch_scale_features).contiguous().clone()

        return patch_geo_embeddings, patch_scale_features

    def _prepare_daily_data(self, daily_data):
        """Convert daily data to tensor and precompute NaN mask."""
        daily_data_t = torch.from_numpy(
            daily_data.to_numpy().astype(np.float32, copy=False)
        ).contiguous()  # (M, T=31, H, W)

        # Precompute the NaN mask before filling NaNs
        # daily_mask: True where NaN (i.e. missing ocean data, not land)
        daily_nan_mask_t = torch.isnan(daily_data_t)  # (M, T=31, H, W)

        # NaNs will be filled with 0 in-place
        daily_data_t.nan_to_num_(nan=0.0)

        return daily_data_t, daily_nan_mask_t

    def _prepare_monthly_data(self, monthly_data):
        """Convert monthly data to tensor."""
        return torch.from_numpy(
            monthly_data.to_numpy().astype(np.float32, copy=False)
        ).contiguous()  # (M, H, W)

    def _prepare_land_mask(self, land_mask):
        """Convert land mask to tensor."""
        if land_mask is not None:
            lm = torch.as_tensor(land_mask.to_numpy(), dtype=torch.bool)
            if lm.ndim == 3:
                lm = lm.squeeze(0)  # (1, H, W) → (H, W)
            return lm
        else:
            return None

    def _load_partial(self, m: int):
        """Materialize one month and keep it in an LRU cache."""
        if m in self._cache:
            self._cache.move_to_end(m)
            return self._cache[m]

        daily_t, daily_nan_mask_t = self._prepare_daily_data(self.daily_data.isel(M=m))

        monthly_t = self._prepare_monthly_data(self.monthly_data.isel(M=m))

        if self.land_mask is not None:
            land_t = self._prepare_land_mask(self.land_mask)
        else:
            land_t = self._zero_land

        data_partial = {
            "daily": daily_t,
            "daily_nan_mask": daily_nan_mask_t,
            "monthly": monthly_t,
            "land": land_t,
        }

        # LRU update
        self._cache[m] = data_partial
        self._cache.move_to_end(m)

        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return data_partial

    def _get_patch(self, m: int, i: int, j: int):
        """Extract a spatiotemporal patch.

        Returns:
            daily_t: (pm, T, pH, pW)
            daily_nan_mask_t: (pm, T, pH, pW)
            monthly_t: (pm, pH, pW)
            land_t: (pH, pW)
        """
        pm, ph, pw = self.patch_size

        if self.lazy_load:
            if pm != 1:
                raise NotImplementedError(
                    "Month cache currently supports patch_size[0] == 1."
                )

            # get the month data from cache or load it if not present
            data_partial = self._load_partial(m)

            daily_t = data_partial["daily"][:, i : i + ph, j : j + pw]
            daily_nan_mask_t = data_partial["daily_nan_mask"][:, i : i + ph, j : j + pw]
            monthly_t = data_partial["monthly"][i : i + ph, j : j + pw]

            if self.land_mask is not None:
                land_t = data_partial["land"][i : i + ph, j : j + pw]
            else:
                land_t = self._zero_land

        else:
            # Slice pre-materialized tensors
            daily_t = self.daily_data_t[m : m + pm, :, i : i + ph, j : j + pw]
            daily_nan_mask_t = self.daily_nan_mask_t[m : m + pm, :, i : i + ph, j : j + pw]

            monthly_t = self.monthly_data_t[m : m + pm, i : i + ph, j : j + pw]

            if self.land_mask_t is not None:
                land_t = self.land_mask_t[i : i + ph, j : j + pw]
            else:
                land_t = self._zero_land

        return daily_t, daily_nan_mask_t, monthly_t, land_t

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        """Get a spatiotemporal patch sample based on the index."""

        if idx < 0 or idx >= len(self.patch_indices):
            raise IndexError("Index out of range")

        m, i, j = self.patch_indices[idx]
        pm, ph, pw = self.patch_size

        # Extract the patch data
        daily_t_patch, daily_nan_mask_t_patch, monthly_t_patch, land_t_patch = self._get_patch(m, i, j)
        daily_t_patch = daily_t_patch.unsqueeze(0)  # (1, pm, T, pH, pW)
        daily_nan_mask_t_patch = daily_nan_mask_t_patch.unsqueeze(0)  # (1, pm, T, pH, pW)

        # daily_mask: NaN locations that are NOT land
        # Reshape land_tensor for broadcasting: (pH, pW) → (1, 1, 1, pH, pW)
        daily_mask_t_patch = daily_nan_mask_t_patch & (
            ~land_t_patch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )

        daily_timef_patch = self.daily_timef_t[m : m + pm]
        padded_days_mask_patch = self.padded_days_t[m : m + pm]

        # Extract lat/lon coordinates for this patch
        lat_patch = self.lat_coords[i : i + ph]  # (H,) -> (pH,)
        lon_patch = self.lon_coords[j : j + pw]  # (W,) -> (pW,)

        # get patch geo pos embedding
        geo_pos_embedding_t = self.patch_geo_embeddings[idx]  # (sh_dim,)

        # get scale feature for patch
        scale_feature_t = self.patch_scale_features[idx]  # (10,)

        # Convert to dictionary
        return {
            "daily_patch": daily_t_patch,  # (C=1, pm, T=31, pH, pW)
            "monthly_patch": monthly_t_patch,  # (pm, pH, pW)
            "daily_mask_patch": daily_mask_t_patch,  # (C=1, pm, T=31, pH, pW)
            "land_mask_patch": land_t_patch,  # (pH,pW) True=Land
            "daily_timef_patch": daily_timef_patch,  # (pm, T=31, 3)
            "padded_days_mask": padded_days_mask_patch,  # (pm, T=31) True=padded
            "scale_feature_patch": scale_feature_t,  # (10,)
            "geo_pos_embedding_patch": geo_pos_embedding_t,  # (sh_embed_dim,)
            "sh_embed_dim": self.sh_embed_dim_t,
            "harmonic_order": self.harmonic_order_t,
            "scale_f_dim": self.scale_f_dim,
            "coords": torch.tensor([m, i, j]),
            "lat_patch": lat_patch,  # (pH,)
            "lon_patch": lon_patch,  # (pW,)
        }
