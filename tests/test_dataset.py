import numpy as np
import pytest
import torch
import xarray as xr

from climanet import STDataset
from climanet.utils import data_preparation


def _make_datasets():
    # 4x4 dataset with, 6 days in one month
    lat = 4
    lon = 4
    daily_time = 6
    monthly_time = 1
    rng = np.arange(daily_time * lat * lon, dtype=np.float32).reshape(
        daily_time, lat, lon
    )
    daily_da = xr.DataArray(
        rng,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.datetime64("2000-01-01")
            + np.arange(daily_time).astype("timedelta64[D]"),
            "lat": np.arange(lat),
            "lon": np.arange(lon),
        },
    )
    daily_da.name = "tos"

    monthly = np.arange(monthly_time * lat * lon, dtype=np.float32).reshape(
        monthly_time, lat, lon
    )
    monthly_da = xr.DataArray(
        monthly,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.datetime64("2000-01-16")
            + np.zeros(monthly_time, dtype="timedelta64[D]"),
            "lat": np.arange(lat),
            "lon": np.arange(lon),
        },
    )
    monthly_da.name = "tos"

    mask = np.zeros((lat, lon), dtype=bool)
    mask[::2, ::2] = True
    land_mask = xr.DataArray(
        mask, dims=("lat", "lon"), coords={"lat": np.arange(lat), "lon": np.arange(lon)}
    )
    land_mask.name = "lsm"
    return daily_da, monthly_da, land_mask


def test_len_and_shapes():
    daily_da, monthly_da, land_mask = _make_datasets()
    input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features = data_preparation(
        daily_da, monthly_da, calculate_residuals=False, save_to_zarr=False
    )
    dataset = STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        monthly_da=monthly_da,
        land_mask=land_mask,
        crop_size=(1, 2, 2),
        model_patch_size=(1, 1, 1),
    )

    assert len(dataset) == 4

    sample = dataset[0]
    assert torch.equal(sample["coords"], torch.tensor([0, 0, 0]))
    assert sample["input_data"].shape == (1, 1, 31, 2, 2)
    assert sample["monthly_data"].shape == (1, 2, 2)
    assert sample["input_data_mask"].shape == (1, 1, 31, 2, 2)
    assert sample["input_data_timef"].shape == (1, 31, 3)
    assert sample["input_data"].dtype == torch.float32
    assert sample["monthly_data"].dtype == torch.float32
    assert sample["input_data_mask"].dtype == torch.bool
    assert sample["input_data_timef"].dtype == torch.float32


def test_index_bounds():
    daily_da, monthly_da, land_mask = _make_datasets()
    input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features = data_preparation(
        daily_da, monthly_da, calculate_residuals=False, save_to_zarr=False
    )
    dataset = STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        monthly_da=monthly_da,
        land_mask=land_mask,
        crop_size=(1, 2, 2),
        model_patch_size=(1, 1, 1),
    )

    with pytest.raises(IndexError):
        _ = dataset[-1]

    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_index_mapping_and_mask_values():
    daily_da, monthly_da, land_mask = _make_datasets()
    input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features = data_preparation(
        daily_da, monthly_da, calculate_residuals=False, save_to_zarr=False
    )
    dataset = STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        monthly_da=monthly_da,
        land_mask=land_mask,
        crop_size=(1, 2, 2),
        model_patch_size=(1, 1, 1),
    )

    sample = dataset[3]
    assert torch.equal(sample["coords"], torch.tensor([0, 2, 2]))

    expected_mask = land_mask.isel(lat=slice(2, 4), lon=slice(2, 4)).to_numpy()
    assert torch.equal(sample["land_mask"], torch.from_numpy(expected_mask))


def test_time_feature_generation():
    daily_da, monthly_da, land_mask = _make_datasets()
    input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features = data_preparation(
        daily_da, monthly_da, calculate_residuals=False, save_to_zarr=False
    )
    dataset = STDataset(
        input_da=input_da,
        input_da_nan_mask=input_da_nan_mask,
        padded_days_mask=padded_days_mask,
        time_features=time_features,
        monthly_da=monthly_da,
        land_mask=land_mask,
        crop_size=(1, 2, 2),
        model_patch_size=(1, 1, 1),
    )

    sample = dataset[0]
    expected_time_feature = torch.tensor(
        [np.float32(0.), np.float32(2 * np.pi * 6 / 365.24), np.float32(0.0)]
    )
    assert torch.equal(sample["input_data_timef"][0, 5, :], expected_time_feature)
