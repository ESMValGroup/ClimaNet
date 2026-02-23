import numpy as np
import pytest
import torch
import xarray as xr

from dataset import SSTDataset


def _make_datasets(lat=4, lon=4, daily_time=6, monthly_time=2):
    rng = np.arange(daily_time * lat * lon, dtype=np.float32).reshape(
        daily_time, lat, lon
    )
    daily_da = xr.DataArray(
        rng,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.datetime64("2000-01-01")
            + 10 * np.arange(daily_time).astype("timedelta64[D]"),
            "lat": np.arange(lat),
            "lon": np.arange(lon),
        },
    )

    monthly = np.arange(monthly_time * lat * lon, dtype=np.float32).reshape(
        monthly_time, lat, lon
    )
    monthly_da = xr.DataArray(
        monthly,
        dims=("time", "lat", "lon"),
        coords={
            "time": np.array(
                [np.datetime64("2000-01-16"), np.datetime64("2000-02-16")]
            ),
            "lat": np.arange(lat),
            "lon": np.arange(lon),
        },
    )

    mask = np.zeros((lat, lon), dtype=bool)
    mask[::2, ::2] = True
    land_mask = xr.DataArray(
        mask, dims=("lat", "lon"), coords={"lat": np.arange(lat), "lon": np.arange(lon)}
    )
    return daily_da, monthly_da, land_mask


def test_len_and_shapes():
    daily_da, monthly_da, land_mask = _make_datasets()
    dataset = SSTDataset(
        daily_da=daily_da,
        monthly_da=monthly_da,
        land_mask=land_mask,
        patch_size=(2, 2),
        overlap=0,
    )

    assert len(dataset) == 8

    sample = dataset[0]
    assert sample["coords"] == (0, 0, 0)
    assert sample["daily_patch"].shape == (
        1,
        4,
        2,
        2,
    )  # First month, 4 temporal steps, 2x2 spatial patch
    assert sample["monthly_patch"].shape == (2, 2)
    assert sample["daily_mask_patch"].shape == (1, 4, 2, 2)
    assert sample["daily_patch"].dtype == torch.float32
    assert sample["monthly_patch"].dtype == torch.float32
    assert sample["daily_mask_patch"].dtype == torch.bool


def test_index_bounds():
    daily_da, monthly_da, land_mask = _make_datasets()
    dataset = SSTDataset(
        daily_da=daily_da,
        monthly_da=monthly_da,
        land_mask=land_mask,
        patch_size=(2, 2),
        overlap=0,
    )

    with pytest.raises(IndexError):
        _ = dataset[-1]

    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_index_mapping_and_mask_values():
    daily_da, monthly_da, land_mask = _make_datasets()
    dataset = SSTDataset(
        daily_da=daily_da,
        monthly_da=monthly_da,
        land_mask=land_mask,
        patch_size=(2, 2),
        overlap=0,
    )

    sample = dataset[7]
    assert sample["coords"] == (1, 2, 2)

    expected_mask = land_mask.isel(lat=slice(2, 4), lon=slice(2, 4)).to_numpy()
    assert torch.equal(sample["land_mask_patch"], torch.from_numpy(expected_mask))
