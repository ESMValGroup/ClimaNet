import numpy as np
import pytest
import torch
import xarray as xr

from dataset import SSTDataset


def _make_datasets(lat=4, lon=4, daily_time=6, monthly_time=2):
    rng = np.arange(daily_time * lat * lon, dtype=np.float32).reshape(
        daily_time, lat, lon
    )
    daily_ds = xr.Dataset(
        {"ts": (("time", "lat", "lon"), rng)},
        coords={
            "time": np.arange(daily_time),
            "lat": np.arange(lat),
            "lon": np.arange(lon),
        },
    )

    monthly = np.arange(monthly_time * lat * lon, dtype=np.float32).reshape(
        monthly_time, lat, lon
    )
    monthly_ds = xr.Dataset(
        {"ts": (("time", "lat", "lon"), monthly)},
        coords={
            "time": np.arange(monthly_time),
            "lat": np.arange(lat),
            "lon": np.arange(lon),
        },
    )

    mask = np.zeros((lat, lon), dtype=bool)
    mask[::2, ::2] = True
    mask_da = xr.DataArray(
        mask, dims=("lat", "lon"), coords={"lat": np.arange(lat), "lon": np.arange(lon)}
    )
    return daily_ds, monthly_ds, mask_da


def test_len_and_shapes():
    daily_ds, monthly_ds, mask_da = _make_datasets()
    dataset = SSTDataset(
        daily_ds=daily_ds,
        monthly_ds=monthly_ds,
        mask_da=mask_da,
        patch_size=(2, 2),
        overlap=0,
    )

    assert len(dataset) == 8

    sample = dataset[0]
    assert sample["coords"] == (0, 0, 0)
    assert sample["daily"].shape == (1, 6, 2, 2)
    assert sample["monthly"].shape == (1, 2, 2)
    assert sample["mask"].shape == (2, 2)
    assert sample["daily"].dtype == torch.float32
    assert sample["monthly"].dtype == torch.float32
    assert sample["mask"].dtype == torch.bool


def test_index_bounds():
    daily_ds, monthly_ds, mask_da = _make_datasets()
    dataset = SSTDataset(
        daily_ds=daily_ds,
        monthly_ds=monthly_ds,
        mask_da=mask_da,
        patch_size=(2, 2),
        overlap=0,
    )

    with pytest.raises(IndexError):
        _ = dataset[-1]

    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_index_mapping_and_mask_values():
    daily_ds, monthly_ds, mask_da = _make_datasets()
    dataset = SSTDataset(
        daily_ds=daily_ds,
        monthly_ds=monthly_ds,
        mask_da=mask_da,
        patch_size=(2, 2),
        overlap=0,
    )

    sample = dataset[7]
    assert sample["coords"] == (1, 2, 2)

    expected_mask = mask_da.isel(lat=slice(2, 4), lon=slice(2, 4)).to_numpy()
    assert torch.equal(sample["mask"], torch.from_numpy(expected_mask))
