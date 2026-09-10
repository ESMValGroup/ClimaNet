import numpy as np
import xarray as xr
from tbparse import SummaryReader

from climanet.utils import data_preparation, lsm_sst2wv, setup_logging


def test_setup_logging(tmp_path):
    writer = setup_logging(tmp_path)
    log_text = "This is a test log entry."
    writer.add_scalar("Test Scalar", 42)
    writer.add_text("Test Text", log_text)
    writer.close()

    # Test that there is one event file
    # The file should have "UTC" keyword in timestamp suffix
    assert len(list(tmp_path.glob("events*UTC*"))) == 1

    # Load the events file with SummaryReader
    reader = SummaryReader(tmp_path)
    assert reader.text["value"].iloc[0] == log_text  # check text
    assert reader.scalars["value"].iloc[0] == 42  # check scalar


def _make_datasets():
    # 4x4 dataset with, 6 days in one month
    lat = 4
    lon = 4
    daily_time = 6
    monthly_time = 1
    rng = np.arange(daily_time * lat * lon, dtype=np.float32).reshape(
        daily_time, lat, lon
    )
    # make some NaNs in the daily data
    rng[1, 1, 1] = np.nan
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

    return daily_da, monthly_da


def test_data_preparation():
    daily_da, monthly_da = _make_datasets()

    input_da, input_da_nan_mask, monthly_da, padded_days_mask, time_features = (
        data_preparation(
            daily_da, monthly_da, calculate_residuals=False, save_to_zarr=False
        )
    )
    assert input_da.shape == (1, 31, 4, 4)  # (M, T=31, H=4, W=4)
    assert isinstance(input_da, xr.DataArray)
    assert input_da_nan_mask[0, 1, 1, 1]  # check that the NaN mask is correctly set
    assert isinstance(input_da_nan_mask, xr.DataArray)
    assert monthly_da.shape == (1, 4, 4)  # (M, H=4, W=4)
    assert isinstance(monthly_da, xr.DataArray)
    assert padded_days_mask.shape == (1, 31)  # (M, T=31)
    assert isinstance(padded_days_mask, xr.DataArray)
    assert time_features.shape == (1, 31, 3)  # (M, T=31, 2) for month and day features
    assert isinstance(time_features, xr.DataArray)


def test_data_preparation_to_zarr(tmp_path):
    daily_da, monthly_da = _make_datasets()

    _ = data_preparation(
        daily_da,
        monthly_da,
        run_dir=tmp_path,
        calculate_residuals=False,
        save_to_zarr=True,
    )
    assert (tmp_path / "input_da.zarr").exists()
    assert (tmp_path / "input_da_nan_mask.zarr").exists()
    assert (tmp_path / "monthly_da.zarr").exists()
    assert (tmp_path / "padded_days_mask.zarr").exists()
    assert (tmp_path / "time_features.zarr").exists()


def test_data_preparation_from_zarr(tmp_path):
    daily_da, monthly_da = _make_datasets()

    _ = data_preparation(
        daily_da,
        monthly_da,
        run_dir=tmp_path,
        calculate_residuals=False,
        save_to_zarr=True,
    )

    # Now load from zarr
    input_da = xr.open_zarr(tmp_path / "input_da.zarr")
    input_da_nan_mask = xr.open_zarr(tmp_path / "input_da_nan_mask.zarr")
    monthly_da = xr.open_zarr(tmp_path / "monthly_da.zarr")
    padded_days_mask = xr.open_zarr(tmp_path / "padded_days_mask.zarr")
    time_features = xr.open_zarr(tmp_path / "time_features.zarr")

    assert input_da["tos"].shape == (1, 31, 4, 4)  # (M, T=31, H=4, W=4)
    assert isinstance(input_da, xr.Dataset)
    assert input_da_nan_mask["tos"][
        0, 1, 1, 1
    ]  # check that the NaN mask is correctly set
    assert isinstance(input_da_nan_mask, xr.Dataset)
    assert monthly_da["tos"].shape == (1, 4, 4)  # (M, H=4, W=4)
    assert isinstance(monthly_da, xr.Dataset)
    assert padded_days_mask["tos"].shape == (1, 31)  # (M, T=31)
    assert isinstance(padded_days_mask, xr.Dataset)
    assert time_features["tos"].shape == (
        1,
        31,
        3,
    )  # (M, T=31, 2) for month and day features
    assert isinstance(time_features, xr.Dataset)


def test_lsm_sst2wv():
    """Downsample a mock lsm with a factor of 2."""
    mask_values = np.zeros((1, 8, 4))  # Mock values for the land-sea mask
    mask_values[0, 0:2, 0:2] = 0.7  # Set 4 values to be above 0.5
    mask_values[0, 2:4, 2:4] = 0.2  # Set another 4 values to be below 0.5
    lsm_mask_025_mock = xr.DataArray(
        mask_values,
        dims=["time", "lat", "lon"],
        coords={
            "time": [0],
            "lat": np.linspace(-2.0, 2.0, 8),
            "lon": np.linspace(-1.0, 1.0, 4),
        },
        name="lsm",
    )
    lsm_mask_05 = lsm_sst2wv(lsm_mask_025_mock)

    assert np.all(
        lsm_mask_05.values[0, 0:1, 0:1]  # after downsampling 0:2 -> 0:1
    )  # all values above 0.5 should be True
    assert not np.any(
        lsm_mask_05.values[0, 1:2, 1:2]  # after downsampling 2:4 -> 1:2
    )  # all values below 0.5 should be False
    assert lsm_mask_05.shape == (1, 4, 2)  # By default downsample with a factor of 2
    assert lsm_mask_05.dtype == bool  # Mask should be of boolean type
