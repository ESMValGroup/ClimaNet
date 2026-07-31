import argparse
from pathlib import Path
from dask.distributed import Client, LocalCluster
import xarray as xr

from climanet.utils import data_preparation, data_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage-path",
        type=str,
        default=Path("./tune_results").resolve(),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=Path("./data").resolve(),
    )
    args = parser.parse_args()

    data_folder = Path(args.data_dir).resolve()
    run_dir = Path(args.storage_path).resolve()
    var_name = "tos"

    hourly_files = data_split(
        data_folder,
        filename_pattern=f"*_hr_ERA5dc_masked_{var_name}.nc",
        train_range=(2018, 2020),
        validation_range=(2021, 2021),
        test_range=(2022, 2022),
    )
    monthly_files = data_split(
        data_folder,
        filename_pattern=f"*_mon_ERA5dc_full_{var_name}.nc",
        train_range=(2018, 2020),
        validation_range=(2021, 2021),
        test_range=(2022, 2022),
    )

    # Set up Dask cluster
    cluster = LocalCluster(
        n_workers=32,
        threads_per_worker=1,
        memory_limit="7GB",
        processes=True,
    )

    client = Client(cluster)
    print(client)

    # read data: only train with 3 years of data (2018-2020)
    input_data = xr.open_mfdataset(hourly_files["train"])
    monthly_data = xr.open_mfdataset(monthly_files["train"])

    # prepare data
    _ = data_preparation(
        input_data[var_name],
        monthly_data[var_name],
        run_dir=run_dir,
        calculate_residuals=True,
        is_hourly=True,
        save_to_zarr=True,
    )

    client.close()
