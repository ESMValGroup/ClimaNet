import argparse
from pathlib import Path

import xarray as xr
import numpy as np
from dask.distributed import Client, LocalCluster

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

    years = [2018, 2019, 2020, 2021, 2022]
    hourly_files = [
        f for f in data_folder.glob(f"*_hr_ERA5dc_masked_{var_name}.nc")
        if int(f.name[:4]) in years
    ]
    monthly_files = [
        f for f in data_folder.glob(f"*_mon_ERA5dc_full_{var_name}.nc")
        if int(f.name[:4]) in years
    ]

    # Set up Dask cluster
    cluster = LocalCluster(
        n_workers=32,
        threads_per_worker=1,
        memory_limit="7GB",
        processes=True,
    )

    client = Client(cluster)
    print(client)

    # read data: lazy
    input_data = xr.open_mfdataset(sorted(hourly_files))
    monthly_data = xr.open_mfdataset(sorted(monthly_files))

    # loop over year
    years = np.unique(monthly_data.time.dt.year)
    for year in years:
        input_data_year = input_data[var_name].sel(
            time=str(year)
        )
        monthly_data_year = monthly_data[var_name].sel(
            time=str(year)
        )
        # create subfolder for each year
        run_dir_year = run_dir / f"{year}"
        run_dir_year.mkdir(parents=True, exist_ok=True)

        # prepare data
        _ = data_preparation(
            input_data_year,
            monthly_data_year,
            run_dir=run_dir_year,
            calculate_residuals=True,
            is_hourly=True,
            save_to_zarr=True,
        )
        print(f"Data preparation for year {year} completed.")

    client.close()
