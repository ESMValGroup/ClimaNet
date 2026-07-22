import xarray as xr
from cdo import Cdo
from datetime import datetime
from dateutil import relativedelta
from pathlib import Path
import cdsapi
import os


def download_era5(
    varname: str,
    year: int,
    month: int,
    days: [int],
    target_filename: str,
) -> None:
    # *** ERA5 data ***

    target_dir = os.path.dirname(target_filename)
    target_file = os.path.basename(target_filename)

    os.makedirs(target_dir, exist_ok=True)
    os.chdir(target_dir)

    print(f"Downloading ERA5 data for {varname} ({year}-{month:02d})...")
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [varname],
        "year": [year],
        "month": [month],
        "day": days,
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "data_format": "netcdf",
        #        "download_format": "unarchived",
    }

    client = cdsapi.Client()
    era5f = client.retrieve(dataset, request).download()
    cdo = Cdo()
    for retry in range(5):
        try:
            cdo.splitday(input=era5f, output=target_file)
            break
        except Exception as e:
            remaining_retries = 5 - retry
            print(
                f"Error splitting {era5f} into daily files: {e}. Trying {remaining_retries} more times..."
            )
    #    os.rename(era5f, target_file)
    os.remove(era5f)

    return None


# =============================================================================
# =============================================================================
# =============================================================================

era5_vars = {
    "sea_surface_temperature": "sst",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_cloud_cover": "tcc",
}

# set time range for checking / downloading
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 1, 1)

root_dir = "/work/bd0854/b380103/eso4clima/download/era5"

for varname in era5_vars:
    shortname = era5_vars[varname]
    loop_date = start_date

    download_days = []
    current_month = loop_date.month

    while loop_date <= end_date:
        yr = loop_date.year
        mo = loop_date.month
        dy = loop_date.day
        checkstr = f"Checking {shortname} {yr}-{mo:02d}-{dy:02d}"
        fnamebase = f"{root_dir}/{varname}/{yr}/era5_{varname}_{yr}{mo:02d}"
        fname = f"{fnamebase}{dy:02d}.nc"
        myfile = Path(fname)
        if myfile.is_file() and myfile.stat().st_size > 0:
            era5 = xr.open_dataset(fname)
            time = era5["valid_time"]
            years = time.dt.year
            months = time.dt.month
            days = time.dt.day
            if (years != yr).any() or (months != mo).any() or (days != dy).any():
                print(
                    f"{checkstr} - !date mismatch! ({str(years.values[0])}-{months.values[0]:02d}-{days.values[0]:02d})"
                )
                download_days.append(loop_date.day)
            else:
                print(f"{checkstr} - OK")
        else:
            print(f"{checkstr} - missing or file size zero")
            download_days.append(loop_date.day)

        loop_date += relativedelta.relativedelta(days=1)

        if loop_date.month != current_month or loop_date >= end_date:
            if len(download_days) > 0:
                download_era5(varname, yr, mo, download_days, fnamebase)
            current_month = loop_date.month
            download_days = []
