import xarray as xr
import xesmf as xe
import os
import fnmatch
import cdsapi
from datetime import datetime
from dateutil import relativedelta


def download_and_process(
    date: datetime.date,
    rootpath: str,
    sat_var: str,
    era5_var: str,
    out_var: str,
) -> None:
    year = f"{date.year}"
    month = f"{date.month:02d}"
    day = f"{date.day:02d}"

    # *** check HOAPS data (files have to be downloaded manually) ***

    #    hoaps_target_dir = os.path.join(rootpath, "HOAPS")
    #    hoapsfilename = f"hoaps-c.r30.h01.wvpa.{year}-{month}-{day}.nc"
    hoaps_target_dir = os.path.join(rootpath, "HOAPS_v5.0", f"{year}", f"{month}")
    hoapsfilename = f"HTWhc{year}{month}{day}000000313MIPOS01GL.nc"

    # check for file(s) with the correct date in the filename in 'hoaps_target_dir'
    if os.path.exists(hoaps_target_dir):
        hoaps_file = [
            n
            for n in fnmatch.filter(os.listdir(hoaps_target_dir), hoapsfilename)
            if os.path.isfile(os.path.join(hoaps_target_dir, n))
        ]

    if hoaps_file:
        print(f"Using HOAPS data found in {hoaps_target_dir}...")
    else:
        print(f"HOAPS data missing for {year}-{month}-{day}. Abort.")
        exit()

    # *** download ERA5 data ***

    era5_target_dir = os.path.join(rootpath, "era5", era5_var, year)
    era5filename = f"era5_{era5_var}_{year}{month}{day}.nc"

    os.makedirs(era5_target_dir, exist_ok=True)

    if os.path.isfile(os.path.join(era5_target_dir, era5filename)):
        print(f"Using ERA5 data found in {era5_target_dir}...")
    else:
        print(f"Downloading ERA5 data for {era5_var}...")
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": [era5_var],
            "year": [year],
            "month": [month],
            "day": [day],
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
            "download_format": "unarchived",
        }

        client = cdsapi.Client()
        era5f = client.retrieve(dataset, request).download()

        os.rename(era5f, os.path.join(era5_target_dir, era5filename))

    # Open the datasets in xarray

    # Open the HAOPS data
    infile = os.path.join(hoaps_target_dir, hoapsfilename)

    hoaps = xr.open_dataset(infile, decode_timedelta=True)

    #    # convert longitudes from -180...180 to 0...360
    #    hoaps['lon'] = hoaps['lon'].where(hoaps['lon'] >= 0, hoaps['lon'] + 360)

    # Drop unnecessary variables for efficiency
    variables_to_keep = [
        # Required uncertainty components
        sat_var,
        sat_var + "_err",
        #    sat_var + '_ran',
    ]

    all_variables = list(hoaps.data_vars)
    variables_to_drop = [var for var in all_variables if var not in variables_to_keep]
    hoaps = hoaps.drop_vars(variables_to_drop)

    era5 = xr.open_dataset(os.path.join(era5_target_dir, era5filename))
    # drop variables "expver" and "number" (if present)
    era5 = era5.drop_vars(["expver", "number"], errors="ignore")

    # rename coordinates and variable names

    hoaps = hoaps.rename(
        {
            sat_var: out_var,
            sat_var + "_err": out_var + "_uretrieval",
            #         sat_var + '_ran': out_var + '_urandom'
        }
    )

    era5_varname = "tcwv"
    era5 = era5.rename(
        {
            "valid_time": "time",
            era5_varname: out_var,
            "latitude": "lat",
            "longitude": "lon",
        }
    )

    # regrid ERA5 data (0.25x0.25) to the same grid as the HOAPS data (0.5x0.5)
    # "bilinear" is good enough and faster than "conservative"

    print("Regridding ERA5 data to HOAPS grid...")
    regridder = xe.Regridder(era5, hoaps, "bilinear", periodic=True)
    era5 = regridder(era5, keep_attrs=True)

    era5[out_var].attrs["units"] = "kg m-2"
    era5[out_var].attrs["long_name"] = "Water Vapor Path"
    era5[out_var].attrs["standard_name"] = "atmosphere_mass_content_of_water_vapor"

    #    # Calculate the total uncertainty of water vapor path in each HOAPS (0.5x0.5) cell
    #    # Procedure: caculate square root of random variance + retrieval uncertainty variance
    #
    #    hoaps[out_var + '_utotal'] = np.sqrt(hoaps[out_var + '_uretrieval']**2 + hoaps[out_var + '_urandom']**2)

    # set some metadata

    hoaps[out_var].attrs["units"] = "kg m-2"
    hoaps[out_var].attrs["long_name"] = "Water Vapor Path"
    hoaps[out_var].attrs["standard_name"] = "atmosphere_mass_content_of_water_vapor"
    hoaps[out_var + "_uretrieval"].attrs["units"] = "kg m-2"
    hoaps[out_var + "_uretrieval"].attrs["long_name"] = (
        "Water Vapor Path retrieval uncertainty"
    )
    #    hoaps[out_var + '_utotal'].attrs['units'] = 'kg m-2'
    #    hoaps[out_var + '_utotal'].attrs['long_name'] = 'Total Uncertainty of Water Vapor Path'
    #    hoaps[out_var + '_uretrieval'].attrs['units'] = 'kg m-2'
    #    hoaps[out_var + '_urandom'].attrs['units'] = 'kg m-2'
    hoaps["lat"].attrs.update(
        {"units": "degrees_north", "standard_name": "latitude", "long_name": "latitude"}
    )
    hoaps["lon"].attrs.update(
        {
            "units": "degrees_east",
            "standard_name": "longitude",
            "long_name": "longitude",
        }
    )
    hoaps["time"].attrs.update({"standard_name": "time", "long_name": "time"})

    print("------------------------------------------")
    print("saving results to netCDF")
    print("------------------------------------------")

    target_dir = f"./output/watervapor/{year}"
    os.makedirs(target_dir, exist_ok=True)

    # Write the output files

    hoaps_ready = hoaps.load()
    hoaps_ready.to_netcdf(
        f"{target_dir}/{year}{month}{day}_HOAPS_{out_var}.nc",
        unlimited_dims="time",
        encoding={
            "time": {"dtype": "float64", "_FillValue": None},
            "lon": {"dtype": "float32", "_FillValue": None},
            "lat": {"dtype": "float32", "_FillValue": None},
            out_var: {"dtype": "float32", "_FillValue": 1e20},
            #            out_var + '_utotal': {"dtype": "float32", "_FillValue": 1e20},
            #            out_var + '_urandom': {"dtype": "float32", "_FillValue": 1e20},
            out_var + "_uretrieval": {"dtype": "float32", "_FillValue": 1e20},
        },
    )

    era5_ready = era5.load()
    era5_ready.to_netcdf(
        f"{target_dir}/{year}{month}{day}_ERA5_full_{out_var}.nc",
        unlimited_dims="time",
        encoding={
            "time": {"dtype": "float64", "_FillValue": None},
            "lon": {"dtype": "float32", "_FillValue": None},
            "lat": {"dtype": "float32", "_FillValue": None},
            out_var: {"dtype": "float32", "_FillValue": 1e20},
        },
    )

    # apply HOAPS missing value mask to ERA5 data

    mask = hoaps[out_var] * 0.0
    era5masked = era5 + mask
    era5masked[out_var].attrs["units"] = "kg m-2"
    era5masked[out_var].attrs["long_name"] = "Water Vapor Path"
    era5masked[out_var].attrs["standard_name"] = (
        "atmosphere_mass_content_of_water_vapor"
    )

    era5masked_ready = era5masked.load()
    era5masked_ready.to_netcdf(
        f"{target_dir}/{year}{month}{day}_ERA5_masked_{out_var}.nc",
        unlimited_dims="time",
        encoding={
            "time": {"dtype": "float64", "_FillValue": None},
            "lon": {"dtype": "float32", "_FillValue": None},
            "lat": {"dtype": "float32", "_FillValue": None},
            out_var: {"dtype": "float32", "_FillValue": 1e20},
        },
    )

    return None


# ---------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# set variable to be processed
era5_var = "total_column_water_vapour"
sat_var = "wvpa"
out_var = "prw"

# set time range for downlaoding and processing

# start_date = datetime(2018, 1, 1)
start_date = datetime(2022, 2, 28)
end_date = datetime(2022, 12, 31)

# Set the root directory for downloads
root_dir = "./download/"

loop_date = start_date

while loop_date <= end_date:
    print(
        f"Downloading and processing {loop_date.year}"
        f"-{loop_date.month:02d}-{loop_date.day:02d}"
    )
    download_and_process(loop_date, root_dir, sat_var, era5_var, out_var)
    loop_date += relativedelta.relativedelta(days=1)
