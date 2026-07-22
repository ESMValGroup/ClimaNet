import numpy as np
import xarray as xr
import xesmf as xe
import os
import subprocess
from typing import List
import fnmatch
import glob
import cdsapi
from datetime import datetime, timedelta
from dateutil import relativedelta
from scipy.signal.windows import gaussian
import pandas as pd

# A python function to control the download of all the files of all the
# available sensors in sensor_list for a stated day. Alternatively to
# download, processing could be done in place on Jasmin if an account
# and group workspace are available for the project.


def download_ceda_netcdf_files(
    sensor_list: List[str],
    target_year: str,
    target_month: str,
    target_day: str,
    target_root_dir: str,
) -> List[str]:
    """
    Downloads NetCDF files from the CEDA archive using the recursive wget
    mirror command, executing the download from the target directory itself.

    Args:
        sensor_list: A list of sensor strings (e.g., ['AVHRRMTA', 'AVHRRMTB']).
        target_year: The target year string (e.g., '2007').
        target_month: The target month string (e.g., '06').
        target_day: The target day string (e.g., '03').
        target_root_dir: The base directory where the downloads will be placed.

    Returns:
        A list of file paths for all successfully downloaded files.
    """

    # The base URL for the CEDA archive path up to the point of
    # sensor/year/month/day
    CEDA_BASE_URL = (
        "https://dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/"
        "sea_surface_temperature/CDR_v3/AVHRR/L3C/v3.0/"
    )

    # Store the original working directory for restoration later
    original_cwd = os.getcwd()
    successful_downloads: List[str] = []

    # 1. Define and create the local target directory
    date_folder = f"{target_year}-{target_month}-{target_day}"
    local_target_dir = os.path.join(target_root_dir, date_folder)

    try:
        os.makedirs(local_target_dir, exist_ok=True)
        print(f"Created/verified local directory: {local_target_dir}")

        # 2. Change the current working directory (CWD) to the target directory
        os.chdir(local_target_dir)
        print(f"Changed CWD to: {os.getcwd()}")

        print(sensor_list)

        # 3. Iterate through each sensor and attempt download
        for sensor in sensor_list:
            # Construct the remote URL path to the directory (note the
            # trailing slash). This is the directory that wget will mirror
            # content from
            remote_dir_url = (
                f"{CEDA_BASE_URL}{sensor}/{target_year}/{target_month}/{target_day}/"
            )

            print(f"\nAttempting recursive download for sensor '{sensor}'...")

            # Construct the wget command:
            # --mirror -r: Recursively download everything below the URL
            # --no-parent: Do not ascend to the parent directory on the server
            # -e robots=off: Ignores robots.txt
            # Since we are already in the target directory, no -O is needed.
            wget_command = [
                "wget",
                "-e",
                "robots=off",
                "--mirror",
                "--no-parent",
                "-r",
                "--cut-dirs",
                "13",  # this is to avoid replicating the whole directory structure
                remote_dir_url,
            ]

            print(" ".join(wget_command))

            # Execute the command
            try:
                result = subprocess.run(
                    wget_command,
                    capture_output=True,
                    text=True,
                    check=False,  # Do not raise an exception for non-zero exit codes
                )

                # Check for major failures (e.g., 404 on the directory itself)
                if result.returncode != 0:
                    print(
                        f"Warning: Wget returned non-zero exit code "
                        f"({result.returncode}) for {sensor}."
                    )
                    print(
                        "   This might indicate the remote directory was "
                        "empty or inaccessible."
                    )
                else:
                    print(f"Wget command executed for {sensor}.")
                    # Note: We rely on the final os.listdir() to check for actual files.

            except FileNotFoundError:
                print(
                    "Error: 'wget' command not found. Please ensure "
                    "'wget' is installed and in your system PATH."
                )
                return []  # Stop processing if wget isn't available
            except Exception as e:
                print(f"An unexpected error occurred during subprocess call: {e}")
                return []  # Stop processing

        # 4. Final step: Check the contents of the current working directory
        # to confirm which files were successfully created.

        # Move and clean up files
        for file_path in glob.glob("./dap.ceda.ac.uk/*.nc"):
            os.rename(file_path, os.path.basename(file_path))
        [os.remove(f) for f in glob.glob("./dap.ceda.ac.uk/*.html")]
        os.rmdir("./dap.ceda.ac.uk")

        # Filter for files ending in .nc and construct their absolute paths
        downloaded_filenames = [
            f for f in os.listdir(".") if f.endswith(".nc") and os.path.isfile(f)
        ]

        # Convert relative filenames to absolute paths based on the root directory
        successful_downloads = [
            os.path.join(local_target_dir, filename)
            for filename in downloaded_filenames
        ]

    except OSError as e:
        print(f"Fatal error during directory setup or CWD change: {e}")
        return []

    finally:
        # 5. RESTORE the original working directory
        if os.getcwd() != original_cwd:
            os.chdir(original_cwd)
            print(f"\nRestored CWD to: {os.getcwd()}")

    return successful_downloads


def create_avhrr_sensor_list() -> List[str]:
    sensor_list = [
        f"AVHRR{i:02d}_G"  # Use f-string formatting with :02d for zero-padding
        for i in range(6, 20)
        if i != 13
    ]
    return sensor_list


def generate_hourly_coverage_mask(
    l3c: xr.Dataset, time_var: str, sat_var: str
) -> xr.DataArray:
    """
    Creates a boolean mask indicating if any sensor had (any) valid SST data
    within each hourly UTC bin (T-0.5 to T+0.5) for a given lat/lon cell
    using a robust time difference method.
    """
    # --- Step A: Calculate Absolute Observation Time and Validity Mask ---
    base_time_expanded = l3c["time"].broadcast_like(l3c[time_var])
    # Compute the core variables into memory to avoid Dask/chunking errors
    absolute_time = (base_time_expanded + l3c[time_var]).compute()
    valid_data = l3c[sat_var].notnull().compute()

    # --- Step B: Define Target Hours (T in TZ) ---

    start_date = l3c["time"].dt.floor("D").item()
    # 24 target timestamps (00:00:00Z to 23:00:00Z)
    target_times = pd.date_range(start_date, periods=24, freq="h", name="target_hour")

    # --- Step C: Vectorised Time Binning (The Core Logic) ---

    # 1. Expand the target_times to be broadcastable against the absolute_time
    #    array. The resulting target_times_expanded has a new dimension
    #    'target_hour'.
    target_times_expanded = xr.DataArray(
        target_times, coords={"target_hour": target_times}
    ).broadcast_like(valid_data)

    # 2. Calculate the time difference between EVERY observation and EVERY
    #    target hour. This array is massive:
    #    (sensor_index, lat, lon, time, target_hour)
    time_difference = absolute_time - target_times_expanded
    absolute_time.close()
    target_times_expanded.close()

    # 3. Create a boolean mask: check if the difference falls within the
    #    half-hour window.
    #    Interval is [T-0.5h, T+0.5h). The 'time_difference' must be:
    #    >= -30 minutes (inclusive start)
    #    <  +30 minutes (exclusive end)
    is_in_bin = (time_difference >= -pd.Timedelta(minutes=30)) & (
        time_difference < pd.Timedelta(minutes=30)
    )
    time_difference.close()

    # 4. Combine the time bin mask with the data validity mask.
    #    We only care about points that are both IN the time bin AND are
    #    valid SST data.
    valid_obs_in_bin = is_in_bin & valid_data

    # --- Step D: Reduction and Final Formatting ---

    # 1. Collapse all the observation-specific dimensions (sensor_index, time)
    #    using .any(). This answers the question: "For this
    #    (lat, lon, target_hour) cell, was there ANY valid observation across
    #    all sensors?"
    hourly_coverage = valid_obs_in_bin.any(dim=["sensor_index", "time"])

    # 2. The resulting array has dimensions (lat, lon, target_hour).
    #    Rename and format.
    coverage_mask = hourly_coverage.rename({"target_hour": "time"}).astype(bool)

    return coverage_mask


def regrid_to_boundary_centered_grid(da: xr.DataArray, roll=False) -> xr.DataArray:
    """
    Interpolates a DataArray from its current center-based grid onto a new
    grid whose coordinates are derived from user-specified boundaries.

    Includes robust handling for 0-360 vs -180-180 longitude domains.

    Assumes dimensions are named 'lat' and 'lon'.
    """
    print("Starting regridding process...")

    # --- 0. Longitude Domain Check and Correction ---

    input_lon = da["longitude"]

    # Check if roll for 0-360 to -180-180 is requested
    if roll:
        print("Applying cyclic roll to -180 to 180...")

        # Calculate the index closest to 180 degrees
        lon_diff = np.abs(input_lon - 180.0)
        # We need to roll such that the 180-degree line is moved to the edge
        # and the new array starts near -180
        roll_amount = (
            int(lon_diff.argmin().item() + (input_lon.size / 2)) % input_lon.size
        )

        # Roll the DataArray and its coordinates
        da = da.roll(longitude=roll_amount, roll_coords=True)

        # Correct the longitude coordinate values: shift values > 180 down by 360
        new_lon_coords = da["longitude"].where(
            da["longitude"] <= 180, da["longitude"] - 360
        )

        # Assign the corrected and sorted coordinates
        da = da.assign_coords(longitude=new_lon_coords).sortby("longitude")
        print(
            f"Longitudes adjusted. New range: {da['longitude'].min().item():.2f} "
            f"to {da['longitude'].max().item():.2f}"
        )

    # --- 1. Define Target Grid Boundaries ---

    # Target latitude boundaries: -90.0 up to 90.0 in 0.25 degree steps
    # (721 points)
    lat_bnds = np.linspace(-90.0, 90.0, 721)

    # Target longitude boundaries: -180.0 up to 180.0 in 0.25 degree steps
    # (1441 points)
    lon_bnds = np.linspace(-180.0, 180.0, 1441)

    # --- 2. Calculate New Grid Centers (Coordinates) ---

    # New latitude centers are the average of consecutive boundaries
    # (720 points)
    new_lats = (lat_bnds[:-1] + lat_bnds[1:]) / 2.0

    # New longitude centers are the average of consecutive boundaries
    # (1440 points)
    new_lons = (lon_bnds[:-1] + lon_bnds[1:]) / 2.0

    # --- 3. Interpolate the Data ---

    # Use linear interpolation (suitable for gappy data) to map data onto the
    # new centers. xarray handles the NaNs automatically.
    da_regridded = da.interp(latitude=new_lats, longitude=new_lons, method="linear")

    print(f"Regridding complete. New dimensions: {da_regridded.dims}")
    return da_regridded


def get_date_range_triplet(yyyy, mm, dd):
    # Because the diurnal model is a local-time model, we will need the day before
    # and after the central target day.
    target_date = datetime(year=yyyy, month=mm, day=dd)

    dates = [
        target_date - timedelta(days=1),
        target_date,
        target_date + timedelta(days=1),
    ]

    integer_sets = [[d.year, d.month, d.day] for d in dates]
    string_dates = [d.strftime("%Y-%m-%d") for d in dates]

    return integer_sets, string_dates


def get_weighted_rolling_subset(
    ds, window_size=23, start_idx=24, length=24, dim="time"
):
    """
    Computes a Gaussian-weighted rolling mean and extracts a specific time slice.
    The window_size is treated as the 4-sigma extent.
    """
    # Calculate sigma such that window_size = 4 * sigma
    sigma = window_size / 4.0

    # Create the Gaussian weights
    # scipy's gaussian function returns a symmetric window
    weights_np = gaussian(window_size, std=sigma)

    # Convert weights to a DataArray for Xarray compatibility
    weights = xr.DataArray(weights_np, dims=["window"])

    # Apply the weighted rolling operation
    # we use construct() to create a virtual dimension of windows for weighting
    ds_rolling = (
        ds.rolling({dim: window_size}, center=True)
        .construct("window")
        .weighted(weights)
        .mean("window")
    )

    # Slice the results to the requested range
    ds_subset = ds_rolling.isel({dim: slice(start_idx, start_idx + length)})

    return ds_subset


def close_diurnal_cycles(da):
    """
    Closes the diurnal cycle by adding a linear offset so that
    the first and last points of the 'tod' dimension are zero.

    Parameters:
        da (xr.DataArray): The input data with a 'tod' dimension.

    Returns:
        xr.DataArray: The corrected DataArray.
    """
    # Identify the first and last indices along the 'tod' dimension
    first_val = da.isel(tod=0)
    last_val = da.isel(tod=-1)

    # Create a normalised coordinate for the linear interpolation (0 to 1)
    # This ensures the correction is applied proportionally across the 24 points
    n_tod = len(da.tod)
    line_coords = xr.DataArray(
        np.linspace(0, 1, n_tod), coords={"tod": da.tod}, dims="tod"
    )

    # Define the linear trend between the first and last points
    # formula: y = first + (last - first) * t
    linear_trend = first_val + (last_val - first_val) * line_coords

    # Subtract the trend to force the first and last points to zero
    da_corrected = da - linear_trend

    # Preserve original attributes
    da_corrected.attrs = da.attrs

    return da_corrected


def add_local_solar_time(ds):
    """
    Adds a 'lst' variable to the dataset representing local solar time
    as a timedelta64[ns] (time since most recent local midnight).
    """
    # 1. Calculate UTC time since midnight for each time step
    # Extract the hour, minute, second from the time coordinate
    utc_time = ds.time.dt
    seconds_since_utc_midnight = (
        utc_time.hour * 3600 + utc_time.minute * 60 + utc_time.second
    )

    # 2. Calculate the longitude offset in seconds
    # 360 degrees = 86400 seconds (24 hours)
    # offset = longitude * (86400 / 360) = longitude * 240
    lon_offset_seconds = ds.longitude * 240

    # 3. Combine to get total seconds since local midnight
    # We use broadcasting: (time, longitude)
    lst_seconds = seconds_since_utc_midnight + lon_offset_seconds

    # 4. Wrap the seconds within a 24-hour window (86400 seconds)
    # This ensures it is always "time since the most recent midnight"
    lst_seconds_wrapped = lst_seconds % 86400

    # 5. Convert to timedelta64[ns]
    ds["lst"] = (lst_seconds_wrapped * 1e9).astype("timedelta64[ns]")

    # Clean up attributes
    ds.lst.attrs["long_name"] = "local solar time"
    ds.lst.attrs["description"] = (
        "Time since most recent local midnight based on longitude"
    )

    return ds


def apply_diurnal_fit(ds, dvo, start_idx=24, length=24, ntstep=48):
    """
    Pre-interpolates Morak bins in U, TCC, latitude and Time.
    """

    # 1. Subset target and identify season
    ds_sub = ds.isel(time=slice(start_idx, start_idx + length))
    central_month = ds_sub.time.dt.month.values[0]
    s_idx = (
        0
        if central_month in [12, 1, 2]
        else 1
        if central_month in [3, 4, 5]
        else 2
        if central_month in [6, 7, 8]
        else 3
    )

    # 2. Build High-Resolution 4D Lookup Table (U, TCC, Lat, ToD)
    mindex = pd.MultiIndex.from_product(
        [[0.1, 0.5, 0.9], [1.5, 5.0, 9.5, 13.0]], names=["tcc", "u"]
    )
    mcoords = xr.Coordinates.from_pandas_multiindex(mindex, "category")
    lut = dvo.isel(season=s_idx).assign_coords(mcoords).unstack("category")

    # Before filling gaps, set the latitude extremes to zero because otherwise they go wild
    lut[0:2, :, :, :] = 0.0
    lut[-1, :, :, :] = 0.0

    # Fill gaps in DVO
    lut = (
        lut.interpolate_na(dim="latitude", method="nearest", fill_value="extrapolate")
        .interpolate_na(dim="u", method="nearest", fill_value="extrapolate")
        .interpolate_na(dim="tcc", method="nearest", fill_value="extrapolate")
    )

    # Linear extrapolation on the SMALL lookup table.
    # This calculates the slope (e.g., between 0.5 and 0.9) and projects it to 0.0 and 1.0.
    lut = lut.interp(
        u=[0.0, 1.5, 5.0, 9.5, 13.0, 13.5],
        tcc=[0.0, 0.1, 0.5, 0.9, 1.0],
        latitude=[lat for lat in range(90, -91, -5)],
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    # PRE-INTERPOLATE everything to a dense grid
    u_dense = np.arange(0.0, 13.5, 0.5)
    tcc_dense = np.arange(0.0, 1.05, 0.1)

    # Time grid: 'ntstep' steps (default=48, i.e. every 30 mins)
    tod_step = 24.0 / ntstep
    tod_dense = pd.to_timedelta(np.arange(0, 23.9, tod_step), unit="h")

    # Generate the dense 4D LUT spatially and temporally
    lut_dense = lut.interp(
        u=u_dense,
        tcc=tcc_dense,
        latitude=ds_sub.latitude,
        tod=tod_dense,
        method="linear",
        kwargs={"bounds_error": False, "fill_value": None},
    ).compute()

    3.0  # Calculate and CLIP indices to prevent NaNs at the exact boundaries (like tcc=1.0)
    max_u_idx = len(u_dense) - 1
    max_tcc_idx = len(tcc_dense) - 1
    max_tod_idx = len(tod_dense) - 1

    u_idx = ((ds_sub.u.clip(0.5, 13.0) - 0.5) / 0.5).round().astype(int).compute()
    u_idx = u_idx.clip(0, max_u_idx)

    tcc_idx = (ds_sub.tcc.clip(0.0, 1.0) / 0.1).round().astype(int).compute()
    tcc_idx = tcc_idx.clip(0, max_tcc_idx)

    # ToD index: Convert ns to hours, then multiply by 2 for the 0.5h step
    tod_hours = ds_sub.lst.dt.total_seconds() / 3600.0
    tod_idx = (tod_hours * ntstep / 24).round().astype(int)
    tod_idx = tod_idx % ntstep
    tod_idx = tod_idx.clip(0, max_tod_idx).compute()

    # 4. Final Vectorised Extraction (No interpolation happens here -- would be too slow!)
    dvfit = lut_dense.isel(u=u_idx, tcc=tcc_idx, tod=tod_idx)
    dvfit = dvfit.fillna(0)

    # 5. Clean and Dimension Alignment
    # The result of the isel above already has dimensions (time, latitude, longitude)
    # because it inherited them from the indexers
    dvfit = dvfit.transpose("time", "latitude", "longitude")
    ds_sub["dvfit"] = dvfit.drop_vars(["u", "tcc", "tod", "category"], errors="ignore")

    return ds_sub.compute()


def download_and_process(
    date: datetime.date,
    rootpath: str,
    aux_rootpath: str,
    sat_var: str,
    era5_vars: dict,
    cmor_var: str,
) -> None:
    # The sensor list available for L3C files is:
    # - under 'AVHRR' via path
    #   /dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/sea_surface_temperature/CDR_v3/:
    #   AVHRR06_G, AVHRR07_G, ... AVHRR12_G, AVHRR14_G, ... AVHRR19_G, AVHRRMTA, AVHRRMTB
    # - under 'SLSTR' via path
    #   /dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/sea_surface_temperature/CDR_v3/:
    #   SLSTRA, SLSTRB
    # - under ATSR via path /dap.ceda.ac.uk/neodc/esacci/sst/data/CDR_v2/ATSR/L3C/v2.1/:
    #   ATSR1, ATSR2, AATSR
    #
    # (For the ATSRs, there was no need to reprocess for version 3, so v2.1 is
    # the up-to-date version of their L3C.)
    # For this project, don't use the AMSR data.
    # The data access is illustrated just for AVHRRs.

    year = f"{date.year}"
    month = f"{date.month:02d}"
    day = f"{date.day:02d}"

    download_root = os.path.join(root_dir, "download")

    # *** CCI data ***

    cci_root_dir = os.path.join(download_root, "cci")
    cci_target_dir = os.path.join(cci_root_dir, f"{year}-{month}-{day}")
    # check for .nc files with the correct date in the filename in 'target_dir'
    pattern = f"{year}{month}{day}*.nc"
    if os.path.exists(cci_target_dir):
        l3c_files = [
            n
            for n in fnmatch.filter(os.listdir(cci_target_dir), pattern)
            if os.path.isfile(os.path.join(cci_target_dir, n))
        ]
    else:
        l3c_files = []

    if l3c_files:
        print(f"Using L3C data found in {cci_target_dir}...")
    else:
        print(f"Downloading data for {sat_var}...")
        all_avhrrs = create_avhrr_sensor_list()
        all_avhrrs.append("AVHRRMTA")
        all_avhrrs.append("AVHRRMTB")

        # Running the function to get all the L3C files for one particular day
        # 1. Define the parameters
        sensors_to_download = all_avhrrs

        # 2. Run the function
        l3c_files = download_ceda_netcdf_files(
            sensor_list=sensors_to_download,
            target_year=year,
            target_month=month,
            target_day=day,
            target_root_dir=cci_root_dir,
        )

        # 3. Print the final results
        print("\n--- Download Summary ---")
        if l3c_files:
            print(f"Successfully downloaded {len(l3c_files)} file(s):")
            for f in l3c_files:
                print(f"- {f}")
        else:
            print("No files were successfully downloaded.")

    # *** ERA5 data ***

    date_ints, date_strings = get_date_range_triplet(int(year), int(month), int(day))

    era5_root_dir = os.path.join(download_root, "era5")

    # list of all era5 files needed for processing this day
    era5_file_list = []

    for era5_var in era5_vars:
        era5_target_dir = os.path.join(era5_root_dir, era5_var)
        os.makedirs(era5_target_dir, exist_ok=True)

        for date_string in date_strings:
            yyyy = date_string[0:4]
            mm = date_string[5:7]
            dd = date_string[8:10]

            era5_target_dir_yyyy = os.path.join(era5_target_dir, yyyy)
            os.makedirs(era5_target_dir_yyyy, exist_ok=True)
            os.chdir(era5_target_dir_yyyy)

            era5filename = f"era5_{era5_var}_{yyyy}{mm}{dd}.nc"
            era5_file_list.append(os.path.join(era5_target_dir_yyyy, era5filename))

            if os.path.isfile(era5filename):
                print(f"Using ERA5 data found for {era5_var} ({date_string})...")
            else:
                print(f"Downloading ERA5 data for {era5_var} ({date_string})...")
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
                os.rename(era5f, era5filename)

    # Open the datasets in xarray

    # Open the AVHRR data
    file_pattern = os.path.join(cci_root_dir, f"{year}-{month}-{day}" + "/*.nc")

    l3c = xr.open_mfdataset(
        file_pattern,
        concat_dim="sensor_index",  # This names the new dimension
        combine="nested",
        decode_timedelta=True,  # Use nested combine for lists/patterns
    )

    # Drop unnecessary variables for efficiency
    variables_to_keep = [
        # Required uncertainty components
        "uncertainty_correlated",
        "uncertainty_random",
        "uncertainty_systematic",
        # SST data
        sat_var,
        "sst_dtime",
    ]

    all_variables = list(l3c.data_vars)
    variables_to_drop = [var for var in all_variables if var not in variables_to_keep]
    l3c = l3c.drop_vars(variables_to_drop)

    # Open ERA5 data
    nwp = xr.open_mfdataset(
        era5_file_list, compat="no_conflicts", drop_variables=["number", "expver"]
    )
    nwp = nwp.rename({"valid_time": "time"})

    # nwp = nwp.drop_vars("season")

    # calculate 10-m wind speed from u- and v-components
    nwp["u"] = np.sqrt(nwp.u10**2 + nwp.v10**2)
    nwp = nwp.drop_vars(["u10", "v10"])

    nwp2 = get_weighted_rolling_subset(nwp, window_size=25, start_idx=0, length=72)
    nwp.close()

    # Assume the interpolated sst (Depth Avg) variable gives a good
    # "foundation temperature" on which to add diurnal variability
    # Read in the diurnal sst anomaly parameterisation
    # Source of sst-anomalies file: https://doi.org/10.6084/m9.figshare.2069049

    dv = xr.open_dataset(
        os.path.join(
            aux_rootpath,
            "sst-anomalies",
            "diurnal_sst_anomalies_from_drifting_buoys_v1.1.nc",
        ),
        decode_timedelta=True,
    ).sstano_fit

    dvc = close_diurnal_cycles(dv)

    # to use the parameterisation with the DV model, will need the local solar time
    nwp3 = add_local_solar_time(nwp2)
    nwp2.close()
    nwp3["lst"].attrs.pop("units", None)

    # Apply diurnal cycle correction from Morak-Bozzo. S., C.J. Merchant,
    # E.C. Kent, D.I. Berry, and G. Carella, Climatological diurnal variability
    # in sea surface temperature characterized from drifting buoy data,
    # Geosci. Data J. 3: 20–28 (2016), doi: 10.1002/gdj3.35
    nwp4 = apply_diurnal_fit(nwp3, dvc, ntstep=1440)
    nwp3.close()

    # drop variables "expver" and "number" (if present)
    # nwp4 = nwp4.drop_vars(["expver", "number"], errors="ignore")

    # First, create a Boolean for the presence of data at the full satellite
    # resolution. Probably far from optimised!

    if sat_var == "sea_surface_temperature":
        time_var = "sst_dtime"
    elif sat_var == "sea_surface_temperature_depth":
        time_var = "sst_depth_dtime"
    else:
        print(f"Error: unknown satellite variable ({sat_var})")
        exit()

    # Before creating hourly coverage mask, eliminate observations in cells
    # that are in 0.25 cells that are partly land

    dst = xr.open_dataset(
        os.path.join(aux_rootpath, "change-climatology", "delta_sst_0.25.nc")
    )
    # 1. Detect where the delta-sst variable is NaN
    #    We use any() across the 'month' dimension to find cells that are NaN at least once.
    nan_mask = dst["delta_sst"].isnull().any(dim="month")

    # 2. Use that to make a variable land25 (1 where delta-sst is nan, 0 otherwise)
    land25 = xr.where(nan_mask, 1, 0).rename("land25")
    nan_mask.close()

    print("\n--- land25 (Coarse Mask) Created ---")

    UPFACTOR = 5
    coarse_mask_np = land25.values.astype(np.int8)
    # np.repeat() is used twice, once along each axis (lat and lon).
    # np.repeat(..., repeats=UPFACTOR, axis=0) repeats each row 5 times.
    # np.repeat(..., repeats=UPFACTOR, axis=1) repeats each column 5 times.
    land05_np = np.repeat(np.repeat(coarse_mask_np, UPFACTOR, axis=0), UPFACTOR, axis=1)
    coarse_mask_np = None

    # create blocky mask (no land within 0.25)
    land05 = xr.DataArray(
        land05_np,
        coords={"lat": l3c.lat, "lon": l3c.lon},
        dims=["lat", "lon"],
        name="land05",
    )

    # apply this to the l3c before further processing
    for varn in variables_to_keep[0:4]:
        l3cm = xr.where(land05 == 1, np.nan, l3c[varn]).transpose(*l3c[varn].dims)
        l3c[varn] = l3cm

    l3cm = None

    mask = generate_hourly_coverage_mask(l3c, time_var, sat_var)

    # ----------------------------------------------------------------------------
    # old regridding method using xarray
    # sstdv = regrid_to_boundary_centered_grid((nwp4.sst+nwp4.dvfit), roll = True)
    # sstdv.name = 'sst'
    # sstndv = regrid_to_boundary_centered_grid((nwp4.sst), roll = True)
    # ----------------------------------------------------------------------------

    # Next, coarsen the observations and mask to the same grid.
    # Here, I assume using a simple average if any data are present.
    # This means the mask should be True if any of the finer resolutions
    # is True.

    mask25 = (
        mask.coarsen(lat=5, lon=5, boundary="exact")
        .max()
        .transpose("time", "lat", "lon")
    )

    # regrid ERA5 data to coarsened CCI grid using xesmf
    # "bilinear" is good enough and faster than "conservative"
    print("Regridding ERA5 data to 0.25 x 0.25 CCI grid...")
    regridder = xe.Regridder(nwp4.sst, mask25, "bilinear", periodic=True)
    sstdv = regridder((nwp4.sst + nwp4.dvfit), keep_attrs=True)
    sstdv.name = "sst"
    #    sstndv = regridder(nwp4.sst, keep_attrs=True)

    # Check the regridded lats and lons are identical to CCI.
    # Results should be close to zero to start with.
    print("Lon check", (sstdv.lon - mask25.lon).mean().values)
    print("Lat check", (sstdv.lat - mask25.lat).mean().values)

    # Now make them identical to avoid masking issues
    mask25["lon"] = sstdv.lon.values
    mask25["lat"] = sstdv.lat.values

    # Next make the raw average of sst and the uncertainty

    # Simple sum of all available values across all sensors
    l3c25 = (
        l3c[sat_var]
        .sum(dim="sensor_index")
        .coarsen(lat=5, lon=5, boundary="exact")
        .sum()
        .to_dataset(name=sat_var)
    )

    # Establish the corresponding count of values
    count = (
        l3c[sat_var]
        .notnull()
        .sum(dim="sensor_index")
        .coarsen(lat=5, lon=5, boundary="exact")
        .sum()
    )

    # Calculate the simple mean
    l3c25[sat_var] /= count

    # Calculate the uncertainty of SST in each populated 0.25 cell
    # Procedure:
    # Caculate random variance
    # calculate correlated variance
    # add and square root
    # This is treating each 0.05 cell value as an estimate of the 0.25 cell
    # mean, and therefore neglects some representation uncertainty, but
    # as the variability in this distance is usually small, this is acceptable

    # Not including the systematic term which is uncertainty in
    # an overall bias for all cells and not discriminative between cells

    # Calculate random uncertainty component
    sumsqs = (
        (l3c["uncertainty_random"] ** 2)
        .sum(dim="sensor_index")
        .coarsen(lat=5, lon=5, boundary="exact")
        .sum()
    )

    variance_random = sumsqs / (count**2)

    # Calculate mean of locally correlated uncertainty assuming perfect
    # correlation on this short scale
    uncertainty_correlated = (
        l3c["uncertainty_correlated"]
        .sum(dim="sensor_index")
        .coarsen(lat=5, lon=5, boundary="exact")
        .sum()
        / count
    )

    l3c25["uncertainty"] = np.sqrt(variance_random + uncertainty_correlated**2)

    l3c25["count"] = count

    # set some metadata
    l3c25[sat_var].attrs["units"] = "K"
    l3c25["uncertainty"].attrs["units"] = "K"
    l3c25["count"].attrs["units"] = "1"
    l3c25["lat"].attrs.update(
        {"units": "degrees_north", "standard_name": "latitude", "long_name": "latitude"}
    )
    l3c25["lon"].attrs.update(
        {
            "units": "degrees_east",
            "standard_name": "longitude",
            "long_name": "longitude",
        }
    )

    # rename variable names in ERA5 datasets
    sstdv = sstdv.rename(cmor_var)
    #    sstndv = sstndv.rename(cmor_var)
    # ----------------------------------------------------------------------
    # rename coordinate names in ERA5 datasets (when regridding with xarray)
    # sstdv = sstdv.rename(
    #         {'latitude': 'lat',
    #         'longitude': 'lon'}
    # )
    # sstndv = sstndv.rename(
    #          {'latitude': 'lat',
    #          'longitude': 'lon'}
    # )
    # ----------------------------------------------------------------------

    print("------------------------------------------")
    print("saving results to netCDF")
    print("------------------------------------------")

    # Write the output files

    output_root = os.path.join(root_dir, "output", "sst", str(year))
    os.makedirs(output_root, exist_ok=True)

    l3c25ready = l3c25.load()
    l3c25ready.to_netcdf(
        os.path.join(output_root, f"{year}{month}{day}_L3C_{sat_var}.nc"),
        unlimited_dims="time",
        encoding={
            "time": {"dtype": "float64", "_FillValue": None},
            "lon": {"dtype": "float32", "_FillValue": None},
            "lat": {"dtype": "float32", "_FillValue": None},
            sat_var: {"dtype": "float32", "_FillValue": 1e20},
            "uncertainty": {"dtype": "float32", "_FillValue": 1e20},
            "count": {"dtype": "float32", "_FillValue": 1e20},
        },
    )

    # save L3C data seperated into the 24 time steps used for ERA5 data
    time = mask25["time"]
    l3c2524h = None

    # create a new dataset that contains 24 copies of the l3c25 fields
    # (1 per hourly timestep)
    for timestep in time:
        newstep = l3c25.assign_coords(time=[timestep.values])
        if l3c2524h is None:
            l3c2524h = newstep
        else:
            l3c2524h = xr.concat([l3c2524h, newstep], dim="time")

    # Apply hourly mask
    l3c_result = l3c2524h.where(mask25)

    # Save hourly L3C data to netCDF
    l3c_result_ready = l3c_result.load()
    l3c_result_ready.to_netcdf(
        os.path.join(output_root, f"{year}{month}{day}_L3C24_{sat_var}.nc"),
        unlimited_dims="time",
        encoding={
            "time": {"dtype": "float64", "_FillValue": None},
            "lon": {"dtype": "float32", "_FillValue": None},
            "lat": {"dtype": "float32", "_FillValue": None},
            sat_var: {"dtype": "float32", "_FillValue": 1e20},
            "uncertainty": {"dtype": "float32", "_FillValue": 1e20},
            "count": {"dtype": "float32", "_FillValue": 1e20},
        },
    )

    tunits = f"hours since {date.year}-{date.month:02d}-{date.day:02d} 00:00:00"

    sstdv_ready = sstdv.load()
    sstdv_ready.to_netcdf(
        os.path.join(output_root, f"{year}{month}{day}_ERA5dc_full_{cmor_var}.nc"),
        unlimited_dims="time",
        encoding={
            "time": {"dtype": "float64", "_FillValue": None, "units": tunits},
            "lon": {"dtype": "float32", "_FillValue": None},
            "lat": {"dtype": "float32", "_FillValue": None},
            cmor_var: {"dtype": "float32", "_FillValue": 1e20},
        },
    )

    #    sstndv["time"] = sstndv.time + pd.Timedelta(days=1)
    #    sstndv.to_netcdf(
    #        os.path.join(output_root, f"{year}{month}{day}_ERA5ndc_full_{cmor_var}.nc"),
    #        unlimited_dims="time",
    #        encoding={
    #            "time": {"dtype": "float64", "_FillValue": None, "units": tunits},
    #            "lon": {"dtype": "float32", "_FillValue": None},
    #            "lat": {"dtype": "float32", "_FillValue": None},
    #            cmor_var: {"dtype": "float32", "_FillValue": 1e20},
    #        },
    #    )

    # make sure time dimension of mask is identical
    mask25["time"] = sstdv.time.values
    #    mask25.to_netcdf(os.path.join(output_root, "mask25.nc"))

    result = sstdv.where(mask25).astype(np.float32)
    result_ready = result.load()

    result_ready.to_netcdf(
        os.path.join(output_root, f"{year}{month}{day}_ERA5dc_masked_{cmor_var}.nc"),
        unlimited_dims="time",
        encoding={
            "time": {"dtype": "float64", "_FillValue": None, "units": tunits},
            "lon": {"dtype": "float32", "_FillValue": None},
            "lat": {"dtype": "float32", "_FillValue": None},
            cmor_var: {"_FillValue": 1e20},
        },
    )

    #    result2 = sstndv.where(mask25).astype(np.float32)
    #
    #    result2.to_netcdf(
    #        os.path.join(output_root, f"{year}{month}{day}_ERA5ndc_masked_{cmor_var}.nc"),
    #        unlimited_dims="time",
    #        encoding={
    #            "time": {"dtype": "float64", "_FillValue": None, "units": tunits},
    #            "lon": {"dtype": "float32", "_FillValue": None},
    #            "lat": {"dtype": "float32", "_FillValue": None},
    #            cmor_var: {"_FillValue": 1e20},
    #        },
    #    )

    return None


# ---------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# set CCI variable to be processed
# sat_var = 'sea_surface_temperature_depth'
sat_var = "sea_surface_temperature"

# ERA5 long names and variable names to be processed
era5_vars = {
    "sea_surface_temperature": "sst",
    # "skin_temperature": "skt",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_cloud_cover": "tcc",
}
# corresponding CMOR variable names for ERA5 data (for output files)
cmor_var = "tos"

# set time range for downloading and processing
start_date = datetime(2021, 1, 1)
end_date = datetime(2021, 12, 31)

# Set the root directory for downloads and output
root_dir = "/work/bd0854/b380103/eso4clima/"
# Set the root directory for auxiliary data
aux_root_dir = "/home/b/b380103/eso4clima/"


loop_date = start_date

while loop_date <= end_date:
    print(
        f"Downloading and processing {loop_date.year}"
        f"-{loop_date.month:02d}-{loop_date.day:02d}"
    )
    download_and_process(
        loop_date, root_dir, aux_root_dir, sat_var, era5_vars, cmor_var
    )
    loop_date += relativedelta.relativedelta(days=1)
