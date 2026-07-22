import xarray as xr

sat_var = "sea_surface_temperature"

mask25 = xr.open_dataset("output/mask25.nc")
l3c25 = xr.open_dataset("output/20210101_L3C_sea_surface_temperature.nc")

time = mask25["time"]
l3c2524h = None

# create new dataset that contains copies of the l3c25 fields
# for 24 timesteps in total
for timestep in time:
    newstep = l3c25.assign_coords(time=[timestep.values])
    if l3c2524h is None:
        l3c2524h = newstep
    else:
        l3c2524h = xr.concat([l3c2524h, newstep], dim="time")

# Apply mask
mask = mask25["__xarray_dataarray_variable__"] == 1
result = l3c2524h.where(mask)

# Save result to netCDF
result.to_netcdf(
    "test.nc",
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
