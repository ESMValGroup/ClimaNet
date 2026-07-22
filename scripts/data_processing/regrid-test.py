# import matplotlib.pyplot as plt
# import numpy as np
import xarray as xr
import xesmf as xe
# import numpy as np
# from cartopy.util import add_cyclic_point

# era5 = xr.open_dataset("/work/bd0854/b380103/eso4clima/download/era5/total_column_water_vapour#/era5_total_column_water_vapour_20200101.nc")
# hoaps = xr.open_dataset("/work/bd0854/b380103/eso4clima/download/HOAPS/hoaps-c.r30.h01.wvpa.2020-01-01.nc")
#
# print(era5)
# print(hoaps)
#
# regridder = xe.Regridder(era5, hoaps, "bilinear") #"conservative")
#
# era5regridded = regridder(era5, keep_attrs=True)
# print(era5regridded)
#
# era5regridded.to_netcdf("/work/bd0854/b380103/eso4clima/era5regridded-bilinear.nc")


l3c = xr.open_dataset(
    "/work/bd0854/b380103/eso4clima/output/20210101_L3C_sea_surface_temperature.nc"
)
era5 = xr.open_dataset(
    "/work/bd0854/b380103/eso4clima/download/era5/sea_surface_temperature/era5_sea_surface_temperature_20200101.nc"
)

# Pad the data and coordinates to make the dataset cyclic
# cyclic_data, cyclic_lon = add_cyclic_point(era5['sst'].values, coord=era5['longitude'].values)

# print(len(cyclic_lon))
# print(len(l3c['lon'].values))

# Rebuild xarray DataArray with the newly wrapped arrays
# era5_cyclic = xr.Dataset(
#    {'sst': (['lat', 'lon'], cyclic_data)},
#    coords={'lat': era5['latitude'], 'lon': cyclic_lon}
# )

regridder = xe.Regridder(era5, l3c, "bilinear", periodic=True)  # "conservative")

era5regridded = regridder(era5, keep_attrs=True)

era5regridded.to_netcdf("/work/bd0854/b380103/eso4clima/era5regridded-bilinear.nc")
