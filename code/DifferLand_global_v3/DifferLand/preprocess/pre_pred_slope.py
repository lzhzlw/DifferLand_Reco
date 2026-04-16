# Code for preprocessing slope based on elevation
# Refer to https://blog.csdn.net/weixin_45561357/article/details/106677574

import xarray as xr
import numpy as np
import rasterio
import os


def pre_pred_slope(inpath, outdir, ref_path, overwrite):
    ''' Function to preprocess slope data based on elevation
    
    :param inpath: input path for elevation data
    :param outpath: output directory
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f'Slope_0.5deg.nc'
    outpath = os.path.join(outdir, outname)
    if os.path.exists(outpath):
        if overwrite:
            print(f'Overwriting {outpath}...')
        else:
            print(f'{outpath} already exists, skip preprocessing')

            return
    else:
        print(f'{outpath} does not exist, preprocessing...')

    # Load data
    ref_ds = xr.open_dataset(ref_path)
    with rasterio.open(inpath) as src:
        data = src.read(1)
        transform = src.transform
        height, width = src.height, src.width
    x = np.arange(width)
    y = np.arange(height)
    lon = transform.c + transform.a * x
    lat = transform.f + transform.e * y
    # Save the processed data as a NetCDF file
    dem = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        name="Elevation"
    )
    # Get the resolution (assumes uniform grid in degrees)
    lat = dem.lat
    lon = dem.lon
    # Calculate grid spacing in meters (assuming EPSG:4326, 1 degree ~ 111.32 km)
    lat_res = np.abs(lat[1] - lat[0]) * 111321.4  # Latitude spacing in meters
    lon_res = np.abs(lon[1] - lon[0]) * 111321.4 * np.cos(np.radians(lat))  # Longitude spacing in meters
    # Compute the gradients in the x and y directions
    dz_dx = dem.differentiate("lon") / lon_res
    dz_dy = dem.differentiate("lat") / lat_res
    # Calculate slope in degrees
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * (180 / np.pi)
    # Save slope to a new NetCDF file
    ds = slope.to_dataset(name="Slope")
    ds['Slope'].attrs["Unit"] = 'deg'
    ds['Slope'].attrs["Standard name"] = 'Topographic_slope'
    ds['Slope'].attrs["Processing method"] = f"Calculated based on elevation"
    ds = ds.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')
    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return

