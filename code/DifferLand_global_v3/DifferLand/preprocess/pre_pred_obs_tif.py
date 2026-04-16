# Code for preprossing snapshot data (.tif file) of observations or predictors
# Observations, Rh from SRDB
# 1. Convert from .tif to .nc
# 2. From g C/m2/yr, converted to g C/m2/d, divided Rh by 365.25
# 3. Resample to 0.5deg*0.5deg
# Predictors, Elevation, Climate zone, Biome zone, Soil type, and Canopy height
# 1. Rename coordinates to lon and lat
# 2. Use forest age with 10% tree cover correction
# 3. Set the age of non-forest as 1
# 4. Resample to 0.5deg*0.5deg


import rasterio
import xarray as xr
import numpy as np
import sys
import os


def pre_pred_obs_tif(varname, inpath, outdir_obs, outdir_pred, ref_path,
                     longnames, units, data_sources, overwrite):
    ''' Function to preprocess snapshot data (.tif file) of observations or predictors
    Rh, Elevation, Climate zone, Biome zone, Soil type, and Canopy height
    
    :param varname: variable to preprocess
    :param inpath: input path
    :param outdir_obs: output directory for observations
    :param outdir_pred: output directory for predictors
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param longnames: long names of variables
    :param units: units of variables
    :param data_sources: data sources
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f"{varname}_0.5deg.nc"
    if varname == "Rh":
        outdir = outdir_obs
        outpath = os.path.join(outdir, varname, outname)
    else:
        outdir = outdir_pred
        outpath = os.path.join(outdir, outname)
    if os.path.exists(outpath):
        if overwrite:
            print(f'Overwriting {outpath}...')

        else:
            print(f'{outpath} already exists, skip preprocessing')

            return
    else:
        print(f'{outpath} does not exist, preprocessing...')

    # Read the reference NetCDF file
    ref_ds = xr.open_dataset(ref_path)
    # Read tif
    with rasterio.open(inpath[varname]) as src:
        data = src.read(1)
        transform = src.transform
        height, width = src.height, src.width
        crs = src.crs
    # Correctly calculate latitude and longitude as 2D arrays
    x = np.arange(width)
    y = np.arange(height)
    lon = transform.c + transform.a * x
    lat = transform.f + transform.e * y
    # Create xarray Dataset for the year
    ds = xr.Dataset(
        {
            varname: (["lat", "lon"], data)
        },
        coords={
            "lat": lat,
            "lon": lon,
        },
        attrs={"crs": str(crs)}
    )
    if varname == 'Rh':
        # Convert from g C/m2/yr to g C/m2/d
        ds = ds/365.25
        ds = ds.where(data >= 0)
    elif varname == 'Climate_type':
        ds = ds.where(ds > 0)
    elif varname in ['Biome_type', 'Soil_type']:
        ds = ds.where((ds > 0) & (ds < 255))
    elif varname == 'Canopy_height':
        ds = ds.where(ds >= 0)

    # Add attributes
    ds[varname].attrs = {
        "Standard name": longnames[varname],
        "Unit": units[varname],
        "Processing method": f"Snapshot of {varname} based on {data_sources[varname]}"
    }
    # Resample
    ds = ds.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')
    # Save as NetCDF for the year
    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return

