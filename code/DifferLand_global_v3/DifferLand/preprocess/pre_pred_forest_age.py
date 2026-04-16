# Code for preprocessing forest age from BGIForestAgeMP
# 1. Rename coordinates to lon and lat
# 2. Use forest age with 10% tree cover correction
# 3. Set the age of non-forest as 1
# 4. Resample to 0.5deg*0.5deg

import xarray as xr
import numpy as np
import os


def pre_pred_age_bgi(inpath, outdir, ref_path, overwrite):
    ''' Function to preprocess forest age data from BGIForestAgeMP
    
    :param inpath: input path for forest age data
    :param outpath: output directory
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f'Forest_age_0.5deg.nc'
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
    ds = xr.open_dataset(inpath)
    ref_ds = xr.open_dataset(ref_path)
    ds = ds["ForestAge_TC010"].squeeze()  # forest age with 10% tree cover correction
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    ds = ds.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')
    # Replace missing values in forest age data with 1
    ds = ds.fillna(1)
    # Set forest age values to NaN where ref_ds are NaN
    first_var = list(ref_ds.data_vars.keys())[0]
    ref_ds = ref_ds[first_var].isel(time=0)
    ds = ds.where(~np.isnan(ref_ds), np.nan)
    # Rename the variable to "forest_age"
    ds = ds.rename("Forest_age")
    # Update metadata for the final processed dataset
    del ds.attrs['units']
    del ds.attrs['long_name']
    ds.attrs = {
        "Standard name": "Forest age",
        "Unit": "years",
        "Processing method": "Estimates with a 10% tree cover correction, non-forest pixels were assigned a value of 1, and ocean regions identified by NaN, based on BGIForestAgeMP)."
    }

    # Save the processed forest age data to a new NetCDF file
    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return

