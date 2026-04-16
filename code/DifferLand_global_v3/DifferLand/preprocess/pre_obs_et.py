# Code for preprocessing et of DOLCE dataset
# Resample to 0.5 deg * 0.5 deg
# Convert unit, 1 W/m2 = 24*60*60/(2.45*10^6) mm/day


import os
import xarray as xr
import numpy as np


def pre_obs_et_dolce(year, indir, outdir_base, ref_path, overwrite):
    ''' Function to preprocess et of DOLCE dataset
    
    :param year: year to preprocess
    :param indir: input directory
    :param outdir_base: output directory
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f'ET_{year}_monthly_0.5deg.nc'
    outdir = os.path.join(outdir_base, 'ET')
    os.makedirs(outdir, exist_ok=True)
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
    file_name = f'DOLCE_v3_{year}.nc'
    file_path = os.path.join(indir, file_name)
    ds = xr.open_dataset(file_path)
    ds = ds.rename({'hfls': 'ET'})
    ds = ds['ET']

    # Convert unit
    ds = ds * 24 * 36 / (2.45 * 10000)
    notnanmask = ~np.isnan(ds)
    ds = xr.where(ds>=0, ds, 0)
    ds = xr.where(notnanmask, ds, np.nan)
    # Resample
    ds = ds.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')
    # Set variable attributes
    ds.attrs['Unit'] = "mm day-1"
    ds.attrs['Standard name'] = "Evapotranspiration"
    ds.attrs['Processing method'] = "Monthly averages of daily sum ET based on DOLCE v3"

    # Save result
    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return

