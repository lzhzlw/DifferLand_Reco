# Code for preprocessing reco of zeng et al., 2020
# 1. Resample to 0.5deg*0.5deg and monthly scale
# 2. Add time var, and remove doy var

import os
import xarray as xr
import numpy as np
from datetime import datetime, timedelta


def pre_obs_reco_zeng2020(year, indir, outdir_base, ref_path, overwrite):
    ''' Function to preprocess reco of zeng et al., 2020
    
    :param year: year to preprocess
    :param indir: input directory
    :param outdir_base: output directory
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f'Reco_{int(year)}_monthly_0.5deg.nc'
    outdir = os.path.join(outdir_base, 'Reco')
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
    file_name = f'RECO.{str(year)}.ver.2020.2.nc'
    file_path = os.path.join(indir, file_name)
    print(f"Preprocessing file: {file_path}")
    # Open the NetCDF file
    ds = xr.open_dataset(file_path)
    ds = ds.where(ds >= 0, np.nan)

    # Extract the DOY values
    doy = ds['doy'].values
    # Convert DOY to time based on the extracted year
    time_values = [datetime(year, 1, 1) + timedelta(days=int(d) - 1) for d in doy]
    # Replace DOY with TIME as a new dimension
    ds = ds.assign_coords(time=("doy", time_values)).swap_dims({"doy": "time"})
    # Optionally, drop the original DOY coordinate and unnecessary dimensions
    ds = ds.drop_vars(["doy", "crs"], errors="ignore")

    # Resample the data to monthly averages
    ds = ds.resample(time="1MS").mean(skipna=True)
    # Align the lon and lat dimension to match the reference file
    ds = ds.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')

    # Save the modified dataset to the output folder
    ds = ds.rename({'RECO': 'Reco'})
    del ds['Reco'].attrs['units']
    del ds['Reco'].attrs['long_name']
    ds['Reco'].attrs['Unit'] = "g C m-2 day-1"
    ds['Reco'].attrs['Standard name'] = "Ecosystem respiration"
    ds['Reco'].attrs['Processing method'] = "Monthly averages of daily sum Reco based on Zeng et al., 2020"
    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return

