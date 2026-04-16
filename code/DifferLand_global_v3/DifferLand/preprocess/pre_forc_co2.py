# Code for preprocessing CO2 data
# 1. Rename coordinates to lon and lat
# 2. Select the near surface level co2, level=0
# 3. Convert units to ppm by multiplying 1000000
# 4. Resample to 0.5deg*0.5deg and monthly scale


import os
import xarray as xr
import numpy as np


def pre_forc_co2_cams(year, indir, outdir_base, ref_path, overwrite):
    ''' Function to preprocess CO2 of CAMS dataset
    
    :param year: year to preprocess
    :param indir: input directory
    :param outdir_base: output directory
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f'CO2_{int(year)}_monthly_0.5deg.nc'
    outdir = os.path.join(outdir_base, 'CO2')
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
    
    ref_ds = xr.open_dataset(ref_path)
    ds_ls = []
    for i in range(1, 13):
        file_name = f'cams73_latest_co2_conc_surface_inst_{str(year)}{i:02d}.nc'
        file_path = os.path.join(indir, file_name)
        print(f'Preprocessing file: {file_path}')

        # Open the NetCDF file
        ds = xr.open_dataset(file_path)
        # Rename coordinates to lon and lat
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
        # Select near-surface CO2 concentration (level=0) and convert units to ppm
        ds = ds['CO2'].sel(level=0)
        ds = ds * 1000000
        ds = xr.where(ds >= 0, ds, np.nan)

        # Resample the data to monthly averages
        ds = ds.resample(time="1MS").mean(skipna=True)

        # Align the lon and lat dimension to match the reference file
        ds = ds.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')

        # Append the dataset to the list
        ds_ls.append(ds)

    # Merge all months into one dataset
    ds = xr.concat(ds_ls, dim='time')
    ds = ds.rename('CO2')
    ds.attrs = {
        "Standard name": "Near surface CO2 concentration",
        "Unit": "ppm",
        "Processing method": "Monthly averages of daily mean for CO2 based on CAMS (level 0)"
    }

    # Save the merged dataset to the output folder
    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return

