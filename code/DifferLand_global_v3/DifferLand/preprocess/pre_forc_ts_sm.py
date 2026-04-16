# Code for preprocessing soil temperature and soil moisture data
# 1. Rename coordinates to lon and lat
# 2. Convert units to deg C by minus 273.15
# 3. Resample to 0.5deg*0.5deg and monthly scale


import os
import xarray as xr
import numpy as np


def pre_forc_ts_sm_era5l(varname, year, inpath, outdir_base, ref_path, 
                         old_varnames, longnames, units, depth, overwrite):
    ''' Function to preprocess soil temperature and soil moisture of ERA5 dataset
    
    :param year: year to preprocess
    :param varname: variable name
    :param inpath: input path
    :param outdir_base: output directory
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param old_varnames: old variable names in the input dataset
    :param longnames: long names for the variable
    :param units: units for the variable
    :param depth: depth to extract
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f'{varname}_0_{int(depth)}cm_ERA5L_{int(year)}_monthly_0.5deg.nc'
    outdir = os.path.join(outdir_base[varname], varname)
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
    ds = xr.open_dataset(inpath[varname])
    # Rename coordinates to lon and lat
    ds = ds.rename({'longitude': 'lon', 'latitude': 'lat', 'valid_time': 'time'})
    # Convert lon from 0-360 to -180-180
    ds['lon'] = xr.where(ds['lon'] > 180, ds['lon'] - 360, ds['lon'])
    # Select specified year
    ds = ds.sel(time=str(year))
    ds = ds.sortby(['lat', 'lon', 'time'])
    # Select specified depth (0-7, 7-28, 28-100, and 100-289 cm)
    vars = [f'{old_varnames[varname]}{i}' for i in range(1, 5)]
    if depth <= 7:
        ds = ds[vars[0]]
    elif depth <= 28:
        ds = (ds[vars[0]] + ds[vars[1]]) / 2
    elif depth <= 100:
        ds = (ds[vars[0]] + ds[vars[1]] + ds[vars[2]]) / 3
    else:
        ds = (ds[vars[0]] + ds[vars[1]] + ds[vars[2]] + ds[vars[3]]) / 4
    # Convert units to deg C
    if varname == 'Ts':
        ds = ds - 273.15
    # Resample to 0.5deg*0.5deg and monthly scale
    ds = ds.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')
    # Rename and update attributes
    ds = ds.rename(varname)
    ds.attrs = {
        "Standard name": longnames[varname],
        "Unit": units[varname],
        "Processing method": f"Monthly averages of daily mean for {longnames[varname]} at {depth} cm based on ERA5-Land"
    }
    # Save the merged dataset to the output folder
    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return

