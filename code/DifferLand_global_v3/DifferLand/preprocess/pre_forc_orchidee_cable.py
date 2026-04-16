# Code for dealing with the soil temperature, matric potential,
# and soil moisture data from orchidee/cable
# Set the year as 2001


import xarray as xr
import pandas as pd
import os


def pre_forc_orchidee_cable(varname, inpath, outdir_base, ref_path, shortnames, 
                            longnames, units, data_sources, th_value, depth, overwrite):
    ''' Function to preprocess soil temperature, matric potential and soil moisture data from orchidee/cable
    
    :param varname: variable to preprocess
    :param inpath: input path
    :param outdir: output directory
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param shortnames: short names of variables
    :param longnames: long names of variables
    :param units: units of variables
    :param data_source: data source
    :param th_value: threshold value
    :param depth: depth of soil layer, cm
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f'{shortnames[varname]}_0_{int(depth)}cm_{data_sources[varname]}_2001_monthly_0.5deg.nc'
    outdir = os.path.join(outdir_base, shortnames[varname])
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outname)
    if os.path.exists(os.path.join(outdir, outname)):
        if overwrite:
            print(f'Overwriting {outpath}...')
        else:
            print(f'{outpath} already exists, skip preprocessing')
            
            return
    else:
        print(f'{outpath} does not exist, preprocessing...')
    
    ds = xr.open_dataset(inpath, decode_times=False)
    ds_var = ds[varname]
    ds_var = ds_var.where(ds_var > th_value)
    # Add new time info
    time_info = pd.date_range(start='2001-01-01', end='2001-12-31', freq='D')
    ds_var['time'] = time_info
    # Calculate multilayer average
    if depth <=20:
        ds_var = ds_var.sel(soil_hwsd=1)
    elif depth <= 40:
        ds_var = ds_var.sel(soil_hwsd=slice(1, 2))
    elif depth <= 60:
        ds_var = ds_var.sel(soil_hwsd=slice(1, 3))
    elif depth <= 80:
        ds_var = ds_var.sel(soil_hwsd=slice(1, 4))
    elif depth <= 100:
        ds_var = ds_var.sel(soil_hwsd=slice(1, 5))
    elif depth <= 150:
        ds_var = ds_var.sel(soil_hwsd=slice(1, 6))
    else:
        ds_var = ds_var.sel(soil_hwsd=slice(1, 7))
    if 'soil_hwsd' in ds_var.dims:
        ds_var = ds_var.mean(dim='soil_hwsd', skipna=True)
    if 'veget' in ds_var.dims:
        ds_var = ds_var.mean(dim='veget', skipna=True)
    if 'PFT' in ds_var.dims:
        ds_var = ds_var.mean(dim='PFT', skipna=True)
    # Calculate monthly average
    ds = ds_var.resample(time='1MS').mean(skipna=True)
    # Resample to 0.5 deg
    ref_ds = xr.open_dataset(ref_path)
    ds = ds.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')
    
    # Rename and update attributes
    ds = ds.rename(shortnames[varname])
    ds.attrs = {
        "Standard name": longnames[varname],
        "Unit": units[varname],
        "Processing method": f"Extract {varname} from {data_sources[varname]}"
    }
    # Save
    ds.to_netcdf(outpath)
    print(f'Saving {outpath}')

    return

