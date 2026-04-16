# Code for preprocessing meteorogical forcing of CRUJRA dataset during 2001-2018
# Ta_max (tmax), Ta_min (tmin), Ta_mean (tmp), Prcp (pre), SSRD (tswrf)
# ALDS (dlwrf), Pres (pres), Wind (wind) 
# 1. Calculate daily max/min/mean/sum
# 2. Resample to monthly scale
# 3. Convert the unit


import os
import xarray as xr
import numpy as np


def pre_forc_meteo_crujra(varname, year, indir_ssdr, indir_other_meteo, outdir_base, 
                          methods, longnames, shortnames, units, overwrite):
    ''' Function to preprocess meteorological forcing of CRUJRA dataset
    
    :param varname: variable to preprocess
    :param year: year to preprocess
    :param indir_ssrdr: directory of ssdr
    :param indir_other_meteo: directory of other variables
    :param outdir_base: directory to store the processed data
    :param methods: methods to preprocess
    :longnames: long names of variables
    :shortnames: short names of variables
    :units: units of variables
    :overwrite: whether to overwrite existing files

    :return None

    '''

    outdir = os.path.join(outdir_base, shortnames[varname])
    os.makedirs(outdir, exist_ok=True)
    outname = f'{shortnames[varname]}_{str(year)}_monthly_0.5deg.nc'
    outpath = os.path.join(outdir, outname)
    if os.path.exists(outpath):
        if overwrite:
            print(f'Overwriting {outpath}...')
        else:
            print(f'{outpath} already exists, skip preprocessing')
            
            return
    else:
        print(f'{outpath} does not exist, preprocessing...')

    if varname == "tswrf":
        input_file = os.path.join(indir_ssdr, f"{varname}_v10_{str(year)}.nc")
    elif varname == "wind":
        input_file = os.path.join(indir_other_meteo, f"crujra.v2.2.5d.ugrd.{str(year)}.365d.noc.nc")
        input_file1 = os.path.join(indir_other_meteo, f"crujra.v2.2.5d.vgrd.{str(year)}.365d.noc.nc")
    else:
        input_file = os.path.join(indir_other_meteo, f"crujra.v2.2.5d.{varname}.{str(year)}.365d.noc.nc")
    print(f'Preprocessing: {input_file}')

    ds = xr.open_dataset(input_file)
    ds1 = xr.open_dataset(input_file1) if varname == "wind" else None

    # Resample to daily statistics
    if varname == 'tmax':
        ds = ds[varname].resample(time='1D').max(skipna=True)
    elif varname == 'tmin':
        ds = ds[varname].resample(time='1D').min(skipna=True)
    elif varname in ['tmp', 'pres']:
        ds = ds[varname].resample(time='1D').mean(skipna=True)
    elif varname in ['pre', 'tswrf', 'dlwrf']:
        ds = ds[varname].resample(time='1D').sum(skipna=True)
    elif varname == 'wind':
        ugrd = ds['ugrd'].resample(time='1D').mean(skipna=True)
        vgrd = ds1['vgrd'].resample(time='1D').mean(skipna=True)
        ds = np.sqrt(vgrd**2 + ugrd**2)

    # Calculate monthly statistics from daily data
    ds = ds.resample(time='1MS').mean(skipna=True)
    ds = xr.where(ds>=0, ds, np.nan)
    if varname in ['tmax', 'tmin', 'tmp']:
        ds = ds - 273.15
    elif varname == 'pres':
        ds = ds / 1000
    
    # Add attributes
    ds = ds.rename(shortnames[varname])
    ds.attrs['Unit'] = units[varname]
    ds.attrs['Standard name'] = longnames[varname]
    if varname == 'wind':
        ds.attrs['Processing method'] = f"Monthly averages of daily mean for wind, daily wind equals to sqrt(u2 + v2) based on CRUJRA v2.2"
    else:
        ds.attrs['Processing method'] = f"Monthly averages of daily {methods[varname]} for {shortnames[varname]} based on CRUJRA v2.2"

    # Save results
    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")
          
    return

