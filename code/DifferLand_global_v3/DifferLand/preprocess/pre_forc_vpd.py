# Code for preprocessing vpd of CRUJRA dataset
# Calculation refer to https://github.com/MagicForrest/CRUJRA_4_GUESS/blob/master/scripts/proc.rhum.daily.sh
# Based on air temperature (tmp, deg C), pressure (pres, kpa), and specific humidity (spfh, kg/kg)


import os
import xarray as xr
import numpy as np


def pre_forc_vpd_crujra(year, indir, outdir_base, overwrite):
    ''' Function to preprocess vpd of CRUJRA dataset
    
    :param year: year to preprocess
    :param indir: input directory
    :param outdir_base: output directory
    :param overwrite: whether to overwrite existing files

    :return None

    '''

    outname = f'VPD_{str(year)}_monthly_0.5deg.nc'
    outdir = os.path.join(outdir_base, 'VPD')
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

    ds_ls = []
    for var in ['tmp', 'pres', 'spfh']:
        file_name = f'crujra.v2.2.5d.{var}.{str(year)}.365d.noc.nc'
        file_path = os.path.join(indir, file_name)
        ds = xr.open_dataset(file_path)
        
        # Convert unit to deg C for tmp or kpa for pres
        if var == 'tmp':
            ds = ds-273.15
        elif var == 'pres':
            ds = ds / 1000
        
        # Resample the data to monthly averages
        ds = ds.resample(time="1MS").mean(skipna=True)
        
        ds_ls.append(ds[var])

    tmp = ds_ls[0]
    pres = ds_ls[1]
    spfh = ds_ls[2]

    # Saturation vapor pressure (kpa), Buck equation
    svp = 0.61121 * np.exp((17.502 * tmp) / (tmp + 240.97))
    # Moisture Ratio (kg/kg)
    mr = spfh / (1 - spfh)
    # Actual Vapor Pressure (kpa)
    avp = (pres * mr) / (0.6221 + mr)
    # Vapor Pressure Deficit (hpa)
    vpd = (svp - avp) * 10
    vpd = xr.where(vpd < 0, 0, vpd)
    ds = vpd.to_dataset(name='VPD')

    ds['VPD'].attrs['Unit'] = 'hPa'
    ds['VPD'].attrs['Standard name'] = 'Vapor pressure deficit'
    ds['VPD'].attrs['Processing method'] = 'Calculated from air temperature, pressure, and specific humidity based on CRUJRA v2.2'

    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return

