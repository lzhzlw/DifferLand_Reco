# Code for preprocessing gpp of FluxSat
# Resample to 0.5deg*0.5deg and monthly scale


import os
import xarray as xr
import numpy as np


def pre_obs_gpp_fluxsat(year, indir, outdir_base, ref_path, overwrite):
    ''' Function to preprocess GPP of FluxSat dataset
    
    :param year: year to preprocess
    :param indir: input directory
    :param outdir_base: output directory
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f'GPP_{int(year)}_monthly_0.5deg.nc'
    outdir = os.path.join(outdir_base, 'GPP')
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
        file_name = f'GPP_FluxSat_daily_v2_{str(year)}{i:02d}.nc4'
        file_path = os.path.join(indir, file_name)
        print(f'Preprocessing file: {file_path}')
        # Open the NetCDF file
        ds = xr.open_dataset(file_path)
        ds = ds['GPP']
        ds = xr.where(ds.values >= 0, ds, np.nan)
        # Resample the data to monthly averages
        ds = ds.resample(time="1MS").mean(skipna=True)
        # Align the lon and lat dimension to match the reference file
        ds = ds.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')
        ds_ls.append(ds)

    # merge all month
    ds = xr.concat(ds_ls, dim='time')
    # Set variable attributes
    ds.attrs['Unit'] = "g C m-2 day-1"
    ds.attrs['Standard name'] = "Gross primary productivity"
    ds.attrs['Processing method'] = "Monthly averages of daily sum GPP based on FluxSat v2"

    # Save the modified dataset to the output folder
    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return


