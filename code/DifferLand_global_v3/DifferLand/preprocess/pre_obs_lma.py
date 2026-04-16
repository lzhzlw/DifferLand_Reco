# Code for preprocessing LMA, leaf mass per area (g m-2) of Butler et al. 2017
# From specific leaf area (SLA, m2/kg) to LMA
# LMA = 1000/SLA


import pandas as pd
import numpy as np
import xarray as xr
import os


def pre_obs_lma_butler2017(inpath, outdir_base, ref_path, overwrite):
    ''' Function to preprocess LMA of Butler et al. 2017
    
    :param inpath: input csv file path
    :param outdir_base: output directory for LMA
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outdir_lma = os.path.join(outdir_base, 'LMA')
    os.makedirs(outdir_lma, exist_ok=True)
    outdir_sla = os.path.dirname(inpath)
    outpath_lma = os.path.join(outdir_lma, f"LMA_0.5deg.nc")
    outpath_sla = os.path.join(outdir_sla, f"SLA_0.5deg.nc")
    if os.path.exists(outpath_lma):
        if overwrite:
            print(f'Overwriting {outpath_lma}...')
            print(f'Overwriting {outpath_sla}...')
        else:
            print(f'{outpath_lma} already exists, skip preprocessing')
            print(f'{outpath_sla} already exists, skip preprocessing')

            return
    else:
        print(f'{outpath_lma} does not exist, preprocessing...')
        print(f'{outpath_sla} does not exist, preprocessing...')

   # Read ref nc
    ref_ds = xr.open_dataset(ref_path)
    lon_ref = ref_ds['lon'].values
    lat_ref = ref_ds['lat'].values
    # Read sla data
    df = pd.read_csv(inpath, usecols=['lon', 'lat', 'mean'])
    # Calculate lma
    df['LMA'] = 1000 / df['mean']

    # Create new nc
    lon_2d, _ = np.meshgrid(lon_ref, lat_ref)
    LMA_data = np.full(lon_2d.shape, np.nan)
    SLA_data = np.full(lon_2d.shape, np.nan)
    for _, row in df.iterrows():
        lon = row['lon']
        lat = row['lat']
        LMA = row['LMA']
        SLA = row['mean']
        lon_idx = np.argmin(np.abs(lon_ref - lon))
        lat_idx = np.argmin(np.abs(lat_ref - lat))
        LMA_data[lat_idx, lon_idx] = LMA
        SLA_data[lat_idx, lon_idx] = SLA
    LMA_ds = xr.Dataset(
        {
            'LMA': (['lat', 'lon'], LMA_data)
        },
        coords={
            'lon': lon_ref,
            'lat': lat_ref
        }
    )
    LMA_ds["LMA"].attrs = {
        "Standard name": "Leaf mass per leaf area",
        "Unit": "g m-2",
        "Processing method": "Snapshot of LMA based on Butler et al. 2017"
    }
    SLA_ds = xr.Dataset(
        {
            'SLA': (['lat', 'lon'], SLA_data)
        },
        coords={
            'lon': lon_ref,
            'lat': lat_ref
        }
    )
    SLA_ds["SLA"].attrs = {
        "Standard name": "Specific leaf area",
        "Unit": "m2 kg-1",
        "Processing method": "Snapshot of SLA based on Butler et al. 2017"
    }

    # Save as NetCDF for the year
    LMA_ds.to_netcdf(outpath_lma)
    SLA_ds.to_netcdf(outpath_sla)
    print(f"Saving {outpath_lma}\nSaving {outpath_sla}")

    return

