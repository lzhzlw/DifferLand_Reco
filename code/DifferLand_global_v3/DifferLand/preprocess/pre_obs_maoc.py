# Code for preprocessing the MAOC data from Georgiou et al. 2022


import xarray as xr
import pandas as pd
import numpy as np
import os


def pre_obs_maoc_georgiou2022(inpath, outdir_base, ref_path, overwrite):
    ''' Function to preprocess MAOC data from Georgiou et al. 2022
    
    :param inpath: input path
    :param outdir_base: output directory
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f'MAOC_0_100cm_0.5deg.nc'
    outdir = os.path.join(outdir_base, 'MAOC')
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
    ds = xr.open_dataset(inpath)
    ds = (ds['moc_s'] + ds['moc_t']) * 1000
    # Resample
    ref_ds = xr.open_dataset(ref_path)
    ds = ds.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')
    # Rename and update attributes
    ds = ds.rename('MAOC')
    ds.attrs.update({
        "Standard name": 'Mineral associated organic carbon',
        "Unit": 'g C m-2',
        "Processing method": 'From Georgiou et al. 2022'
    })
    # Save result
    ds.to_netcdf(outpath)
    print(f'Saved {outpath}')

    return


def pre_obs_maoc_point(inpath_maoc, inpath_soil, outpath, depth, overwrite):
    ''' Function to preprocess MAOC data from Huang et al. 2025

    :param inpath: input path for MAOC
    :param inpath_soil: input path for soil properties
    :param outpath: output path
    :param depth: soil depth in cm
    :param overwrite: whether to overwrite existing files

    :return None

    '''

    if os.path.exists(outpath):
        if overwrite:
            print(f'Overwriting {outpath}...')
        else:
            print(f'{outpath} already exists, skip preprocessing')
            
            return
    else:
        print(f'{outpath} does not exist, preprocessing...')
    
    data_source = os.path.basename(outpath).split('.')[0].split('_')[-1]

    # Load data
    if data_source == 'yyhuang':
        df = pd.read_csv(inpath_maoc, usecols=['lat', 'lon', 'upp', 'bot', 'MAOC', 'POC'])
        col_ls = ['MAOC', 'POC']
    elif data_source == 'wosis':
        df = pd.read_csv(inpath_maoc, sep='\t', usecols=['latitude', 'longitude', 'upper_depth', 'lower_depth', 'value_avg'])
        df = df.rename(columns={
            'latitude': 'lat',
            'longitude': 'lon',
            'upper_depth': 'upp',
            'lower_depth': 'bot',
            'value_avg': 'SOC',
        })
        col_ls = ['SOC']
    df = df.apply(pd.to_numeric, errors='coerce')
    # Select the first layer
    df = df[df['upp'] == 0]
    # Remove row with any missing value
    df = df.dropna()
    df['Thick'] = df['bot'] - df['upp'] # cm
    # Load soil properties
    ds = xr.open_dataset(inpath_soil)
    # Get 'BD'  and 'CFVO' for each row from soil properties
    for idx, row in df.iterrows():
        df.loc[idx, 'BD'] = ds['BD'].sel(lat=row['lat'], lon=row['lon'], method='nearest').values # kg m-3
        df.loc[idx, 'CFVO'] = ds['CFVO'].sel(lat=row['lat'], lon=row['lon'], method='nearest').values # cm3 cm-3
    beta = 0.9786
    for col in col_ls:
        # Convert POC, MAOC, and SOC from g kg-1 to g C m-2
        df[col] = df[col] * df['BD'] * df['Thick']/100 * (1 - df['CFVO'])
        # Standardization to depth
        df[col] = df[col] * (1 - np.power(beta, depth))/(1 - np.power(beta, df['Thick']))
    # Calculate SOC
    if data_source == 'yyhuang':
        df['SOC'] = df['MAOC'] + df['POC']
        col_ls.append('SOC')
    # Group by lon and lat
    df = df.dropna()
    df = df.groupby(['lon', 'lat'])[col_ls].mean().reset_index()

    # Save result
    df.to_csv(outpath, index=False)
    print(f'Saving {outpath}')

    return

