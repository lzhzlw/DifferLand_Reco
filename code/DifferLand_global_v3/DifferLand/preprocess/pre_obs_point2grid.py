# Code for preprocessing point data to grid scale


import xarray as xr
import pandas as pd
import numpy as np
import os


def pre_obs_point2grid(varname, inpath, outdir_base, outdir_base_weight, ref_path, 
                       cols, longnames, units, data_sources, depth, overwrite):
    ''' Function to preprocess point data to grid scale
    
    :param varname: variable to preprocess
    :param inpath: input path
    :param outdir_base: output directory for data
    :param outdir_weight: output directory for weight
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param cols: columns to use dict with values: [lon, lat, value]
    :param shortnames: short name of variable
    :param longnames: long name of variable
    :param units: unit of variable
    :param data_sources: data source
    :param depth: depth
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''
    
    outname = f'{varname}_0_{int(depth)}cm_{data_sources[varname]}_0.5deg.nc'
    outdir = os.path.join(outdir_base, varname.split('_')[0])
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outname)
    outdir_weight = os.path.join(outdir_base_weight, varname.split('_')[0])
    os.makedirs(outdir_weight, exist_ok=True)
    outpath_weight = os.path.join(outdir_weight, outname)
    if os.path.exists(outpath):
        if overwrite:
            print(f'Overwriting {outpath}...')
        else:
            print(f'{outpath} already exists, skip preprocessing')
            
            return
    else:
        print(f'{outpath} does not exist, preprocessing...')
    
    # Load reference grid
    ref_ds = xr.open_dataset(ref_path)
    lon_ref = ref_ds['lon'].values
    lat_ref = ref_ds['lat'].values
    
    # Read point data
    df = pd.read_csv(inpath[varname], usecols=cols[varname])
    df = df.dropna() # remove rows with any missing values
    df = df.rename(columns={
        cols[varname][0]: 'lon', 
        cols[varname][1]: 'lat', 
        cols[varname][2]: varname
    })
    
    # Initialize arrays for statistics
    nlat, nlon = len(lat_ref), len(lon_ref)
    data_sum = np.zeros((nlat, nlon))
    data_sum_sq = np.zeros((nlat, nlon))
    data_count = np.zeros((nlat, nlon), dtype=int)
    # Assign each point to nearest grid cell and accumulate statistics
    print(f'Total valid points for {varname} from {data_sources[varname]}: {len(df)}') # print total valid points
    for _, row in df.iterrows():
        lon = row['lon']
        lat = row['lat']
        value = row[varname]
        # Find nearest grid cell
        lon_idx = np.argmin(np.abs(lon_ref - lon))
        lat_idx = np.argmin(np.abs(lat_ref - lat))
        # Accumulate statistics
        data_sum[lat_idx, lon_idx] += value
        data_sum_sq[lat_idx, lon_idx] += value ** 2
        data_count[lat_idx, lon_idx] += 1
    
    # Calculate global mean (for cells with no data)
    global_mean = np.nanmean(df[varname].values)
    # Initialize output arrays
    data_value = np.full((nlat, nlon), global_mean)  # Fill with global mean by default
    data_unc = np.zeros((nlat, nlon))
    # Calculate value and uncertainty for each cell
    mask_has_data = data_count > 0
    mask_multiple = data_count > 1
    # Cells with data: calculate mean
    data_value[mask_has_data] = data_sum[mask_has_data] / data_count[mask_has_data]
    # Cells with multiple data: calculate CV (std/mean)
    if np.any(mask_multiple):
        # Calculate variance and std for cells with multiple points
        variance = (data_sum_sq[mask_multiple] - 
                   data_sum[mask_multiple]**2 / data_count[mask_multiple]) / data_count[mask_multiple]
        std = np.sqrt(np.maximum(variance, 0))  # Avoid negative due to numerical error
        data_unc[mask_multiple] = std / data_value[mask_multiple]
    # Calculate weights for each cell
    data_weight = np.zeros((nlat, nlon))
    data_weight[mask_has_data] = 1.0
    
    # Create Dataset
    new_varname = varname.split('_')[0]
    ds = xr.Dataset(
        {
            new_varname: (['lat', 'lon'], data_value),
        },
        coords={
            'lon': lon_ref,
            'lat': lat_ref
        }
    )
    ds[new_varname].attrs = {
        "Standard name": longnames[varname],
        "Unit": units[varname],
        "Processing method": f"Gridded from point data, global mean used for empty cells",
        "Global_mean_used": float(global_mean),
        "Data source": data_sources[varname]
    }
    ds_weight = xr.Dataset(
        {
            new_varname: (['lat', 'lon'], data_weight)
        },
        coords={
            'lon': lon_ref,
            'lat': lat_ref
        }
    )
    ds_weight[new_varname].attrs = {
        "Standard name": f"Weight for {longnames[varname]}",
        "Unit": "unitless",
        "Processing method": "1 for cells with data, 0 for cells with no data",
        "Data source": data_sources[varname]
    }

    # Save files
    first_var = list(ref_ds.data_vars.keys())[0]
    ref_ds = ref_ds[first_var].isel(time=0)
    ds = ds.where(~np.isnan(ref_ds), np.nan)
    ds_weight = ds_weight.where(~np.isnan(ref_ds), np.nan)
    ds.to_netcdf(outpath)
    ds_weight.to_netcdf(outpath_weight)
    print(f"Saved: {outpath}")
    print(f"Saved: {outpath_weight}")

    return

