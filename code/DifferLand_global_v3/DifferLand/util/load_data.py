# Functions to load data, including predictors, time-variant and invariant forcings and observations


import os
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
import pandas as pd
import re


def load_predictors(pred_path, region=None):
    '''
    Function to load predictor variables

    :param pred_path: (str) multi-year averaged predictor variables path ("xxxx.nc")
    :param region: (dict) region of interest, e.g. {"lat_min": -90, "lat_max": 90, "lon_min": -180, "lon_max": 180}

    :return predictor: (jnp.array) predictor variables (n_grid, 21)
    :return valid_pred: (jnp.array) bool mask for valid grids of predictor variables (n_grid, )

    '''

    # Read data
    ds = xr.open_dataset(pred_path)
    ds = ds.sortby(["lon", "lat"])
    # Select region
    if region:
        lat_min, lat_max = region.get("lat_min", -90), region.get("lat_max", 90)
        lon_min, lon_max = region.get("lon_min", -180), region.get("lon_max", 180)
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    # 19 other prediction variables except for lat, lon (1D)
    features = [var for var in ds.data_vars if var not in ['lat', 'lon']]
    predictor = jnp.stack([ds[var].values.ravel() for var in features], axis=1)
    # Convert 1D lat, lon to 2D, and add to the predictor variables
    lat = ds['lat'].values
    lon = ds['lon'].values
    lat_grid, lon_grid = jnp.broadcast_arrays(lat[:, None], lon[None, :])
    # Flatten lat and lon and add to the feature data
    lat_grid = lat_grid.ravel()
    lon_grid = lon_grid.ravel()
    lat_grid = lat_grid[:, None]
    lon_grid = lon_grid[:, None]
    predictor = jnp.hstack([predictor, lat_grid, lon_grid])
    # Standardization
    mean = jnp.nanmean(predictor, axis=0)
    std = jnp.where(jnp.nanstd(predictor, axis=0) == 0, 1.0, jnp.nanstd(predictor, axis=0))
    predictor = (predictor - mean) / std
    # Select non-NaN samples
    valid_pred = ~jnp.any(jnp.isnan(predictor), axis=1)
    ds.close()

    return predictor, valid_pred, lat, lon


def load_time_invariant_data(base_dir, var_name_ls, is_weight=False, region=None):
    '''
    Function to load time invariant forcings and observations

    :param base_dir: (str) dir of invariant forcings or observations, e.g. ./xxxx/xxxx/
    :param var_name_ls: (list) the variable list of invariant forcings or observations
    :param is_weight: (bool) whether to load grid weights
    :param region: (dict) region of interest, e.g. {"lat_min": -90, "lat_max": 90, "lon_min": -180, "lon_max": 180}

    :return invariant: (jnp.array) invariant forcings or observations (n_grid, n_var)
    :return valid_invar: (jnp.array) bool mask for valid grids of invariant forcings or observations (n_grid, )

    '''

    data_ls = []
    if not var_name_ls:

        return None, None

    else:
        nodata_vi_ls = [] # for which have no grid weights, set 1 as default
        for vi, var_name in enumerate(var_name_ls):
            if var_name == 'Lat':
                dir_var_name = var_name_ls[0]
                file_name = var_name_ls[0]
            else:
                file_name = var_name
                if re.search(r'SOC_|MAOC_|POC_', var_name):
                    dir_var_name = var_name.split('_')[0]
                else:
                    dir_var_name = var_name
            var_dir = os.path.join(base_dir, dir_var_name)
            if os.path.exists(var_dir):
                var_file_ls = os.listdir(var_dir)
                var_file_ls = [var_file for var_file in var_file_ls if file_name in var_file]
                if len(var_file_ls) > 0:
                    var_path = os.path.join(var_dir, var_file_ls[0])
                else:
                    nodata_vi_ls.append(vi)
                    continue
            else:
                nodata_vi_ls.append(vi)
                continue
            ds = xr.open_dataset(var_path, decode_cf=True, use_cftime=True)
            ds = ds.sortby(["lon", "lat"])
            if region:
                lat_min, lat_max = region.get("lat_min", -90), region.get("lat_max", 90)
                lon_min, lon_max = region.get("lon_min", -180), region.get("lon_max", 180)
                ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            if var_name == 'Lat':
                lat_values = ds.lat.values
                lon_values = ds.lon.values
                lat_grid = jnp.tile(lat_values[:, None], (1, len(lon_values)))
                lat_grid = lat_grid.ravel()
                data_ls.append(lat_grid)
            else:
                ds = ds[dir_var_name]
                var_data = ds.values.ravel()
                data_ls.append(var_data)
            
        if len(data_ls) == 0:
            return None, None
        elif len(data_ls) == 1:
            invariant = jnp.asarray(data_ls[0][:, None])
        else:
            invariant = jnp.stack(data_ls, axis=1)
        if is_weight:
            total_vars = len(var_name_ls)
            data_cols = jnp.array([vi for vi in range(total_vars) if vi not in nodata_vi_ls])
            invariant = jnp.ones((invariant.shape[0], total_vars)).at[:, data_cols].set(invariant)
        valid_invar = ~jnp.any(jnp.isnan(invariant), axis=1)

        return invariant, valid_invar


def load_time_variant_data(base_dir, var_name_ls, is_weight=False, region=None, year_range=[2001, 2018]):
    '''
    Function to load time variant forcings and observations

    :param base_dir: (str) dir of variant forcings or observations, e.g. ./xxxx/xxxx/
    :param var_name_ls: (list) the variable list of variant forcings or observations
    :param is_weight: (bool) whether to load grid weights
    :param region: (dict) region of interest, e.g. {"lat_min": -90, "lat_max": 90, "lon_min": -180, "lon_max": 180}
    :param year_range: (list) start year and end year of variant forcings and observations

    :return variant: (jnp.array) variant forcings or observations (n_grid, n_time, n_var)
    :return valid_var: (jnp.array) bool mask for valid grids of variant forcings or observations (n_grid, )

    '''

    data_ls = []
    if not var_name_ls:

        return None, None

    else:
        nodata_vi_ls = [] # for which have no grid weights, set 1 as default
        for vi, var_name in enumerate(var_name_ls):
            if var_name == 'DOY':
                dir_var_name = var_name_ls[1] # for select specific filedir and ds variable
                file_name = var_name_ls[1] # for select specific file
            else:
                file_name = var_name
                if re.search(r'Ts_|SM_|SMP_', var_name):
                    dir_var_name = var_name.split('_')[0]
                else:
                    dir_var_name = var_name
            var_dir = os.path.join(base_dir, dir_var_name)
            # Get list of files and ensure they are sorted
            var_file_ls = sorted(os.listdir(var_dir))
            var_file_ls = [var_file for var_file in var_file_ls if file_name in var_file]
            # Initialize an empty list for the datasets
            start_year, end_year = year_range
            year_files = [
                os.path.join(var_dir, f)
                for f in var_file_ls
                if any(str(year) in f for year in range(start_year, end_year + 1))
            ]
            if len(year_files) == 0:
                nodata_vi_ls.append(vi)
            else:
                var_ls = []
                for file_path in year_files:
                    ds = xr.open_dataset(file_path, decode_cf=True, use_cftime=True)
                    ds = ds.sortby(["lon", "lat", "time"])
                    # Filter by region if provided
                    if region:
                        lat_min, lat_max = region.get("lat_min", -90), region.get("lat_max", 90)
                        lon_min, lon_max = region.get("lon_min", -180), region.get("lon_max", 180)
                        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                    # Add filtered dataset to the list
                    var_ls.append(ds)
                # Merge datasets along the time dimension
                var_ds = xr.concat(var_ls, dim="time")
                var_data = var_ds[dir_var_name]
                _, lat, lon = var_data.shape
                if var_name == 'DOY':
                    # Add doy
                    time_vals = var_ds["time"]
                    time_vals = np.array(time_vals, dtype='datetime64[D]')
                    doy = np.array([t.dayofyear for t in pd.to_datetime(time_vals)]) + 14
                    doy_broadcasted = jnp.tile(doy[None, :], (lat * lon, 1))
                    data_ls.append(doy_broadcasted)
                else:
                    var_data = var_data.stack(latlon=("lat", "lon"))
                    var_data = var_data.transpose("latlon", "time")
                    data_ls.append(var_data.values)
        variant = jnp.stack(data_ls, axis=-1)
        if is_weight:
            total_vars = len(var_name_ls)
            data_cols = jnp.array([vi for vi in range(total_vars) if vi not in nodata_vi_ls])
            variant = jnp.ones((variant.shape[0], variant.shape[1], total_vars)).at[:, :, data_cols].set(variant)
        valid_var = ~jnp.any(jnp.isnan(variant), axis=(1, 2))

        return variant, valid_var


def split_combined_mask(*masks, grid_weights, p2g_idx, train_ratio=0.8, develop_ratio=0.1, random_seed=42):
    '''
    Function to combine all valid mask and split them into 3 parts
    training index (80%), developing index (10%), and testing index (10%), jnp.array

    '''

    masks = [jnp.asarray(mask) for mask in masks if mask is not None]
    valid_mask = masks[0]
    for mask in masks[1:]:
        valid_mask = jnp.logical_and(valid_mask, mask)
    # Get true array
    true_array = valid_mask[valid_mask]
    true_indices = jnp.arange(len(true_array))
    key = jax.random.PRNGKey(random_seed)

    ref_min_samples = 100
    if len(p2g_idx) == 0:
        # Devided into 3 parts
        n_total = len(true_indices)
        min_samples = jnp.minimum(ref_min_samples, n_total)
        n_train = jnp.maximum(min_samples, int(n_total * train_ratio))
        n_develop = jnp.maximum(min_samples, int(n_total * develop_ratio))
        n_test = jnp.maximum(min_samples, n_total - n_train - n_develop)

        key, subkey = jax.random.split(key)
        true_indices_shuffle = jax.random.permutation(subkey, true_indices)
        if n_total >= 10 * min_samples:
            train_indices = true_indices_shuffle[:n_train]
            develop_indices = true_indices_shuffle[n_train:n_train+n_develop]
            test_indices = true_indices_shuffle[n_total-n_test:]
        else:
            train_indices = jax.random.choice(subkey, true_indices_shuffle, (n_train, ), replace=False)
            key, subkey = jax.random.split(key)
            true_indices_shuffle = jax.random.permutation(subkey, true_indices)
            develop_indices = jax.random.choice(subkey, true_indices_shuffle, (n_develop, ), replace=False)
            key, subkey = jax.random.split(key)
            true_indices_shuffle = jax.random.permutation(subkey, true_indices)
            test_indices = jax.random.choice(subkey, true_indices_shuffle, (n_test, ), replace=False)
        train_indices_ls = [train_indices]
    else:
        # Divide into 3 parts but ensure that each parts include proportional point-to-grid samples with non-zero weights
        grid_weights = grid_weights[valid_mask][:, p2g_idx]
        n_var = len(p2g_idx)
        grid_mask = jnp.where(grid_weights > 0, 1, 0)
        powers = jnp.power(10, jnp.arange(n_var-1, -1, -1))
        classify = jnp.sum(grid_mask * powers, axis=1)
        unique_ls = jnp.unique(classify)
        train_indices_ls = []
        develop_indices_ls = []
        test_indices_ls = []
        for unique in unique_ls:
            true_indices_var = true_indices[classify == unique]
            if len(true_indices_var) == 0:
                continue

            # Devided into 3 parts
            n_total_var = len(true_indices_var)
            min_samples = jnp.minimum(ref_min_samples, n_total_var)
            n_train_var = jnp.maximum(min_samples, int(n_total_var * train_ratio))
            n_develop_var = jnp.maximum(min_samples, int(n_total_var * develop_ratio))
            n_test_var = jnp.maximum(min_samples, n_total_var - n_train_var - n_develop_var)

            key, subkey = jax.random.split(key)
            true_indices_shuffle = jax.random.permutation(subkey, true_indices_var)
            if n_total_var >= 10*min_samples:
                train_indices_ls.append(true_indices_shuffle[:n_train_var])
                develop_indices_ls.append(true_indices_shuffle[n_train_var:n_train_var+n_develop_var])
                test_indices_ls.append(true_indices_shuffle[n_total_var-n_test_var:])
            else:
                train_indices_ls.append(jax.random.choice(subkey, true_indices_shuffle, (n_train_var, ), replace=False))
                key, subkey = jax.random.split(key)
                true_indices_shuffle = jax.random.permutation(subkey, true_indices_var)
                develop_indices_ls.append(jax.random.choice(subkey, true_indices_shuffle, (n_develop_var, ), replace=False))
                key, subkey = jax.random.split(key)
                true_indices_shuffle = jax.random.permutation(subkey, true_indices_var)
                test_indices_ls.append(jax.random.choice(subkey, true_indices_shuffle, (n_test_var, ), replace=False))

        train_indices = jnp.concatenate(train_indices_ls)
        develop_indices = jnp.concatenate(develop_indices_ls)
        test_indices = jnp.concatenate(test_indices_ls)

    # Create new mask
    train_mask = jnp.zeros_like(true_array, dtype=bool)
    develop_mask = jnp.zeros_like(true_array, dtype=bool)
    test_mask = jnp.zeros_like(true_array, dtype=bool)

    train_mask = train_mask.at[train_indices].set(True)
    develop_mask = develop_mask.at[develop_indices].set(True)
    test_mask = test_mask.at[test_indices].set(True)

    train_mask_ls = []
    for train_indices in train_indices_ls:
        train_mask_var = jnp.zeros_like(true_array, dtype=bool)
        train_mask_var = train_mask_var.at[train_indices].set(True)
        train_mask_ls.append(train_mask_var)
    
    # jax.debug.breakpoint()

    return train_mask_ls, train_mask, develop_mask, test_mask, valid_mask


def load_all_data(pred_path, forc_dir, ivf_ls, vf_ls, obs_dir, ivobs_ls, vobs_ls,
                  grid_weight_dir, uncertainty, region=None, year_range=None, random_seed=42):
    '''
    Function to load all data

    :param predictor_path: (str) predictor variables path, e.g., ./xxxx/xxxx.nc
    :param forcing_dir: (str) forcing variables path, e.g., ./xxxx/xxxx/
    :param ivf_ls: (list) invariant forcing variables list
    :param vf_ls: (list) variant forcing variables list
    :param obs_dir: (str) observation variables path, e.g., ./xxxx/xxxx/
    :param ivobs_ls: (list) invariant observation variables list
    :param vobs_ls: (list) variant observation variables list
    :param grid_weight_dir: (str) grid weight variables path, e.g., ./xxxx/xxxx/
    :param region: (dict) region of interest, e.g. {"lat_min": -90, "lat_max": 90, "lon_min": -180, "lon_max": 180}
    :param year_range: (list) start year and end year of variant forcings and observations
    :param random_seed: (int) random seed

    '''

    # Load predictive variables
    pred, valid_pred, lat, lon = load_predictors(pred_path, region)
    # Load forcings
    invar_forc, valid_invar_forc = load_time_invariant_data(forc_dir, ivf_ls, False, region)
    var_forc, valid_var_forc = load_time_variant_data(forc_dir, vf_ls, False, region, year_range)
    # Load observations
    invar_obs, valid_invar_obs = load_time_invariant_data(obs_dir, ivobs_ls, False, region)
    var_obs, valid_var_obs = load_time_variant_data(obs_dir, vobs_ls, False, region, year_range)
    valid_obs = jnp.logical_and(valid_invar_obs, valid_var_obs)
    # Load weights, here we DO NOT consider the uncertianty/weight at temporal dimension
    obs_name_ls = vobs_ls + ivobs_ls
    grid_weig, valid_grid_weig = load_time_invariant_data(grid_weight_dir, obs_name_ls, True, region)
    grid_idx = [idx for idx, name in enumerate(obs_name_ls) if "p2g" not in name]
    if grid_weig is None:
        grid_weig = jnp.ones((pred.shape[0], len(obs_name_ls)))
        valid_grid_weig = jnp.ones(pred.shape[0], dtype=bool)
    else:
        if uncertainty and len(grid_idx) > 0:
            grid_weig = grid_weig.at[:, grid_idx].set(1.0)

    # Split data
    p2g_idx = [idx for idx, name in enumerate(obs_name_ls) if "p2g" in name]
    train_mask_ls, train_mask, develop_mask, test_mask, valid_mask = split_combined_mask(
        valid_pred, valid_invar_forc, valid_var_forc, valid_obs, valid_grid_weig, grid_weights=grid_weig,
        p2g_idx=p2g_idx, train_ratio=0.8, develop_ratio=0.1, random_seed=random_seed
    )

    pred = pred[valid_mask]
    invar_forc = invar_forc[valid_mask]
    var_forc = var_forc[valid_mask]
    invar_obs = invar_obs[valid_mask]
    var_obs = var_obs[valid_mask]
    grid_weig = grid_weig[valid_mask]
    
    # jax.debug.breakpoint()

    return pred, invar_forc, var_forc, invar_obs, var_obs, grid_weig, train_mask_ls, train_mask, develop_mask, test_mask, valid_mask, lat, lon


def restore_to_original_length(mask_data, valid_mask, fill_value=jnp.nan):
    '''
    Function to restore a valid array to its original length
    invalid positions are filled with "fill_value"

    '''

    # Create a full array with the length of valid_mask and other dimensions (generally no or one) of mask_data
    restored_data = jnp.full((valid_mask.shape[0],) + mask_data.shape[1:], fill_value)
    # Insert valid values into their original positions using the valid_mask
    restored_data = restored_data.at[valid_mask].set(mask_data)

    return restored_data

