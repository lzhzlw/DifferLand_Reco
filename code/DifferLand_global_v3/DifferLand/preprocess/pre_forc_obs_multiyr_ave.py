# Code for calculating the multiyear averages of GPP, LAI, Reco, ET, air temperature, and precipitation


import xarray as xr
import os


def pre_forc_obs_multiyr_ave(varname, dir_obs, dir_forc, start_yr, end_yr, overwrite):
    ''' Function to calculate the multiyear averages of GPP, LAI, Reco, ET, air temperature, and precipitation
    
    :param varname: variable name
    :param dir_obs: directory for observations
    :param dir_forc: directory for forcings
    :param start_yr: start year
    :param end_yr: end year
    :param overwrite: whether to overwrite existing files
    
    :return None

    '''

    outdir = os.path.join(dir_forc, f'ave_{varname}')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'ave_{varname}_0.5deg_{start_yr}_{end_yr}.nc')
    if os.path.exists(outpath):
        if overwrite:
            print(f'Overwriting {outpath}...')
        else:
            print(f'{outpath} already exists, skip preprocessing')
            
            return
    else:
        print(f'{outpath} does not exist, preprocessing...')
    
    years = range(start_yr, end_yr + 1)
    ds_ls = []
    indir_base = dir_obs if varname in ['GPP', 'LAI', 'Reco', 'ET'] else dir_forc
    indir = os.path.join(indir_base, varname)
    for year in years:
        file_path = os.path.join(indir, f'{varname}_{str(year)}_monthly_0.5deg.nc')
        ds = xr.open_dataset(file_path)
        ds_ls.append(ds)

    ds_merged = xr.concat(ds_ls, dim='time')
    ds_merged = ds_merged.mean(dim='time', skipna=True)
    ds_merged[varname].attrs['Unit'] = ds[varname].attrs['Unit']
    ds_merged[varname].attrs['Standard name'] = ds[varname].attrs['Standard name']
    ds_merged[varname].attrs['Processing method'] = f'Multiyear averages of {varname} during {start_yr}-{end_yr}'
    ds_merged = ds_merged.rename({varname: f'ave_{varname}'})

    # Save result
    ds_merged.to_netcdf(outpath)
    print(f'Saving {outpath}')
    
    return

