# Code for calculating the multiyear avereages of seasonality (standard deviation) of air temperature and precipitation


import xarray as xr
import os


# Define the function to process each variable
def pre_pred_seasonality_crujra(varname, indir_base, outdir, start_yr, end_yr, 
                                shortnames, longnames, units, overwrite):
    ''' Function to calculate the multiyear averages of seasonality of air temperature and precipitation
    
    :param varname: variable to preprocess
    :param inpath: input path
    :param outdir: output directory
    :param start_yr: start year
    :param end_yr: end year
    :param shortnames: short names of variables
    :param longnames: long names of variables
    :param units: units of variables
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f'{shortnames[varname]}_0.5deg.nc'
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
    indir = indir_base[varname]
    ds_ls = []
    years = range(start_yr, end_yr + 1)
    for year in years:
        file_path = os.path.join(indir, f'{varname}_{str(year)}_monthly_0.5deg.nc')
        ds = xr.open_dataset(file_path)
        ds_ls.append(ds)

    merged_ds = xr.concat(ds_ls, dim='time')
    merged_ds = merged_ds.std(dim='time', skipna=True)
    merged_ds[varname].attrs['Unit'] = units[varname]
    merged_ds[varname].attrs['Standard name'] = longnames[varname]
    merged_ds[varname].attrs['Processing method'] = f'Calculated by monthly standard deviation based on CRUJRA V2.2 during {start_yr}-{end_yr}'
    merged_ds = merged_ds.rename({varname: shortnames[varname]})
    merged_ds.to_netcdf(outpath)
    print(f'Saving {outpath}')

    return

