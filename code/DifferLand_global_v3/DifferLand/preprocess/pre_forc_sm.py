# Code for preprocessing soil moisture data
# Resample to 0.5deg*0.5deg and monthly scale


import os
import xarray as xr


def pre_forc_sm_esacci(year, indir_base, outdir_base, ref_path, overwrite):
    ''' Function to preprocess soil moisture data from ESACCI
    
    :param year: year to preprocess
    :param indir_base: input directory
    :param outdir_base: output directory
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f'SM_0_5cm_ESACCI_{int(year)}_monthly_0.5deg.nc'
    outdir = os.path.join(outdir_base, 'SM')
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
    var_name = list(ref_ds.data_vars.keys())[0]
    ref_ds = ref_ds[var_name].isel(time=0)
    indir = os.path.join(indir_base, str(year))
    files = os.listdir(indir)
    files.sort()
    ds_ls = []
    for f in files:
        inpath = os.path.join(indir, f)
        ds = xr.open_dataset(inpath)
        ds = ds['sm']

        # Align the lon and lat dimension to match the reference file
        ds = ds.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')

        # Append the dataset to the list
        ds_ls.append(ds)

    # Merge all days into one dataset and resample to monthly scale
    ds = xr.concat(ds_ls, dim='time')
    ds = ds.resample(time="1MS").mean(skipna=True)
    ds = ds.rename('SM')
    ds.attrs = {
        "Standard name": "Surface volumetric soil moisture",
        "Unit": "m3 m-3",
        "Processing method": "Monthly averages of daily mean for soil moisture based on ESACCI"
    }
    ds.sortby(['lat', 'lon', 'time'])

    # Save the merged dataset to the output folder
    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return

