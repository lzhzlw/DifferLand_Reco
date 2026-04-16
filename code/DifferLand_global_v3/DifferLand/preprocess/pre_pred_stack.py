# Code for stacking all predictors into a .nc file


import xarray as xr
import os


def pre_pred_stack(indir_soil, indir_others, outdir, depth, overwrite):
    ''' Function to preprocess predictor data into a single .nc file

    :param indir_soil: input directory for soil predictors
    :param indir_others: input directory for other predictors
    :param outdir: output directory
    :param depth: depth of soil in cm
    :param overwrite: whether to overwrite existing files

    :return None

    '''

    outname = f'predictor_stack_0_{depth}cm_0.5deg.nc'
    outpath = os.path.join(outdir, outname)
    if os.path.exists(outpath):
        if overwrite:
            print(f'Overwriting {outpath}...')
        else:
            print(f'{outpath} already exists, skip preprocessing')

            return
    else:
        print(f'{outpath} does not exist, preprocessing...')

    # Get file paths
    file_paths = [os.path.join(indir_others, f) for f in os.listdir(indir_others) if f.endswith('.nc')]
    depth_str = f'_0_{depth}cm_'
    file_paths += [os.path.join(indir_soil, f) for f in os.listdir(indir_soil) if (f.endswith('.nc')) and (depth_str in f)]
    # Load data
    ds_ls = []
    for file_path in file_paths:
        # Open the dataset
        ds = xr.open_dataset(file_path)
        ds_ls.append(ds)
    # Merge datasets after renaming
    merged_ds = xr.merge(ds_ls)
    # Create a mask for valid points where all variables are not NaN
    valid_mask = ~merged_ds.isnull().to_array().any(dim="variable")
    # Apply the mask to all variables, keeping original shape
    merged_ds = merged_ds.where(valid_mask)
    if "time" in merged_ds.variables:
        merged_ds = merged_ds.drop_vars("time")
    # Save result
    merged_ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return

