# Code for preprocessing soc and other soil related predictors
# from tif to nc
# resample to 0.5*0.5deg
# cal weighted averaeges of all soil layers
# convert ocd (hg/m3) to soc (g/m2) through weighted by thickness (cm),
# as well as cfvo (proportion of coarse fragments, cm3/dm3) of all soil layers
# soc = sum(ocd_i * thickness_i * (1-cfvo_i/1000))
# bd (cg/cm3) or cfvo (cm3/dm3): weight_var = sum(var_i * thickness_i)/sum(thickness_i)
# ph: divided by scale 10 first, then covert to concentration, con = 10^(-ph), then cal the weighted concentration
# wieght_con = sum(con_i * thickness_i)/sum(thickness_i), finally return to ph, weight_ph = -log(weight_con)
# cec (mmol(c)/kg), clay (g/kg), nitrogen (cg/kg), sand (g/kg), silt (g/kg):
# weight_var = sum(var_i * thickness_i * bd_i)/sum(thickness_i * bd_i)


import os
import re
import numpy as np
import rasterio
import xarray as xr


def extract_thick_soilgrids(filename):
    ''' Function to extract the thickness of each soil layer from the filename of soilgrids data 

    :param filename: str, both including the path dir or not are ok

    :return thickness: int, thickness in cm
    
    '''
    
    match = re.search(r"(\d+)-(\d+)cm", filename)
    if match:
        top, bottom = map(int, match.groups())
        return top, bottom, bottom - top  # Thickness in cm

    return None


def pre_pred_obs_soilgrids(varname, depth, indir_base, outdir_obs, outdir_pred_soil, 
                           ref_path, longnames, shortnames, units, scalars, overwrite):
    ''' Function to aggregate soilgrids data from six layers to a single layer
    
    :param varname: variable to preprocess
    :param depth: soil depth in cm
    :param indir_base: directory of soilgrids data
    :param outdir_obs: output directory for observations
    :param outdir_pred_soil: output directory for predictors
    :param ref_path: reference path for resampling, 0.5*0.5 deg nc file
    :param longnames: long names of variables
    :param shortnames: short names of variables
    :param units: units of variables
    :param scalars: scalars of variables
    :overwrite: whether to overwrite existing files

    :return None
    
    '''

    # Define output NetCDF ff path
    outname = f'{shortnames[varname]}_0_{int(depth)}cm_0.5deg.nc'
    if varname == 'ocd':
        outpath = os.path.join(outdir_obs, 'SOC', outname)
    else:
        outpath = os.path.join(outdir_pred_soil, outname)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    if os.path.exists(outpath):
        if overwrite:
            print(f'Overwriting {outpath}...')
        else:
            print(f'{outpath} already exists, skip preprocessing')

            return
    else:
        print(f'{outpath} does not exist, preprocessing...')

    # Aggregation
    indir = os.path.join(indir_base, varname)
    indir_bd = os.path.join(indir_base, 'bdod')
    indir_cfvo = os.path.join(indir_base, 'cfvo')
    var_i = None
    weight_i = 1
    ref_ds = xr.open_dataset(ref_path)
    for ff in os.listdir(indir):
        if ff.endswith('.tif'):
            file_path = os.path.join(indir, ff)
            # Get the layer boundary and thickness_i
            _, bottom_i,thickness_i = extract_thick_soilgrids(ff)
            if (bottom_i is None) or (bottom_i > depth):
                print(f"Skipping {ff}, over depth {depth}")
                continue
            print(f"Processing {ff}")

            # Open the .tif ff and process the data
            with rasterio.open(file_path) as src:
                var_data_i = src.read(1)
                var_data_i = np.where(var_data_i>=0, var_data_i, np.nan)
                transform = src.transform
                height, width = src.height, src.width

            if varname == 'ocd':
                # Read cfvo data as part of weight
                cfvo_name = ff.replace(varname, 'cfvo')
                cfvo_path = os.path.join(indir_cfvo, cfvo_name)
                with rasterio.open(cfvo_path) as cfvo_src:
                    cfvo_data_i = cfvo_src.read(1) / 1000
                    cfvo_data_i = np.where(cfvo_data_i>=0, cfvo_data_i, np.nan)
                if var_i is None:
                    var_i = var_data_i * thickness_i * (1 - cfvo_data_i)
                else:
                    var_i += var_data_i * thickness_i * (1 - cfvo_data_i)

            elif varname in ['bdod', 'cfvo']:
                if var_i is None:
                    var_i = var_data_i * thickness_i
                    weight_i = thickness_i
                else:
                    var_i += var_data_i * thickness_i
                    weight_i += thickness_i

            elif varname == 'phh2o':
                var_data_i = var_data_i / 10
                var_data_i = 10 ** (-var_data_i)
                if var_i is None:
                    var_i = var_data_i * thickness_i
                    weight_i = thickness_i
                else:
                    var_i += var_data_i * thickness_i
                    weight_i += thickness_i

            elif varname in ['cec', 'clay', 'nitrogen', 'sand', 'silt']:
                # Read bdod data
                bd_name = ff.replace(varname, 'bdod')
                bd_path = os.path.join(indir_bd, bd_name)
                with rasterio.open(bd_path) as bd_src:
                    bd_data_i = bd_src.read(1)
                    bd_data_i = np.where(bd_data_i>=0, bd_data_i, np.nan)

                if var_i is None:
                    var_i = var_data_i * thickness_i * bd_data_i
                    weight_i = thickness_i * bd_data_i
                else:
                    var_i += var_data_i * thickness_i * bd_data_i
                    weight_i += thickness_i * bd_data_i

            print(f"Finished processing {ff}")
    result = var_i / weight_i
    if varname == 'phh2o':
        result = -np.log10(result)
    result = np.where(result >= 0, result, np.nan)
    result = result * scalars[varname]

    # Convert to .nc
    x = np.arange(width)
    y = np.arange(height)
    lon = transform.c + transform.a * x
    lat = transform.f + transform.e * y
    ds = xr.DataArray(
        result,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        name=shortnames[varname]
    )
    # Set variable attributes
    ds.attrs["Unit"] = units[varname]
    ds.attrs["Standard name"] = longnames[varname]
    ds.attrs["Processing method"] = f"Weighted averages of {shortnames[varname]} over 6 layers based on Soilgrids v2"
    # Resample
    ds = ds.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')
    # Save result
    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return

