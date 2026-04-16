# Code for calculating soil matric potential from soil moisture and soil texture


import os
import numpy as np
import xarray as xr
from soiltexture import getTexture


# Fixed parameters for calculating soil matric potential (SMP) (Ghezzehei et al. 2019, Biogeosciences)
USDA_mapping = {
    'clay': 1,
    'clay loam': 2,
    'loam': 3,
    'loamy sand': 4,
    'sand': 5,
    'sandy clay': 6,
    'sandy clay loam': 7,
    'sandy loam': 8,
    'silt': 9,
    'silt clay': 10,
    'silt clay loam': 11,
    'silt loam': 12
}

SMP_qr = np.array([0.098, 0.079, 0.061, 0.049, 0.053, 0.117, 0.063, 0.039, 0.05, 0.111, 0.09, 0.065])  # Residual water content
SMP_qs = np.array([0.459, 0.442, 0.399, 0.39, 0.375, 0.385, 0.384, 0.387, 0.489, 0.481, 0.482, 0.439])  # Saturated water content
SMP_alpha = 10 ** np.array([-1.825, -1.801, -1.954, -1.459, -1.453, -1.476, -1.676, -1.574, -2.182, -1.79, -2.076, -2.296])
SMP_n = 10 ** np.array([0.098, 0.151, 0.168, 0.242, 0.502, 0.082, 0.124, 0.161, 0.225, 0.121, 0.182, 0.221])


def cal_matric_potential(sand, clay, soil_moist):
    '''
    Function to calculate matric potential, kpa

    :param sand: sand content, %
    :param clay: clay content, %
    :param soil_moist: soil moisture, m3 m-3

    :return: matpot, soil matric potential, kPa

    '''

    try:
        # Get soil texture based on USDA
        soil_texture = getTexture(sand=sand, clay=clay, classification='USDA')
        # Get fixed patameters
        idx = USDA_mapping[soil_texture] - 1
        qr = SMP_qr[idx]
        qs = SMP_qs[idx]
        alpha = SMP_alpha[idx]
        n = SMP_n[idx]
        m = 1 - 1/n
        # Calculate matric potential
        matpot = -1/alpha * (((qs - qr)/(soil_moist - qr)) ** (1/m) - 1) ** (1/n)
        matpot = matpot * 0.098 # convert from cm H2O to kPa
        matpot = np.maximum(matpot, -1500.0)
    except:
        matpot = np.full_like(soil_moist, np.nan)

    return matpot


def pre_forc_smp(year, indir_sm, outdir_base, sand_ds, clay_ds, depth, data_source, overwrite):
    '''
    Function to calculate matric potential, kpa

    :param year: year for which to process data
    :param indir_sm: input directory for soil moisture
    :param outdir_base: output directory
    :param sand_ds: xarray dataset for sand content, %
    :param clay_ds: xarray dataset for clay content, %
    :param depth: soil depth in cm
    :param data_source: data source
    :param overwrite: whether to overwrite existing files

    :return None

    '''

    outname = f'SMP_0_{int(depth)}cm_{data_source}_{int(year)}_monthly_0.5deg.nc'
    outdir = os.path.join(outdir_base, 'SMP')
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
    inname_sm = outname.replace('SMP_', 'SM_')
    inpath_sm = os.path.join(indir_sm, inname_sm)
    sm_ds = xr.open_dataset(inpath_sm)['SM']

    # Calculate soil matric potential
    smp = xr.apply_ufunc(cal_matric_potential, 
                         sand_ds, clay_ds, sm_ds,
                         input_core_dims=[[], [], ['time']],
                         output_core_dims=[['time']],
                         vectorize=True,
                         dask='parallelized')
    smp = smp.rename('SMP')
    smp.attrs = {
        "Standard name": "Surface soil matric potential",
        "Unit": "kPa",
        "Processing method": f"Calculated from soil moisture ({data_source}) and soil texture based on van Genuchten model"
    }

    # Save the dataset
    smp.to_netcdf(outpath)
    print(f'Saving {outpath}')

    return

