# Code for preprocessing LAI of GlobMap dataset
# 1. Convert from .tif to .nc
# 2. Multiply scale by 0.01
# 3. Resample to 0.5deg*0.5deg grid
# 4. Aggregate to monthly scale


import rasterio
import xarray as xr
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator


def pre_obs_lai_globmap(year, indir, outdir_base, ref_path, overwrite):
    ''' Function to preprocess LAI of GlobMap dataset
    
    :param year: year to preprocess
    :param indir: input directory
    :param outdir_base: output directory
    :param ref_path: reference for resampling, 0.5*0.5 deg nc file
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    outname = f"LAI_{year}_monthly_0.5deg.nc"
    outdir = os.path.join(outdir_base, "LAI")
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
    ref_lat = ref_ds['lat']
    ref_lon = ref_ds['lon']

    # Prepare a dictionary to store the data for each month
    monthly_data = {i: [] for i in range(1, 13)}
    # Find all TIFF files for the year
    tif_files = sorted(
        [f for f in os.listdir(indir) if f.startswith(f"GlobMapLAIV3.A{year}") and f.endswith(".tif")]
    )

    # Process each TIFF file for the year
    for tif_file in tif_files:
        tiff_path = os.path.join(indir, tif_file)

        # Extract DOY from filename (e.g., A2001001 -> DOY 001)
        doy = int(tif_file.split(".")[1][5:8])  # Extract DOY from filename
        current_date = datetime(year, 1, 1) + timedelta(days=doy - 1)  # Convert DOY to datetime

        # Read TIFF using rasterio
        with rasterio.open(tiff_path) as src:
            data = src.read(1)  # Read the first band
            transform = src.transform
            height, width = src.height, src.width
            crs = src.crs

        # Correctly calculate latitude and longitude as 2D arrays
        x = np.arange(width)
        y = np.arange(height)
        lon, lat = np.meshgrid(
            transform.c + transform.a * x,  # Longitude values
            transform.f + transform.e * y  # Latitude values
        )

        # Rescale the data (divide by 100)
        data = data * 0.01
        data = np.where(data>=0, data, np.nan)

        # Use RegularGridInterpolator for nearest-neighbor interpolation
        grid_lat = lat[:, 0]  # shape: (height,)
        grid_lon = lon[0, :]  # shape: (width,)
        interpolator = RegularGridInterpolator(
            (grid_lat, grid_lon), data,
            method='nearest', bounds_error=False, fill_value=None
        )

        ref_lon2d, ref_lat2d = np.meshgrid(ref_lon, ref_lat)
        pts = np.array([ref_lat2d.ravel(), ref_lon2d.ravel()]).T
        resampled_data = interpolator(pts).reshape(ref_lat2d.shape)

        # Append the data for the corresponding month
        month = current_date.month
        monthly_data[month].append(resampled_data)

    # Calculate the monthly mean for the year
    monthly_means = {}
    for month, data_list in monthly_data.items():
        if data_list:  # If there is data for the month
            # Stack the data for the month and calculate the mean
            monthly_means[month] = np.nanmean(np.stack(data_list), axis=0)

    # Add time dimension using the 1st day of each month as representative date
    months = sorted(monthly_means.keys())
    times = pd.to_datetime([f"{year}-{month:02d}-01" for month in months])
    lai_data = np.stack([monthly_means[month] for month in months], axis=0)

    # Create xarray Dataset for the year
    ds = xr.Dataset(
        {
            "LAI": (["time", "lat", "lon"], lai_data)
        },
        coords={
            "time": times,
            "lat": ref_lat,
            "lon": ref_lon,
        },
        attrs={"crs": str(crs)}
    )

    # Add attributes to lai, lat and lon
    ds["LAI"].attrs = {
        "Standard name": "Leaf area index",
        "Unit": "m2 m-2",
        "Processing method": "Monthly averages of daily mean LAI based on GlobMap V3",
    }

    # Save as NetCDF for the year
    ds.to_netcdf(outpath)
    print(f"Saving {outpath}")

    return

