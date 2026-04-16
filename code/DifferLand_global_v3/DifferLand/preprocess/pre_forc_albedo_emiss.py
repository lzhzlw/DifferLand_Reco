# Code for preprocessing albedo and surface emissivity from MODIS dataset
# 1. Convert from .tif to .nc
# 2. Scaler, albedo, 1000, emissivity: y = x*0.002+0.49
# 3. Resample to 0.5deg*0.5deg grid


import rasterio
import xarray as xr
import numpy as np
import pandas as pd
import os
import re


def pre_forc_albedo_emiss_modis(varname, year, indir_base, outdir_base, ref_path,
								longnames, units, data_sources, overwrite):
	''' Function to preprocess albedo and emissivity from MODIS dataset

	:param varname: variable to preprocess
	:param year: year to preprocess
	:param indir_base: directory of input data
	:param outdir_base: directory to store the processed data
	:param ref_path: path of reference data, 0.5*0.5 deg nc file
	:param longnames: long names of variables
	:param units: units of variables
	:param data_sources: sources of data
	:param overwrite: whether to overwrite existing files

	:return None

	'''
	
	outdir = os.path.join(outdir_base, varname)
	os.makedirs(outdir, exist_ok=True)
	outname = f'{varname}_{str(year)}_monthly_0.5deg.nc'
	outpath = os.path.join(outdir, outname)
	if os.path.exists(outpath):
		if overwrite:
			print(f'Overwriting {outpath}...')
		else:
			print(f'{outpath} already exists, skip preprocessing')

			return
	else:
		print(f'{outpath} does not exist, preprocessing...')

	# Find all TIFF files for the year
	indir = os.path.join(indir_base, f'{varname}_{data_sources[varname]}')
	tiff_files = os.listdir(indir)
	tiff_files = sorted(tiff_files, key=lambda x: int(re.search(r'(\d+)', x).group()))
	start_idx = (year - 2001) * 12
	tiff_files = tiff_files[start_idx:start_idx+12]

	year_data = []
	# Process each TIFF file for the year
	for tiff_file in tiff_files:
		tiff_path = os.path.join(indir, tiff_file)
		# Read TIFF using rasterio
		with rasterio.open(tiff_path) as src:
			monthly_data = src.read(1)  # Read the first band
			transform = src.transform
			height, width = src.height, src.width
			crs = src.crs

		# Correctly calculate latitude and longitude as 2D arrays
		x = np.arange(width)
		y = np.arange(height)
		lon = transform.c + transform.a * x
		lat = transform.f + transform.e * y

		# Rescale the data (divide by 100)
		if varname == 'Albedo':
			monthly_data = monthly_data / 1000
		elif varname == 'Emiss':
			monthly_data = monthly_data * 0.002 + 0.49
		monthly_data = np.where((monthly_data >= 0) & (monthly_data <= 1), monthly_data, np.nan)
		year_data.append(monthly_data)

	times = pd.to_datetime([f"{year}-{month:02d}-01" for month in range(1, 13)])
	year_data = np.stack(year_data, axis=0)

	# Create xarray Dataset for the year
	ds = xr.Dataset(
		{
			varname: (["time", "lat", "lon"], year_data)
		},
		coords={
			"time": times,
			"lat": lat,
			"lon": lon,
		},
		attrs={"crs": str(crs)}
	)

	# Add attributes
	ds[varname].attrs = {
		"Standard name": longnames[varname],
		"Unit": units[varname],
		"Processing method": f'Monthly averages of {varname} from {data_sources[varname]}',
	}
	
	# Resample
	ref_ds = xr.open_dataset(ref_path)
	ref_lat = ref_ds.lat
	ref_lon = ref_ds.lon
	ds = ds.reindex(lat=ref_lat, lon=ref_lon, method='nearest')

	# Save as NetCDF for the year
	ds.to_netcdf(outpath)
	print(f"Saving {outpath}")

	return

