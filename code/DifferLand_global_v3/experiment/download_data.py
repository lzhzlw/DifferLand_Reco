# Code for downloading CO2 of CAMS dataset, and Soilgrids v2 data
# Soilgrids v2 data can also be downloaded through GEE, see in download_predictor_obs_soilgrids_GEE.txt
# Canopy height data is downloaded from GEE, see in download_predictor_canopy_height_GEE.txt
# Other data can be downloaded from the corresponding data source website according to the user guide
# Köppen-Geiger global 1‑km climate classification map, https://www.gloh2o.org/koppen/
# CRU JRA v2.2 gridded meteorological dataset, https://data-search.nerc.ac.uk/geonetwork/srv/api/records/4bdf41fc10af4caaa489b14745c665a6
# Soil type data, https://data.isric.org/geonetwork/srv/api/records/5c301e97-9662-4f77-aa2d-48facd3c9e14
# Other soil related data, https://www.isric.org/explore/soilgrids
# Terrestrial ecoregions of the world, https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world
# Global canopy height map, https://www.research-collection.ethz.ch/entities/researchdata/2ecca65c-bc7b-4e72-9b62-534d606a062e. 
# Elevation data, https://www.ncei.noaa.gov/products/etopo-global-relief-model
# Global 1 km forest age dataset, https://www.bgc-jena.mpg.de/geodb/projects/FileDetails.php
# CAMS global inversion-optimised greenhouse gas fluxes and concentrations, https://ads.atmosphere.copernicus.eu/datasets/cams-global-greenhouse-gas-inversion?tab=overview 
# Global MODIS and FLUXNET-derived daily GPP v2 is publicly available in https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1835 (Accessed on 12/09/2024). 
# GLOBMAP global LAI since 1981 v3, https://zenodo.org/records/12698637
# Data-driven upscale product of Reco ver.2020.2, https://www.nies.go.jp/doi/10.17595/20200227.001-e.html
# Derived optimal linear combination ET (DOLCE) v3.0, https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f3535_5828_5475_8492
# Global gridded 1-km soil and soil Rh, https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1928
# Global map of specific leaf area, https://github.com/abhirupdatta/global_maps_of_plant_traits/tree/master/sla


import os
import numpy as np
import multiprocessing
from multiprocessing import cpu_count
import sys
sys.path.insert(1, '../')
from DifferLand.download.download_forc_ts_sm import download_forc_ts_sm_era5l
from DifferLand.download.download_forc_co2 import download_forc_co2_cams
from DifferLand.download.download_pred_obs_soil import download_pred_obs_soilgrids


def main():

    overwrite = False # If True, overwrite existing files
    start_year = 2001
    end_year = 2018

    # Download soilgrids data
    # Note that these data may need to run multiple times until all data are downloaded
    # =============================================================================
    print(f'\nDownloading Soilgrids data...')
    outdir_base = f'/data4/dataset/belowground_topo/SoilGrids250m_v2.0/soilgrids_v2_0.5deg'
    os.makedirs(outdir_base, exist_ok=True)
    lon_size = 720
    lat_size = 360
    extent = [-180, 180, -90, 90] # [west, east, south, north]
    crs='urn:ogc:def:crs:EPSG::4326'
    varname_ls = ['ocd', 'soc', 'silt', 'sand', 'phh2o', 'clay', 'nitrogen', 'cfvo', 'cec', 'bdod', 'ocs', 'wrb']
    depth_ls = ['5-15', '15-30', '0-5', '30-60', '60-100','100-200', '0-30']
    num_processes = np.minimum(66, cpu_count())
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(download_pred_obs_soilgrids, 
                     [(varname, depth, outdir_base, lon_size, lat_size, extent, crs, overwrite) 
                      for varname in varname_ls for depth in depth_ls])
    print(ccc)

    # Download soil temperature and soil moisture of ERA5_Land dataset
    # =============================================================================
    print(f'\nDownloading soil temperature and soil moisture of ERA5_Land dataset...')
    start_year = 2001
    end_year = 2018
    var_names = ["soil_temperature", "volumetric_soil_water"]
    outpaths = [rf'/data1/fyliu/ERA5_Land/soil_temperature/Ts_monthly_{start_year}_{end_year}.nc',
                rf'/data1/fyliu/ERA5_Land/soil_moisture/SM_monthly_{start_year}_{end_year}.nc']
    for var_name, outpath in zip(var_names, outpaths):
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if os.path.exists(outpath):
            if overwrite:
                print(f'Overwriting {outpath}...')
                download_forc_ts_sm_era5l(var_name, start_year, end_year, outpath)
            else:
                print(f'{outpath} already exists, skip downloading')
        else:
            print(f'{outpath} does not exist, downloading...')
            download_forc_ts_sm_era5l(var_name, start_year, end_year, outpath)

    # Download co2 cams data
    # =============================================================================
    print(f'\nDownloading CAMS CO2 data...')
    outpath = r'/data4/dataset/atmosphere/co2/CAMS_3hours/CO2_3_hours.netcdf.zip'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    if os.path.exists(outpath):
        if overwrite:
            print(f'Overwriting {outpath}...')
            download_forc_co2_cams(start_year, end_year, outpath)
        else:
            print(f'{outpath} already exists, skip downloading')
    else:
        print(f'{outpath} does not exist, downloading...')
        download_forc_co2_cams(start_year, end_year, outpath)

    return


if __name__ == '__main__':

    main()

