# Code for preproceding data, forcings, observations, and predictors


import argparse
import os
import numpy as np
import xarray as xr
import multiprocessing
from multiprocessing import cpu_count
import time
import itertools
import shutil
import sys
sys.path.insert(1, '../')
from DifferLand.preprocess.pre_forc_co2 import pre_forc_co2_cams
from DifferLand.preprocess.pre_forc_meteo import pre_forc_meteo_crujra
from DifferLand.preprocess.pre_forc_vpd import pre_forc_vpd_crujra
from DifferLand.preprocess.pre_forc_albedo_emiss import pre_forc_albedo_emiss_modis
from DifferLand.preprocess.pre_forc_orchidee_cable import pre_forc_orchidee_cable
from DifferLand.preprocess.pre_forc_ts_sm import pre_forc_ts_sm_era5l
from DifferLand.preprocess.pre_forc_smp import pre_forc_smp
from DifferLand.preprocess.pre_forc_sm import pre_forc_sm_esacci
from DifferLand.preprocess.pre_obs_et import pre_obs_et_dolce
from DifferLand.preprocess.pre_obs_gpp import pre_obs_gpp_fluxsat
from DifferLand.preprocess.pre_obs_lai import pre_obs_lai_globmap
from DifferLand.preprocess.pre_obs_reco import pre_obs_reco_zeng2020
from DifferLand.preprocess.pre_obs_lma import pre_obs_lma_butler2017
from DifferLand.preprocess.pre_obs_point2grid import pre_obs_point2grid
from DifferLand.preprocess.pre_obs_maoc import pre_obs_maoc_georgiou2022, pre_obs_maoc_point
from DifferLand.preprocess.pre_pred_obs_soil import pre_pred_obs_soilgrids
from DifferLand.preprocess.pre_pred_seasonality import pre_pred_seasonality_crujra
from DifferLand.preprocess.pre_pred_forest_age import pre_pred_age_bgi
from DifferLand.preprocess.pre_pred_slope import pre_pred_slope
from DifferLand.preprocess.pre_pred_stack import pre_pred_stack
from DifferLand.preprocess.pre_pred_obs_tif import pre_pred_obs_tif
from DifferLand.preprocess.pre_forc_obs_multiyr_ave import pre_forc_obs_multiyr_ave


# Variable name includes:
# 'all', deal with all variables
# 'tif', deal with snapshot predictors or observations in .tif format
# including Rh, Elevation, Climate zone, Biome zone, Soil type, and Canopy height
# 'seasonality', deal with seasonality of air temperature and precipitation
# 'meteo', deal with CRUJRA meteorological data
# including Ta_max, Ta_min, Ta_mean, Prcp, SSRD, ALDS, Pres, Wind
# 'soil', deal with Soilgirds data
# including SOC, BD, CFVO, CEC, Clay, Nitrogen, pH, Sand, Silt
# 'emiss' or 'albedo', deal with MODIS emissivity or albedo
# 'orchidee' or 'cable' (default), deal with soil temperature, matric potential and soil moisture data from orchidee/cable
# 'point', deal with point data, resample to grid scale
# 'ave', deal with multiyear averages of GPP, LAI, Reco, ET, air temperature, and precipitation
# 'stack', deal with all predictors, stacking them into a nc.file
# 'copy', copy BD, Clay, Silt, Ph, Canopy height to forcing foder, ave_Ta_mean and ave_Prcp to predictor folder
# other single variables
# including 'co2', 'vpd', 'ts', 'et', 'gpp', 'lai', 'reco', 'lma', 'sla', 'age', 'slope', 'maoc', 'sm', 'smp'

description = (
    "Preprocess forcings, observations, and predictors for DifferLand model.\n"
    "all: all variables\n"
    "tif: snapshot predictors or observations in .tif format, \n"
    "     including Rh, Elevation, Climate zone, Biome zone, Soil type, and Canopy height\n"
    "seasonality: seasonality of air temperature and precipitation\n"
    "meteo: CRUJRA meteorological data, including Ta_max, Ta_min, Ta_mean, Prcp, SSRD, ALDS, Pres, Wind\n"
    "soil: Soilgirds data\n"
    "emiss: MODIS emissivity\n"
    "albedo: MODIS albedo\n"
    "orchidee: soil temperature, matric potential and soil moisture data from orchidee\n"
    "cable: soil temperature, matric potential and soil moisture data from cable\n"
    "point: point data, resample to grid scale, including SOC, MAOC, POC\n"
    "ave: multiyear averages of GPP, LAI, Reco, ET, air temperature, and precipitation\n"
    "stack: all predictors, stacking them into a nc.file\n"
    "copy: copy BD, Clay, Silt, Ph, Canopy height to forcing foder, ave_Ta_mean and ave_Prcp to predictor folder\n"
    "co2: co2\n"
    "vpd: vapour pressure deficit\n"
    "ts_era5l: soil temperature from ERA5-Land\n"
    "et: evapotranspiration\n"
    "gpp: gross primary productivity\n"
    "lai: leaf area index\n"
    "reco: ecosystem respiration\n"
    "lma: leaf mass per area\n"
    "sla: specific leaf area\n"
    "age: forest age\n"
    "slope: slope\n"
    "maoc: mineral associated organic carbon\n"
    "sm_esacci: soil moisture from ESACCI\n"
    "sm_era5l: soil moisture from ERA5-Land\n"
    "smp_esacci: soil matric potential from ESACCI\n"
    "smp_era5l: soil matric potential from ERA5-Land\n"
)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog="DifferLand Preprocessing", 
                                     description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--var", metavar="VARIABLE", 
                        help="Enter the variable name to preprocess", required=True)
    
    args = parser.parse_args()
    
    return args


def main(variable, overwrite):
    ''' Function to preprocess data, forcings, observations, and predictors
    
    :param variable: variable name, 'all' refers to preprocess all variables
    :param overwrite: whether to overwrite existing files
    
    :return None
    
    '''

    t1 = time.time()
    ref_path =  r'/data1/fyliu/DifferLand_forcing/forcing_monthly_05deg/Ta_max/Ta_max_2001_monthly_0.5deg.nc'
    outdir_base = r'/data1/fyliu/DifferLand_forcing'
    outdir_obs = os.path.join(outdir_base, 'observation_monthly_05deg')
    outdir_pred = os.path.join(outdir_base, 'predictor_05deg')
    os.makedirs(outdir_pred, exist_ok=True)
    outdir_pred_soil = os.path.join(outdir_pred, 'soil')
    os.makedirs(outdir_pred_soil, exist_ok=True)
    outdir_pred_stack = os.path.join(outdir_base, 'predictor_stack_05deg')
    os.makedirs(outdir_pred_stack, exist_ok=True)
    outdir_forc = os.path.join(outdir_base, 'forcing_monthly_05deg')
    outdir_weight = os.path.join(outdir_base, 'weight_05deg')
    os.makedirs(outdir_weight, exist_ok=True)
    start_yr = 2001
    end_yr = 2018
    years = np.arange(start_yr, end_yr + 1)
    num_processes = np.minimum(100, cpu_count())
    soil_depth = 5 # cm
    # Data path
    indir_co2 = r'/data4/dataset/atmosphere/co2/CAMS_3hours/' # co2
    indir_ssdr = r'/data4/share/CRUJRA/v2.2/Radiation-fields' # ssdr
    indir_other_meteo = r'/data4/share/CRUJRA/v2.2/' # other meteorology
    indir_albedo_emiss = r'/data4/dataset/aboveground/albedo_emissivity_MODIS/' # albedo & emiss
    inpath_orchidee = r'/data1/fyliu/j_DifferLand_ET_SOC/mes-c/obs/hwsd1990/orchidee/ORCHIDEE_ave_1990_2020_time_varing_HWSD_soil_layer_compress.nc' # orchidee
    inpath_cable = r'/data1/fyliu/j_DifferLand_ET_SOC/mes-c/obs/hwsd1990/cable/CABLE_Time_varying_HWSD_soil_layer.nc' # cable
    inpath_ts_era5l = r'/data4/dataset/atmosphere/ERA5_land/soil_temperature/Ts_monthly_2001_2018.nc' # soil temperature
    indir_sm_esacci = r'/data4/dataset/et_drought_hydrology/soil_moisture/esacci_soil_moisture_COMBINED_v09.1/' # soil moisture
    inpath_sm_era5l = r'/data4/dataset/atmosphere/ERA5_land/soil_moisture/SM_monthly_2001_2018.nc'
    indir_smp = os.path.join(outdir_obs, 'SM') # calculate soil matric potential based on soil moisture and soil texture
    inpath_sand = os.path.join(outdir_pred_soil, f'Sand_0_{soil_depth}cm_0.5deg.nc')
    inpath_clay = os.path.join(outdir_pred_soil, f'Clay_0_{soil_depth}cm_0.5deg.nc')
    indir_et = r'/data4/dataset/et_drought_hydrology/ET_DOLCE_v3/' # et
    indir_gpp = r'/data4/dataset/aboveground/gpp_FluxSat_daily/' # gpp
    indir_lai = r'/data4/dataset/aboveground/lai_GlobMap_8day/2001_2020/' # lai
    indir_reco = r'/data4/dataset/aboveground/gpp_npp_reco_ZengFluxnet/RECO/' # reco
    inpath_sla = r'/data4/dataset/aboveground/SLA/spat_1_pft_sla_large.csv' # sla & lma
    inpath_maoc = r'/data4/dataset/belowground_topo/MAOC_MAOC_max/global_MOC.nc' # maoc
    inpath_maoc_point_yyhuang = r'/data1/fyliu/j_DifferLand_ET_SOC/data/MAOC_merge_2026_v1.csv' # soc, maoc, poc
    outpath_maoc_point_yyhuang = rf'/data1/fyliu/j_DifferLand_ET_SOC/data/SOC_MAOC_POC_0_{soil_depth}cm_gC_per_m2_yyhuang.csv'
    inpath_maoc_point_wosis = r'/data4/dataset/belowground_topo/WoSIS_2023_December/wosis_202312_orgc.tsv' # soc
    outpath_maoc_point_wosis = rf'/data1/fyliu/j_DifferLand_ET_SOC/data/SOC_MAOC_POC_0_{soil_depth}cm_gC_per_m2_wosis.csv'
    inpath_pred_stack = os.path.join(outdir_pred_stack, f'predictor_stack_0_{soil_depth}cm_0.5deg.nc') # get soil properties to convert unit
    indir_soil = f'/data4/dataset/belowground_topo/SoilGrids250m_v2.0/soilgrids_v2_0.5deg' # soilgrids
    inpath_rh = r'/data4/dataset/belowground_topo/SRDB_V5_GRID/data/soil_Rh_mean.tif' # rh
    inpath_ch = r'/data4/dataset/aboveground/canopy_height/ETH_Lang/Canopy_height_0_5_degree.tif' # canopy height
    inpath_ele = r'/data4/dataset/belowground_topo/DEM/ETOPO_2022_v1_60s_N90W180_surface.tif' # elevation
    inpath_soil_type = r'/data4/dataset/belowground_topo/SoilGrids250m_v1.0/TAXNWRB_250m_ll.tif' # soil type
    inpath_biome = r'/data4/dataset/lucc_boundary_map/biome_ecoregion/Biome/biome.tif' # biome type
    inpath_climate = r'/data4/dataset/lucc_boundary_map/Koppen_Geiger/koppen_geiger/1991_2020/koppen_geiger_0p5.tif' # climate type
    inpath_age = r'/data4/dataset/lucc_boundary_map/forest_age_BGI/BGIForestAgeMP.nc'  # forest age
    
    # Preprocessing CO2 CAMS data
    # =============================================================================
    if variable in ['all', 'co2']:
        print(f'\nPreprocessing CO2 data from CAMS...')
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_forc_co2_cams, [(year, indir_co2, outdir_forc, ref_path, overwrite) for year in years])
    
    # Preprocessing CRUJRA meteorological data
    # =============================================================================
    if variable in ['all', 'meteo']:
        print(f'\nPreprocessing meteorological data from CRUJRA...')
        methods = {
            "tmax": "max",
            "tmin": "min",
            "pre": "sum",
            "tswrf": "sum",
            "tmp": "mean",
            "dlwrf": "sum",
            "pres": "mean",
            "wind": "mean"
        }
        longnames = {
            "tmax": "Daily maximum air temperature",
            "tmin": "Daily minimum air temperature",
            "pre": "Daily precipitation",
            "tswrf": "Daily solar shortwave downward radiation",
            "tmp": "Daily mean air temperature",
            "dlwrf": "Daily atmospheric longwave downward radiation",
            "pres": "Daily air pressure",
            "wind": "Daily wind speed"
        }
        shortnames = {
            "tmax": "Ta_max",
            "tmin": "Ta_min",
            "pre": "Prcp",
            "tswrf": "SSDR",
            "tmp": "Ta_mean",
            "dlwrf": "ALDR",
            "pres": "Pres",
            "wind": "Wind"
        }
        units = {
            "tmax": "deg C",
            "tmin": "deg C",
            "pre": "mm day-1",
            "tswrf": "W m-2",
            "tmp": "deg C",
            "dlwrf": "W m-2",
            "pres": "kPa",
            "wind": "m s-1"
        }
        varnames = ['dlwrf'] # ['tmax', 'tmin', 'tmp', 'pre', 'tswrf', 'dlwrf', 'pres', 'wind']
        param_combinations = list(itertools.product(varnames, years, [indir_ssdr], [indir_other_meteo], 
                                                    [outdir_forc], [methods], [longnames], 
                                                    [shortnames], [units], [overwrite]))
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_forc_meteo_crujra, param_combinations)
    
    # Preprocessing VPD, calculation based on air temperature, pressure, and specific humidity from crujra
    # =============================================================================
    if variable in ['all', 'vpd']:
        print(f'\nPreprocessing VPD data from CRUJRA...')
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_forc_vpd_crujra, [(year, indir_other_meteo, outdir_forc, overwrite) for year in years])
    
    # Preprocessing Albedo and Surface Emissivity MODIS data
    # =============================================================================
    if variable in ['all', 'albedo', 'emiss']:
        print(f'\nPreprocessing Albedo and Emiss data from MODIS...')
        longnames = {
            "Albedo": "Albedo",
            "Emiss": "Surface emissivity"
        }
        units = {
            "Albedo": "unitless",
            "Emiss": "unitless"
        }
        data_sources = {
            "Albedo": "MCD43A3006",
            "Emiss": "MOD11A2006"
        }
        varnames = ['Albedo', 'Emiss']
        param_combinations = list(itertools.product(varnames, years, [indir_albedo_emiss], 
                                                    [outdir_forc], [ref_path], [longnames], 
                                                    [units], [data_sources], [overwrite]))
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_forc_albedo_emiss_modis, param_combinations)
    
    # Preprocessing Orchidee/CABLE data
    # =============================================================================
    if variable in ['all', 'orchidee', 'cable']:
        source = 'Orchidee' if variable == 'orchidee' else 'Cable'
        print(f'\nPreprocessing Ts, SMP, and SM data from {source}...')
        th_value = -1500
        shortnames = {
            'SoilTemp': 'Ts',
            'water_potential': 'SMP',
            'SoilMoist': 'SM'
        }
        longnames = {
            'SoilTemp': 'Soil temperature',
            'water_potential': 'Soil matric potential',
            'SoilMoist': 'Soil moisture'
        }
        units = {
            'SoilTemp': 'degC',
            'water_potential': 'kPa',
            'SoilMoist': 'm3 m-3'
        }
        data_sources = {
            'SoilTemp': source,
            'water_potential': source,
            'SoilMoist': source
        }
        varnames = ['SoilTemp', 'water_potential', 'SoilMoist']
        outdirs = [outdir_forc, outdir_forc, outdir_forc]
        inpath = inpath_orchidee if variable == 'orchidee' else inpath_cable
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_forc_orchidee_cable, 
                         ([varname, inpath, outdir, ref_path, shortnames, longnames, 
                           units, data_sources, th_value, soil_depth, overwrite] for (varname, outdir) in zip(varnames, outdirs)))
    
    # Preprocessing soil temperature and soil moisture data from ERA5-Land
    # =============================================================================
    if variable in ['all', 'ts_era5l', 'sm_era5l']:
        print(f'\nPreprocessing soil temperature and soil moisture data from ERA5-Land...')
        varnames = ['Ts', 'SM']
        inpath = {
            'Ts': inpath_ts_era5l,
            'SM': inpath_sm_era5l
        }
        outdir_base = {
            'Ts': outdir_forc,
            'SM': outdir_forc
        }
        old_varnames = {
            'Ts': 'stl',
            'SM': 'swvl'
        }
        longnames = {
            'Ts': 'Soil temperature',
            'SM': 'Soil moisture'
        }
        units = {
            'Ts': 'deg C',
            'SM': 'm3 m-3'
        }

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_forc_ts_sm_era5l, 
                         [(varname, year, inpath, outdir_base, ref_path, old_varnames, longnames, units, soil_depth, overwrite) 
                          for varname in varnames for year in years])
    
    # Preprocessing soil moisture data from ESACCI
    # =============================================================================
    if variable in ['all', 'sm_esacci']:
        print(f'\nPreprocessing soil moisture data from ESACCI...')
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_forc_sm_esacci, 
                         [(year, indir_sm_esacci, outdir_forc, ref_path, overwrite) for year in years])
            
    # Preprocessing ET DOLCE data
    # =============================================================================
    if variable in ['all', 'et']:
        print(f'\nPreprocessing ET data from DOLCE...')
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_obs_et_dolce, [(year, indir_et, outdir_obs, ref_path, overwrite) for year in years])
    
    # Preprocessing GPP FLUXSAT data
    # =============================================================================
    if variable in ['all', 'gpp']:
        print(f'\nPreprocessing GPP data from FluxSat...')
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_obs_gpp_fluxsat, [(year, indir_gpp, outdir_obs, ref_path, overwrite) for year in years])
    
    # Preprocessing LAI GLOBMAP data
    # =============================================================================
    if variable in ['all', 'lai']:
        print(f'\nPreprocessing LAI data from GlobMap...')
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_obs_lai_globmap, [(year, indir_lai, outdir_obs, ref_path, overwrite) for year in years])
    
    # Preprocessing Reco from Zeng et al. 2020
    # =============================================================================
    if variable in ['all', 'reco']:
        print(f'\nPreprocessing Reco data from Zeng et al. 2020...')
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_obs_reco_zeng2020, [(year, indir_reco, outdir_obs, ref_path, overwrite) for year in years])
    
     # Preprocessing multiyear averages of GPP, LAI, Reco, ET, Ta_mean, and Prcp
    # =============================================================================
    if variable in ['all', 'ave']:
        print(f'\nPreprocessing multiyear averages of GPP, LAI, Reco, ET, Ta_mean, and Prcp...')
        varnames = ['GPP', 'LAI', 'Reco', 'ET', 'Ta_mean', 'Prcp']
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_forc_obs_multiyr_ave, 
                         [(varname, outdir_obs, outdir_forc, start_yr, end_yr, overwrite) for varname in varnames])
    
    # Preprocessing LMA and SLA from Butler et al. 2017
    # =============================================================================
    if variable in ['all', 'lma', 'sla']:
        print(f'\nPreprocessing LMA and SLA data from  Butler et al. 2017...')
        pre_obs_lma_butler2017(inpath_sla, outdir_obs, ref_path, overwrite)
    
    # Preprocessing MAOC data from Georgiou et al. 2022
    # =============================================================================
    if variable in ['all', 'maoc']:
        print(f'\nPreprocessing MAOC data from Georgiou et al. 2022...')
        pre_obs_maoc_georgiou2022(inpath_maoc, outdir_obs, ref_path, overwrite)
    
    # Preprocessing point data to grid scale
    # =============================================================================
    if variable in ['all', 'point']:
        print(f'\nPreprocessing point data to grid scale...')
        # Yuanyuan Huang， MAOC, POC, and SOC
        if not os.path.exists(outpath_maoc_point_yyhuang) or overwrite:
            pre_obs_maoc_point(inpath_maoc_point_yyhuang, inpath_pred_stack, outpath_maoc_point_yyhuang, soil_depth, overwrite)
        else:
            print(f'{outpath_maoc_point_yyhuang} already exists, skip preprocessing')
        inpath = {
            'SOC_p2g': outpath_maoc_point_yyhuang,
            'MAOC_p2g': outpath_maoc_point_yyhuang,
            'POC_p2g': outpath_maoc_point_yyhuang
        }
        cols = {
            'SOC_p2g': ['lon', 'lat', 'SOC'],
            'MAOC_p2g': ['lon', 'lat', 'MAOC'],
            'POC_p2g': ['lon', 'lat', 'POC']
        }
        longnames = {
            'SOC_p2g': 'Soil organic carbon',
            'MAOC_p2g': 'Mineral associated organic carbon',
            'POC_p2g': 'Particulate organic carbon'
        }
        units = {
            'SOC_p2g': 'g C m-2',
            'MAOC_p2g': 'g C m-2',
            'POC_p2g': 'g C m-2'
        }
        data_sources = {
            'SOC_p2g': 'yyhuang',
            'MAOC_p2g': 'yyhuang',
            'POC_p2g': 'yyhuang'
        }
        varnames = ['SOC_p2g', 'MAOC_p2g', 'POC_p2g']
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_obs_point2grid, 
                         [(varname, inpath, outdir_obs, outdir_weight, ref_path, cols, longnames, 
                           units, data_sources, soil_depth, overwrite) for varname in varnames])
        # WOSIS, only SOC
        if not os.path.exists(outpath_maoc_point_wosis) or overwrite:
            pre_obs_maoc_point(inpath_maoc_point_wosis, inpath_pred_stack, outpath_maoc_point_wosis, soil_depth, overwrite)
        else:
            print(f'{outpath_maoc_point_wosis} already exists, skip preprocessing')
        
        varname = 'SOC_p2g'
        inpath = {'SOC_p2g': outpath_maoc_point_wosis}
        cols = {'SOC_p2g': ['lon', 'lat', 'SOC']}
        longnames = {'SOC_p2g': 'Soil organic carbon'}
        units = {'SOC_p2g': 'g C m-2'}
        data_sources = {'SOC_p2g': 'wosis'}
        pre_obs_point2grid(varname, inpath, outdir_obs, outdir_weight, ref_path, cols, longnames, units, data_sources, soil_depth, overwrite)
    
    # Preprocessing soil properties from Soilgrids v2
    # =============================================================================
    if variable in ['all', 'soil']:
        print(f'\nPreprocessing SoilGrids v2 data...')
        # Define variable and output names
        longnames = {
            'ocd': 'Soil organic carbon',
            'bdod': 'Bulk density',
            'cfvo': 'Volumetric fraction of coarse fragments',
            'cec': 'Cation exchange capacity',
            'clay': 'Clay content',
            'nitrogen': 'Nitrogen content',
            'phh2o': 'Soil pH',
            'sand': 'Sand content',
            'silt': 'Silt content'
        }
        shortnames = {
            'ocd': 'SOC',
            'bdod': 'BD',
            'cfvo': 'CFVO',
            'cec': 'CEC',
            'clay': 'Clay',
            'nitrogen': 'Nitrogen',
            'phh2o': 'pH',
            'sand': 'Sand',
            'silt': 'Silt'
        }
        scalars = {
            'ocd': 1,
            'bdod': 10,
            'cfvo': 1/1000,
            'cec': 1,
            'clay': 1/1000,
            'nitrogen': 1/100,
            'phh2o': 1,
            'sand': 1/1000,
            'silt': 1/1000
        }
        units = {
            'ocd': 'g C m-2',
            'bdod': 'kg m-3',
            'cfvo': 'cm3 cm-3',
            'cec': 'mmol(c) kg-1',
            'clay': 'g g-1',
            'nitrogen': 'g kg-1',
            'phh2o': 'unitless',
            'sand': 'g g-1',
            'silt': 'g g-1'
        }
        varnames = ['ocd', 'bdod', 'cfvo', 'cec', 'clay', 'nitrogen', 'phh2o', 'sand', 'silt']
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_pred_obs_soilgrids,
                        [(varname, soil_depth, indir_soil, outdir_obs, outdir_pred_soil, ref_path, longnames, shortnames, units, scalars,overwrite) 
                        for varname in varnames])
    
    # Preprocessing soil matric potential based on soil moisture and soil texture
    # =============================================================================
    if variable in ['all', 'smp_esacci', 'smp_era5l']:
        print(f'\nPreprocessing soil matric potential based on soil moisture and soil texture...')
        sand_ds = xr.open_dataset(inpath_sand)['Sand']
        clay_ds = xr.open_dataset(inpath_clay)['Clay']
        data_source = 'ESACCI' if variable == 'smp_esacci' else 'ERA5L'
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_forc_smp, 
                        [(year, indir_smp, outdir_forc, sand_ds, clay_ds, soil_depth, data_source, overwrite) for year in years])
            
    # Preprocessing Rh, Elevation, Climate zone, Biome zone, Soil type, and Canopy height (snapshot in tif format)
    # =============================================================================
    if variable in ['all', 'tif']:
        print(f'\nPreprocessing snapshot predictors or observations in .tif format...')
        inpath = {
            'Rh': inpath_rh,
            'Elevation': inpath_ele,
            'Climate_type': inpath_climate,
            'Biome_type': inpath_biome,
            'Soil_type': inpath_soil_type,
            'Canopy_height': inpath_ch
        }
        longnames = {
            'Rh': 'Heterotrophic respiration',
            'Elevation': 'Elevation',
            'Climate_type': 'Climate type',
            'Biome_type': 'Biome type',
            'Soil_type': 'Soil type',
            'Canopy_height': 'Canopy height'
        }
        units = {
            'Rh': 'g C m-2 day-1',
            'Elevation': 'm',
            'Climate_type': 'unitless',
            'Biome_type': 'unitless',
            'Soil_type': 'unitless',
            'Canopy_height': 'm'
        }
        data_sources = {
            'Rh': 'SRDB v5',
            'Elevation': 'ETOPO 2022 v1',
            'Climate_type': 'Koppen Geiger 1991-2020',
            'Biome_type': 'Olson et al., 2001',
            'Soil_type': 'Soilgrids v1',
            'Canopy_height': 'ETH global Sentinel-2, Lang et al., 2023'
        }
        varnames = ['Rh', 'Elevation', 'Climate_type', 'Biome_type', 'Soil_type', 'Canopy_height']
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(pre_pred_obs_tif, 
                        [(varname, inpath, outdir_obs, outdir_pred, ref_path, longnames, units, data_sources, overwrite) 
                        for varname in varnames])
    
    # Preprocessing seasonality of air temperature and precipitation
    # =============================================================================
    if variable in ['all', 'seasonality']:
        print(f'\nPreprocessing seasonality of Ta_mean and Prcp based on CRUJRA data...')
        indir_base = {
            'Ta_mean': os.path.join(outdir_forc, 'Ta_mean'),
            'Prcp': os.path.join(outdir_forc, 'Prcp')
        }
        shortnames = {
            'Ta_mean': 'Ta_seasonality',
            'Prcp': 'Prcp_seasonality'
        }
        longnames = {
            'Ta_mean': 'Seasonality of air temperature',
            'Prcp': 'Seasonality of precipitation'
        }
        units = {
            'Ta_mean': 'deg C',
            'Prcp': 'mm day-1'
        }
        varnames = ['Ta_mean', 'Prcp']
        if __name__ == "__main__":
            with multiprocessing.Pool(processes=2) as pool:
                pool.starmap(pre_pred_seasonality_crujra, 
                            [(varname, indir_base, outdir_pred, start_yr, end_yr, shortnames, longnames, units, overwrite)
                            for varname in varnames])
        
    # Preprocessing forest age from BGIForestAgeMP
    # =============================================================================
    if variable in ['all', 'age']:
        print(f'\nPreprocessing forest age data from BGIForestAgeMP...')
        pre_pred_age_bgi(inpath_age, outdir_pred, ref_path, overwrite)
    
    # Preprocessing slope from elevation
    # =============================================================================
    if variable in ['all', 'slope']:
        print(f'\nPreprocessing slope data from elevation...')
        pre_pred_slope(inpath_ele, outdir_pred, ref_path, overwrite)
    
    # Stacking all predictors into a singel .nc file
    # =============================================================================
    if variable in ['all', 'stack']:
        print(f'\nStacking all predictors into a .nc file...')
        pre_pred_stack(outdir_pred_soil, outdir_pred, outdir_pred_stack, soil_depth, overwrite)
    
    # Copy BD, Clay, Silt, Ph, Canopy height to forcing foder, ave_Ta_mean and ave_Prcp to predictor folder
    # =============================================================================
    if variable in ['all', 'copy']:
        print(f'\nCopying BD, Clay, Silt, Ph, Canopy height to forcing folder...')
        varname = 'Canopy_height'
        outname = f'{varname}_0.5deg.nc'
        outpath = os.path.join(outdir_pred, outname)
        outdir_copy = os.path.join(outdir_forc, varname)
        os.makedirs(outdir_copy, exist_ok=True)
        outpath_copy = os.path.join(outdir_copy, outname)
        shutil.copy2(outpath, outpath_copy)
        print(f'Copying {outpath} to {outpath_copy}')
        for varname in ['BD', 'Clay', 'Silt', 'pH']:
            outname_ls = os.listdir(outdir_pred_soil)
            outname_ls = [f for f in outname_ls if varname in f and f.endswith('.nc')]
            for outname in outname_ls:
                outpath = os.path.join(outdir_pred_soil, outname)
                outdir_copy = os.path.join(outdir_forc, varname)
                os.makedirs(outdir_copy, exist_ok=True)
                outpath_copy = os.path.join(outdir_copy, outname)
                shutil.copy2(outpath, outpath_copy)
                print(f'Copying {outpath} to {outpath_copy}')
        print(f'Copying ave_Ta_mean and ave_Prcp to predictor folder...')
        for varname in ['ave_Ta_mean', 'ave_Prcp']:
            outname = f'{varname}_0.5deg_{start_yr}_{end_yr}.nc'
            outpath = os.path.join(outdir_forc, varname, outname)
            outpath_copy = os.path.join(outdir_pred, outname)
            shutil.copy2(outpath, outpath_copy)
            print(f'Copying {outpath} to {outpath_copy}')

    t2 = time.time()
    running_time = t2 - t1
    hours = int(running_time // 3600)
    minutes = int((running_time % 3600) // 60)
    seconds = int(running_time % 60)
    print(f"Running time: {hours} h, {minutes} min, {seconds} s")

    return


if __name__ == "__main__":

    overwrite = True # If True, overwrite existing files
    args = parse_args()
    variable = args.var
    main(variable, overwrite)

