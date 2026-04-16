# Function to run specific calibration


import re
import jax.numpy as jnp
import time
import logging
import sys
sys.path.insert(1, '../')
from DifferLand.util.calibration_global import parse_args, calibration_global
from DifferLand.model.DALEC990_info import (param_info, pool_info, target_info,
                                            common_info, acm_wue_info, pml2_info,
                                            op_info, cs_info, mesc_info)


# Basic settings
# This part gennerally does not need to change, use default values is ok
# If necessary, you can use command line arguments to change them
# ============================================================================================
# ============================================================================================
# Parameters for optimization
args = parse_args()
VERBOSE = not args.silent  # whether to print information, deault is True, i.e., silence = False
SOC_CONFIG = args.soc_config  # 1: one pool; 2: simplified Century; 3: MES-C (default)
GE_CONFIG = args.gpp_et_config  # 1: ACM and WUE; 2: PML2 (default)
UNTM = args.unseen_time  # 0: all time (default); 1: unseen time (only use half of the temporal data to train the model)
RUN_INDEX = args.run_index  # index for independent calibrations, default = 1
RANDOM_SEED = args.random_seed  # default = 42
BATCH_SIZE = args.batch_size  # batch size for each batch training, default = 400
LEARNING_RATE = args.learning_rate  # learning rate for gradient based optimization, default = 0.0005
ITERATIONS = args.iteration  # iterations for each batch training, default = 1
EPOCHS = args.epoch  # total epochs for gradient based optimization, default = 1000
POINT2GRID = args.point2grid  # whether to use point2grid observations for SOC, SOC1 (POC) and SOC2 (MAOC)
UNCERTAINTY = args.uncertainty  # whether to consider uncertainty of observations, default = False

# Manual setting
# ============================================================================================
# ============================================================================================
# Change this part according to your needs
# Parameters for soil depth, spatial and temporal range
SOIL_DEPTH = 5  # soil depth in cm
REGION = {
        "lat_min": -60,
        "lat_max": 80,
        "lon_min": -180,
        "lon_max": 180
    }
# For test
# REGION = {
#     "lat_min": 40,
#     "lat_max": 50,
#     "lon_min": 45,
#     "lon_max": 55
# }
YEAR_RANGE = [2001, 2001]  # start year and end year (included)
TIME_STEP = 30  # running time step of the model, monthly here, need to be consistent with the time-variant forcings/observations

# Parameters for different data sources of SM and Ts
if RUN_INDEX == 1:
    SM_SOURCE = "ERA5L"
    Ts_SOURCE = "ERA5L"
elif RUN_INDEX == 2:
    SM_SOURCE = "ESACCI"
    Ts_SOURCE = "ERA5L"
elif RUN_INDEX == 3:
    SM_SOURCE = "Cable"
    Ts_SOURCE = "Cable"
else:
    raise ValueError("RUN_INDEX should be 1, 2, or 3")

# Parameters for output dir
OUT_DIR = f"../../../output/soc{SOC_CONFIG}_ge{GE_CONFIG}"
LOG_DIR = f"../../../log/soc{SOC_CONFIG}_ge{GE_CONFIG}"
FIG_DIR = f"../../../fig/soc{SOC_CONFIG}_ge{GE_CONFIG}"

# Parameters for data path
PREDICTOR_PATH = rf"/data1/fyliu/DifferLand_forcing/predictor_stack_05deg/predictor_stack_0_{SOIL_DEPTH}cm_0.5deg.nc"  # predictor
FORCING_DIR = r"/data1/fyliu/DifferLand_forcing/forcing_monthly_05deg/"  # forcing
OBSERVATION_DIR = r"/data1/fyliu/DifferLand_forcing/observation_monthly_05deg/"  # observation
WEIGHT_DIR = r"/data1/fyliu/DifferLand_forcing/weight_05deg/"  # weight

# Observation name list
# Two kind of observations can be used for assimilation
# One is the carbon pools or fluxes, which will be compared with the process-based model outputs, e.g., SOC, GPP, LAI
# The other is the process-based model parameters and initial pools, if having their observations,
# they can also be used for assimilation, e.g., LMA
VAR_OBS_LS = ["ET", "GPP", "LAI", "Reco"]  # obs name with temporal information
INVAR_OBS_LS = ["Rh", f"SOC_0_{SOIL_DEPTH}cm"]  # obs name without temporal information (snapshot)
INIT_PARAMS_LS = ["LMA"]  # these obs name is process-based model parameters
INIT_POOLS_LS = []  # these obs name is process-based model pools, gennerally it is empty
# Specify two soil related forcing variables with different data source (Cable or ESACCI for SM and SMP, ERA5L for Ts)
Ts = f"Ts_0_{SOIL_DEPTH}cm_{Ts_SOURCE}"   # soil temperature, degC
SM = f"SM_0_{SOIL_DEPTH}cm_{SM_SOURCE}"   # soil moisture, m3 m-3
SMP = f"SMP_0_{SOIL_DEPTH}cm_{SM_SOURCE}" # soil matric potential,kPa

# Parameters for the Neural Network module
NUM_HIDDEN = 32  # numbers of neurons
NUM_LAYER = 3  # numbers of hidden layers

# Parameters for visualization
VISUALIZATION = True # whether to plot the results
MASK = 'test'  # Plot results with testing ('test'), training ('train') or all (None) samples
METRIC = 'R2'  # R2, r or nnse, calculate performance based on which metric

# Manual setting end here
# ============================================================================================
# ============================================================================================

VAR_OBS_LS_COPY = [var.split('_')[0] for var in VAR_OBS_LS]
INVAR_OBS_LS_COPY = [var.split('_')[0] for var in INVAR_OBS_LS]
if 'POC' in INVAR_OBS_LS_COPY:
    INVAR_OBS_LS_COPY[INVAR_OBS_LS_COPY.index('POC')] = 'SOC1'
if 'MAOC' in INVAR_OBS_LS_COPY:
    INVAR_OBS_LS_COPY[INVAR_OBS_LS_COPY.index('MAOC')] = 'SOC2'
VARNAME_LS = [VAR_OBS_LS_COPY, INVAR_OBS_LS_COPY, INIT_PARAMS_LS, INIT_POOLS_LS]

# Get their idx for comparison with the model's outputs or parameters
VAR_OBS_IND = jnp.array([getattr(target_info, field) for field in VAR_OBS_LS_COPY])
INVAR_OBS_IND = jnp.array([getattr(target_info, field) for field in INVAR_OBS_LS_COPY])
PARAMS_IND = jnp.array([getattr(param_info, field) for field in INIT_PARAMS_LS])
for idx, idx_value in enumerate(PARAMS_IND):
    if (common_info[3] + acm_wue_info[3]) < idx_value < (common_info[3] + acm_wue_info[3] + pml2_info[3]):
        PARAMS_IND[idx] = PARAMS_IND[idx] - acm_wue_info[3]  # PML2 module, minus ACM & WUE module's parameter number
    if idx_value >= (common_info[3] + acm_wue_info[3] + pml2_info[3]):
        # SOC module
        additional_idx = acm_wue_info[3] if GE_CONFIG == 2 else pml2_info[3]
        if SOC_CONFIG == 1:
            PARAMS_IND[idx] = PARAMS_IND[idx] - additional_idx  # One-pool
        elif SOC_CONFIG == 2:
            PARAMS_IND[idx] = PARAMS_IND[idx] - additional_idx - op_info[3]  # Simplified Century
        elif SOC_CONFIG == 3:
            PARAMS_IND[idx] = PARAMS_IND[idx] - additional_idx - op_info[3] - cs_info[3]  # MES-C
POOLS_IND = jnp.array([getattr(pool_info, field) for field in INIT_POOLS_LS])
for idx, idx_value in enumerate(POOLS_IND):
    if idx_value >= (common_info[3] + acm_wue_info[3] + pml2_info[3]):
        # SOC module
        additional_idx = acm_wue_info[3] if GE_CONFIG == 2 else pml2_info[3]
        if SOC_CONFIG == 1:
            PARAMS_IND[idx] = PARAMS_IND[idx] - additional_idx  # One-pool
        elif SOC_CONFIG == 2:
            PARAMS_IND[idx] = PARAMS_IND[idx] - additional_idx - op_info[3]  # Simplified Century
        elif SOC_CONFIG == 3:
            PARAMS_IND[idx] = PARAMS_IND[idx] - additional_idx - op_info[3] - cs_info[3]  # MES-C


def main():
    t0 = time.time()

    # Get total observation name list
    TOTAL_VAR_OBS_LS = VAR_OBS_LS.copy()
    TOTAL_INVAR_OBS_LS = (INVAR_OBS_LS + INIT_PARAMS_LS + INIT_POOLS_LS).copy()
    if POINT2GRID:
        for var_name in TOTAL_INVAR_OBS_LS:
            if re.search(r'SOC_|POC_|MAOC_', var_name):
                key = var_name.split('_')[0]
                new_key = f'{key}_p2g'
                new_var_name = var_name.replace(key, new_key)
                TOTAL_INVAR_OBS_LS[TOTAL_INVAR_OBS_LS.index(var_name)] = new_var_name

    # Get forcing name list
    INVAR_FORCING_LS = ["ave_Ta_mean", "ave_Prcp", "BD", "Canopy_height"]
    if GE_CONFIG == 1:
        INVAR_FORCING_LS = INVAR_FORCING_LS + ["Lat"]
    if SOC_CONFIG == 3:
        INVAR_FORCING_LS = INVAR_FORCING_LS + ["pH", "Clay", "Silt"]
    VAR_FORCING_LS = ["DOY", "Ta_max", "Ta_min", "Prcp", "SSDR", "CO2", "VPD"]
    if GE_CONFIG == 2:
        VAR_FORCING_LS = VAR_FORCING_LS + ["Ta_mean", "Albedo", "ALDR", "Emiss", "Pres", "Wind"]
    if SOC_CONFIG == 3:
        VAR_FORCING_LS = VAR_FORCING_LS + [SM, Ts, SMP]

    ge_info = acm_wue_info if GE_CONFIG == 1 else pml2_info
    soc_info = op_info if SOC_CONFIG == 1 else cs_info if SOC_CONFIG == 2 else mesc_info
    THETA_POOLS = common_info[2] + ge_info[2] + soc_info[2]  # output initial pools for DALEC2
    THETA_OTHERS = common_info[3] + ge_info[3] + soc_info[3]  # output other params for DALEC2

    OBS_WEIGHTS = jnp.ones(len(TOTAL_VAR_OBS_LS + TOTAL_INVAR_OBS_LS))
    # OBS_WEIGHTS = jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 0])

    calibration_global(VERBOSE, SOC_CONFIG, GE_CONFIG, UNTM, RUN_INDEX, RANDOM_SEED,  # params for optimization
                       OBS_WEIGHTS, BATCH_SIZE, LEARNING_RATE, ITERATIONS, EPOCHS, 
                       POINT2GRID, UNCERTAINTY,
                       VISUALIZATION, MASK, METRIC,  # params for visualization
                       LOG_DIR, OUT_DIR, FIG_DIR,   # params for output dir
                       PREDICTOR_PATH, FORCING_DIR, INVAR_FORCING_LS, VAR_FORCING_LS,  # params for load data
                       OBSERVATION_DIR, TOTAL_INVAR_OBS_LS, TOTAL_VAR_OBS_LS, WEIGHT_DIR,
                       TIME_STEP, REGION, YEAR_RANGE, SOIL_DEPTH,
                       NUM_HIDDEN, NUM_LAYER, THETA_POOLS, THETA_OTHERS,  # params for NN model
                       VARNAME_LS, VAR_OBS_IND, INVAR_OBS_IND, PARAMS_IND, POOLS_IND  # other params
                       )

    t1 = time.time()
    running_time = t1 - t0
    hours = int(running_time // 3600)
    minutes = int((running_time % 3600) // 60)
    seconds = int(running_time % 60)
    logging.info(f"Calibration COMPLETE! Running time: {hours} h, {minutes} min, {seconds} s")
    if VERBOSE:
        print(f"\nCalibration COMPLETE! Running time: {hours} h, {minutes} min, {seconds} s")

    return


if __name__ == "__main__":

    main()

