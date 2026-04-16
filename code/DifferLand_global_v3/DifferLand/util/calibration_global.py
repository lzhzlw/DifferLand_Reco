# Function for calibration


import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
import optax
from functools import partial
import os
import logging
import sys
import pickle
sys.path.insert(1, '../')
from DifferLand.util.load_data import load_all_data
from DifferLand.model.FCNN import FCNN
from DifferLand.model.DALEC990 import DALEC990
from DifferLand.util.visualization import visualization_global


# Part1, functions for basic process
# ===============================================================================================
# ===============================================================================================
def parse_weights(value):
    try:
        return float(eval(value))
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid weight value: {value}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog="DifferLand Calibration",
                                     description="Run the script to calibrate the physical and neural parameters "
                                                 "in the DifferLand model.")
    parser.add_argument("-soc", "--soc_config", metavar="SOC_IDX",
                        help="Enter the index for SOC module configuration, range from 1-3. "
                             + "(1) One pool; (2) Simplified Century; (3) MES-C", type=int, default=3)
    parser.add_argument("-ge", "--gpp_et_config", metavar="GE_IDX",
                        help="Enter an index for GPP and ET coupling module, range from 1-2, (1) ACM and WUE; (2) pml2", 
                        type=int, default=2)
    parser.add_argument("-r", "--random_seed", metavar="RANDOM_SEED_IDX",
                        help="Enter an index for random seed", type=int, default=42)
    parser.add_argument("-i", "--run_index", metavar="RUN_IDX",
                        help="Enter an index for the run, used to identify independent calibrations", type=int,
                        default=1)
    parser.add_argument("-bs", "--batch_size", metavar="BATCH_SIZE",
                        help="Enter an index for batch size used in batch training", type=int, default=400)
    parser.add_argument("-lr", "--learning_rate", metavar="LEARNING_RATE",
                        help="Enter an index for learning rate used for gradient based optimization", type=float,
                        default=0.0005)
    parser.add_argument("-ep", "--epoch", metavar="EPOCH",
                        help="Enter an index for total epochs used for gradient based optimization", type=int,
                        default=1000)
    parser.add_argument("-it", "--iteration", metavar="ITERATION",
                        help="Enter an index for total iterations used in each batch training", type=int,
                        default=1)
    parser.add_argument("-untm", "--unseen_time", metavar="UNSEEN_TIME",
                        help="Enter an index for whether predict unseen time", type=int, default=0)
    parser.add_argument("-s", "--silent", help="Silence message printing in the terminal.",
                        action=argparse.BooleanOptionalAction, default=False),
    parser.add_argument("-p2g", "--point2grid", metavar="POINT2GRID", 
                        help="Enter an index for whether to use point2grid observations", type=int, default=0),
    parser.add_argument("-unc", "--uncertainty", metavar="UNCERTAINTY",
                        help="Enter an index for whether to consider uncertainty of observations", type=int, default=0)
    args = parser.parse_args()
    
    return args


def get_soc_type(SOC_CONFIG):
    """Get SOC model type str for the scm_type index"""
    if SOC_CONFIG == 1:
        return "onepool"
    elif SOC_CONFIG == 2:
        return "century"
    elif SOC_CONFIG == 3:
        return "mesc"
    else:
        print("ERROR: SOC model type must be between 1-3:")
        print("1) One pool")
        print("2) Simplified Century")
        print("3) MES-C")
        sys.exit(1)


def get_ge_type(GE_CONFIG):
    """Get GPP and ET coupling type"""
    if GE_CONFIG == 1:
        return "acm_wue"
    elif GE_CONFIG == 2:
        return "pml2"
    else:
        print("ERROR: GPP and ET coulping module type must be between 1-2:")
        print("1) ACM and WUE")
        print("2) PML2")
        sys.exit(1)


def get_unseen_time_type(UNTM):
    """Get unseen time type"""
    if UNTM == 1:
        return "unseen_time"
    elif UNTM == 0:
        return "all_time"
    else:
        print("ERROR: Unseen time type must be between 0-1:")
        print("0) All time")
        print("1) Unseen time")
        sys.exit(1)


def start_message(VERBOSE, RUN_INDEX, SOC_TYPE, GE_TYPE, UNTM_TYPE, RANDOM_SEED, WEIGHTS,
                  BATCH_SIZE, LEARNING_RATE, ITERATIONS, EPOCHS, TIME_STEP, SOIL_DEPTH, VARNAME_LS,
                  POINT2GRID, UNCERTAINTY):
    logging.info("*" * 70)
    logging.info("Welcome to DifferLand!")
    logging.info("an automatically differentiable hybrid terrestrial biosphere model")
    logging.info("*" * 70)
    logging.info("SOC module type: {}".format(SOC_TYPE))
    logging.info("GPP and ET coupling module type: {}".format(GE_TYPE))
    logging.info("Unseen time type: {}".format(UNTM_TYPE))
    logging.info("TIME STEP: {} days".format(TIME_STEP))
    logging.info("SOIL DEPTH: {} cm".format(SOIL_DEPTH))
    logging.info("Batch size: {} grids".format(BATCH_SIZE))
    flat_index = [item for sublist in VARNAME_LS for item in sublist]
    logging.info("Weight for ")
    if VERBOSE:
        print("Weight for ")
    for index_ls in VARNAME_LS:
        for var_name in index_ls:
            flat_var_index = flat_index.index(var_name)
            logging.info("{}, {}".format(var_name, WEIGHTS[flat_var_index]))
            if VERBOSE:
                print("{}, {}".format(var_name, WEIGHTS[flat_var_index]))
    logging.info("LEARNING RATE: {}".format(LEARNING_RATE))
    logging.info("TOTAL EPOCHES: {}".format(EPOCHS))
    logging.info("TOTAL ITERATIONS: {}".format(ITERATIONS))
    logging.info("Run index: {}".format(RUN_INDEX))
    logging.info("Random seed: {}".format(RANDOM_SEED))
    logging.info("Point to grid observations: {}".format(bool(POINT2GRID)))
    logging.info("Uncertainty of observations: {}".format(bool(UNCERTAINTY)))

    if VERBOSE:
        print("*" * 70)
        print("Welcome to DifferLand!")
        print("an automatically differentiable hybrid terrestrial biosphere model")
        print("*" * 70)
        print("SOC module type: {}".format(SOC_TYPE))
        print("GPP and ET coupling module type: {}".format(GE_TYPE))
        print("Unseen time type: {}".format(UNTM_TYPE))
        print("TIME STEP: {} days".format(TIME_STEP))
        print("SOIL DEPTH: {} cm".format(SOIL_DEPTH))
        print("Batch size: {} grids".format(BATCH_SIZE))
        print("LEARNING RATE: {}".format(LEARNING_RATE))
        print("TOTAL EPOCHES: {}".format(EPOCHS))
        print("TOTAL ITERATIONS: {}".format(ITERATIONS))
        print("Run index: {}".format(RUN_INDEX))
        print("Random seed: {}".format(RANDOM_SEED))
        print("Point to grid observations: {}".format(bool(POINT2GRID)))
        print("Uncertainty of observations: {}".format(bool(UNCERTAINTY)))

    return


def create_folder_if_not_exists(FOLDER_PATH, VERBOSE):
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)
        if VERBOSE:
            print("Created folder: {}".format(FOLDER_PATH))
    else:
        if VERBOSE:
            print("Folder path exists: {}".format(FOLDER_PATH))
    return


# Part2, functions for optimization
# ===============================================================================================
# ===============================================================================================
# Function to compute loss
def compute_loss(params, state, predictors, dalec_model, variant_forcings, invariant_forcings, variant_obs,
                 invariant_obs, grid_weights, obs_weights, obs_mean, obs_std, var_obs_ind, invar_obs_ind, params_ind, pools_ind):
    # Call FCNN to predict initial pools and DALEC parameters
    init_pools, init_params = state.apply_fn(params, predictors)
    # Call dalec_model.compute_loss_batch to cal total loss
    obs_loss_ls = dalec_model.compute_loss_batch(
        init_pools, variant_forcings, invariant_forcings, init_params, variant_obs, invariant_obs, 
        grid_weights, obs_mean, obs_std, var_obs_ind, invar_obs_ind, params_ind, pools_ind
    )
    loss_sum = [jnp.mean(obs_loss) for obs_loss in obs_loss_ls]
    loss_sum = jnp.array(loss_sum)
    total_loss = jnp.sum(obs_weights * loss_sum)

    # jax.debug.breakpoint()

    return total_loss


# Function to train at each batch
def train_with_batches(num_batches, ITERATIONS, tr_pred_ls, tr_invar_forc_ls, tr_var_forc_ls, 
                       tr_invar_obs_ls, tr_var_obs_ls, tr_grid_weig_ls, loss_grad_fn, state, dalec_model, 
                       obs_weights, obs_mean, obs_std, var_obs_ind, invar_obs_ind, params_ind, pools_ind, key):
    ref_min_samples = 30
    for ba_idx in range(num_batches):
        ba_tr_pred_ls = []
        ba_tr_invar_forc_ls = []
        ba_tr_var_forc_ls = []
        ba_tr_invar_obs_ls = []
        ba_tr_var_obs_ls = []
        ba_tr_grid_weig_ls = []
        for tr_pred, tr_invar_forc, tr_var_forc, tr_invar_obs, tr_var_obs, tr_grid_weig in zip(
            tr_pred_ls, tr_invar_forc_ls, tr_var_forc_ls, tr_invar_obs_ls, tr_var_obs_ls, tr_grid_weig_ls):
            tr_samples = tr_pred.shape[0]
            min_samples = jnp.minimum(tr_samples, ref_min_samples)
            mini_batch_size = tr_samples // num_batches + (1 if tr_samples % num_batches != 0 else 0)
            mini_batch_size = jnp.maximum(mini_batch_size, min_samples)
            if tr_samples < num_batches*min_samples:
                key, subkey = jax.random.split(key)
                indices = jax.random.choice(subkey, tr_samples, (mini_batch_size,), replace=False)
                ba_tr_pred_ls.append(tr_pred[indices])
                ba_tr_invar_forc_ls.append(tr_invar_forc[indices])
                ba_tr_var_forc_ls.append(tr_var_forc[indices])
                ba_tr_invar_obs_ls.append(tr_invar_obs[indices])
                ba_tr_var_obs_ls.append(tr_var_obs[indices])
                ba_tr_grid_weig_ls.append(tr_grid_weig[indices])
            else:
                if ba_idx < num_batches - 1:
                    st_idx = ba_idx * mini_batch_size 
                    end_idx = (ba_idx + 1) * mini_batch_size
                else:
                    st_idx = tr_samples - mini_batch_size
                    end_idx = tr_samples
                ba_tr_pred_ls.append(tr_pred[st_idx:end_idx])
                ba_tr_invar_forc_ls.append(tr_invar_forc[st_idx:end_idx])
                ba_tr_var_forc_ls.append(tr_var_forc[st_idx:end_idx])
                ba_tr_invar_obs_ls.append(tr_invar_obs[st_idx:end_idx])
                ba_tr_var_obs_ls.append(tr_var_obs[st_idx:end_idx])
                ba_tr_grid_weig_ls.append(tr_grid_weig[st_idx:end_idx])
            
            # jax.debug.breakpoint()

        ba_tr_pred = jnp.concatenate(ba_tr_pred_ls, axis=0)
        ba_tr_var_forc = jnp.concatenate(ba_tr_var_forc_ls, axis=0)
        ba_tr_invar_forc = jnp.concatenate(ba_tr_invar_forc_ls, axis=0)
        ba_tr_var_obs = jnp.concatenate(ba_tr_var_obs_ls, axis=0)
        ba_tr_invar_obs = jnp.concatenate(ba_tr_invar_obs_ls, axis=0)
        ba_tr_grid_weig = jnp.concatenate(ba_tr_grid_weig_ls, axis=0)
        for _ in range(ITERATIONS):
            train_loss, grads = loss_grad_fn(state.params, state, ba_tr_pred, dalec_model, ba_tr_var_forc,
                                             ba_tr_invar_forc, ba_tr_var_obs, ba_tr_invar_obs, ba_tr_grid_weig, 
                                             obs_weights, obs_mean, obs_std, var_obs_ind, invar_obs_ind, params_ind, pools_ind)
            # Apply gradients to update the model parameters
            state = state.apply_gradients(grads=grads)
        
        # jax.debug.breakpoint()

    return state, train_loss


# Part3, functions for training and predicting
# ===============================================================================================
# ===============================================================================================
def calibration_global(VERBOSE, SOC_CONFIG, GE_CONFIG, UNTM, RUN_INDEX, RANDOM_SEED,  # params for optimization
                       OBS_WEIGHTS, BATCH_SIZE, LEARNING_RATE, ITERATIONS, EPOCHS, 
                       POINT2GRID, UNCERTAINTY,
                       VISUALIZATION, MASK, METRIC,  # params for visualization
                       LOG_DIR, OUT_DIR, FIG_DIR,   # params for output dir
                       PREDICTOR_PATH, FORCING_DIR, INVAR_FORCING_LS, VAR_FORCING_LS,  # params for load data
                       OBSERVATION_DIR, TOTAL_INVAR_OBS_LS, TOTAL_VAR_OBS_LS, GRID_WEIGHT_DIR, 
                       TIME_STEP, REGION, YEAR_RANGE, SOIL_DEPTH,
                       NUM_HIDDEN, NUM_LAYER, THETA_POOLS, THETA_OTHERS,  # params for NN model
                       VARNAME_LS, VAR_OBS_IND, INVAR_OBS_IND, PARAMS_IND, POOLS_IND  # other params
                       ):
    # Create folder to save result
    create_folder_if_not_exists(LOG_DIR, VERBOSE)
    create_folder_if_not_exists(OUT_DIR, VERBOSE)
    create_folder_if_not_exists(FIG_DIR, VERBOSE)
    SOC_TYPE = get_soc_type(SOC_CONFIG)
    GE_TYPE = get_ge_type(GE_CONFIG)
    UNTM_TYPE = get_unseen_time_type(UNTM)
    OBS_WEIGHTS = jnp.array(OBS_WEIGHTS)
    VAR_NUM = len(OBS_WEIGHTS)

    # Print basic info
    # ========================================================================================
    training_log_path = os.path.join(LOG_DIR,
                                     "{}_{}_{}_ts_{}_bs_{}_lr_{}_epo_{}_it_{}_r_{}_unc_{}_p2g_{}_i_{}.log".format(
                                         SOC_TYPE, GE_TYPE, UNTM_TYPE, TIME_STEP, BATCH_SIZE, LEARNING_RATE,
                                         EPOCHS, ITERATIONS, RANDOM_SEED, UNCERTAINTY, POINT2GRID, RUN_INDEX))
    logging.basicConfig(level=logging.INFO,
                        filename=training_log_path,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filemode='w',
                        force=True)
    start_message(VERBOSE, RUN_INDEX, SOC_TYPE, GE_TYPE, UNTM_TYPE, RANDOM_SEED, OBS_WEIGHTS, 
                  BATCH_SIZE, LEARNING_RATE, ITERATIONS, EPOCHS, TIME_STEP, SOIL_DEPTH, VARNAME_LS,
                  POINT2GRID, UNCERTAINTY)

    if VERBOSE:
        print("\nTraining log will be stored in: {}".format(training_log_path))

    # Load data
    # ========================================================================================
    (predictors, invariant_forcings, variant_forcings, invariant_obs, variant_obs, grid_weights,
     train_mask_ls, train_mask, develop_mask, test_mask, valid_mask, lat, lon) = load_all_data(
        PREDICTOR_PATH, FORCING_DIR, INVAR_FORCING_LS, VAR_FORCING_LS, OBSERVATION_DIR,
        TOTAL_INVAR_OBS_LS, TOTAL_VAR_OBS_LS, GRID_WEIGHT_DIR, UNCERTAINTY, 
        REGION, YEAR_RANGE, RANDOM_SEED
    )

    # Cal mean value and std of each observation
    grid_weights_mask = grid_weights > 0
    obs_mean_ls = []
    obs_std_ls = []
    for vi in range(variant_obs.shape[2]):
        obs_mean = jnp.nanmean(variant_obs[grid_weights_mask[:, vi], :, vi], axis=(0, 1))
        obs_std = jnp.nanstd(variant_obs[grid_weights_mask[:, vi], :, vi], axis=(0, 1))
        obs_mean_ls.append(obs_mean)
        obs_std_ls.append(obs_std)
    n_var = variant_obs.shape[2]
    for vi in range(invariant_obs.shape[1]):
        obs_mean = jnp.nanmean(invariant_obs[grid_weights_mask[:, vi+n_var], vi], axis=0)
        obs_std = jnp.nanstd(invariant_obs[grid_weights_mask[:, vi+n_var], vi], axis=0)
        obs_mean_ls.append(obs_mean)
        obs_std_ls.append(obs_std)
    obs_mean = jnp.array(obs_mean_ls)
    obs_std = jnp.array(obs_std_ls)

    # jax.debug.breakpoint()

    logging.info("\n--- Data Shapes and Data Types ---")
    logging.info(f"Predictors: shape={predictors.shape}, type={type(predictors)}")
    logging.info(f"Invariant forcigns: shape={invariant_forcings.shape}, type={type(invariant_forcings)}")
    logging.info(f"Variant forcings: shape={variant_forcings.shape}, type={type(variant_forcings)}")
    logging.info(f"Invariant observations: shape={invariant_obs.shape}, type={type(invariant_obs)}")
    logging.info(f"Variant observations: shape={variant_obs.shape}, type={type(variant_obs)}")
    logging.info(f"Grid weights: shape={grid_weights.shape}, type={type(grid_weights)}")
    logging.info("\n--- Mask Information ---")
    logging.info(f"Total samples={valid_mask.size}, not-nan samples={np.sum(valid_mask)}, type={type(valid_mask)}")
    logging.info(f"Training samples={np.sum(train_mask)}, type={type(train_mask)}")
    logging.info(f"Developing samples={np.sum(develop_mask)}, type={type(develop_mask)}")
    logging.info(f"Testing samples={np.sum(test_mask)}, type={type(test_mask)}")
    if VERBOSE:
        print("\n--- Data Shapes and Data Types ---")
        print(f"Predictors: shape={predictors.shape}, type={type(predictors)}")
        print(f"Invariant forcings: shape={invariant_forcings.shape}, type={type(invariant_forcings)}")
        print(f"Variant forcigns: shape={variant_forcings.shape}, type={type(variant_forcings)}")
        print(f"Invariant observations: shape={invariant_obs.shape}, type={type(invariant_obs)}")
        print(f"Variant observations: shape={variant_obs.shape}, type={type(variant_obs)}")
        print(f"Grid weights: shape={grid_weights.shape}, type={type(grid_weights)}")
        print("\n--- Mask Information ---")
        print(f"Total samples={valid_mask.size}, not-nan samples={np.sum(valid_mask)}, type={type(valid_mask)}")
        print(f"Training samples={np.sum(train_mask)}, type={type(train_mask)}")
        print(f"Developing samples={np.sum(develop_mask)}, type={type(develop_mask)}")
        print(f"Testing samples={np.sum(test_mask)}, type={type(test_mask)}")

    # Create model
    # ========================================================================================
    # Dalec2 model
    dalec_model = DALEC990(ge_type=GE_TYPE, soc_type=SOC_TYPE, time_step=TIME_STEP, soil_depth=SOIL_DEPTH)
    # FCNN
    fcnn_model = FCNN(
        num_hidden=NUM_HIDDEN,
        num_layers=NUM_LAYER,
        theta_pools=THETA_POOLS,
        theta_others=THETA_OTHERS
    )
    # Optimizer based on gradient
    loss_grad_fn = jax.value_and_grad(partial(compute_loss), argnums=0)

    # TrainState, for initial and update state of FCNN
    class TrainState(train_state.TrainState):
        pass

    # Start calibration
    # ========================================================================================
    if UNTM == 1:
        new_variant_forcings = variant_forcings[:, :int(variant_forcings.shape[1] / 2), :]
        new_variant_obs = variant_obs[:, :int(variant_obs.shape[1] / 2), :]
    else:
        new_variant_forcings = variant_forcings
        new_variant_obs = variant_obs
    # Developing samples
    del_predictors = predictors[develop_mask]
    del_invariant_forcings = invariant_forcings[develop_mask]
    del_variant_forcings = new_variant_forcings[develop_mask]
    del_invariant_obs = invariant_obs[develop_mask]
    del_variant_obs = new_variant_obs[develop_mask]
    del_grid_weights = grid_weights[develop_mask]
    # Testing samples
    te_predictors = predictors[test_mask]
    te_invariant_forcings = invariant_forcings[test_mask]
    te_variant_forcings = new_variant_forcings[test_mask]
    te_invariant_obs = invariant_obs[test_mask]
    te_variant_obs = new_variant_obs[test_mask]
    te_grid_weights = grid_weights[test_mask]
    # Divided training samples into small batches
    tr_predictors_ls = []
    tr_invariant_forcings_ls = []
    tr_variant_forcings_ls = []
    tr_invariant_obs_ls = []
    tr_variant_obs_ls = []
    tr_grid_weights_ls = []
    for tr_mask in train_mask_ls:
        tr_predictors_ls.append(predictors[tr_mask])
        tr_invariant_forcings_ls.append(invariant_forcings[tr_mask])
        tr_variant_forcings_ls.append(new_variant_forcings[tr_mask])
        tr_invariant_obs_ls.append(invariant_obs[tr_mask])
        tr_variant_obs_ls.append(new_variant_obs[tr_mask])
        tr_grid_weights_ls.append(grid_weights[tr_mask])
    # Total training samples
    num_samples = np.sum(train_mask)
    num_batches = num_samples // BATCH_SIZE + (1 if num_samples % BATCH_SIZE != 0 else 0)

    COMPLETE = False
    key = random.PRNGKey(RANDOM_SEED)
    init_num = 0
    while not COMPLETE:
        loss_ls = [[] for i in range(VAR_NUM + 1)]
        best_del_loss = float('inf')
        update_learning_rate = LEARNING_RATE
        optimizer = optax.adam(learning_rate=update_learning_rate)
        # Initial state
        key, subkey = random.split(key)
        params = fcnn_model.init(subkey, predictors)
        state = TrainState.create(apply_fn=fcnn_model.apply, params=params, tx=optimizer)
        init_num += 1
        # Constrain total initial times
        if init_num > 100:
            break
        for i in range(EPOCHS):
            state, train_loss = train_with_batches(num_batches, ITERATIONS, tr_predictors_ls, tr_invariant_forcings_ls, 
                                                   tr_variant_forcings_ls, tr_invariant_obs_ls, tr_variant_obs_ls, 
                                                   tr_grid_weights_ls, loss_grad_fn, state, dalec_model, 
                                                   OBS_WEIGHTS, obs_mean, obs_std, VAR_OBS_IND, INVAR_OBS_IND, 
                                                   PARAMS_IND, POOLS_IND, key)
            # Monitor with developing samples
            init_pools, init_params = state.apply_fn(state.params, del_predictors)
            obs_loss_ls = dalec_model.compute_loss_batch(
                init_pools, del_variant_forcings, del_invariant_forcings, init_params,
                del_variant_obs, del_invariant_obs, del_grid_weights, obs_mean, obs_std, 
                VAR_OBS_IND, INVAR_OBS_IND, PARAMS_IND, POOLS_IND)
            loss_sum = [jnp.mean(obs_loss) for obs_loss in obs_loss_ls]
            loss_sum = jnp.array(loss_sum)
            del_loss = jnp.sum(OBS_WEIGHTS * loss_sum)

            # jax.debug.breakpoint()

            loss_ls[0].append(del_loss)
            for j in range(VAR_NUM):
                loss_ls[j + 1].append(OBS_WEIGHTS[j] * loss_sum[j])

            test_loss, _ = loss_grad_fn(state.params, state, te_predictors, dalec_model, te_variant_forcings,
                                        te_invariant_forcings, te_variant_obs, te_invariant_obs, te_grid_weights, OBS_WEIGHTS,
                                        obs_mean, obs_std, VAR_OBS_IND, INVAR_OBS_IND, PARAMS_IND, POOLS_IND)

            # Check if nan or inf
            if (jnp.isnan(train_loss) or jnp.isinf(train_loss) or jnp.isnan(del_loss) or jnp.isinf(del_loss) or
                    jnp.isnan(test_loss) or jnp.isinf(test_loss)):
                if VERBOSE:
                    print(
                        f"\nEpoch {i + 1}, training loss: {train_loss:.3f}, developing loss: {del_loss:.3f}, "
                        f"testing loss: {test_loss:.3f}"
                    )
                    print("Numerical issue encountered, reinitializing the model...")
                logging.info(
                    f"\nEpoch {i + 1}, training loss: {train_loss:.3f}, developing loss: {del_loss:.3f}, "
                    f"testing loss: {test_loss:.3f}"
                )
                logging.info("Numerical issue encountered, reinitializing the model...")
                break
            if i % 100 == 0:
                if VERBOSE:
                    print(
                        f"\nEpoch {i + 1}, training loss: {train_loss:.3f}, developing loss: {del_loss:.3f}, "
                        f"testing loss: {test_loss:.3f}"
                    )
                logging.info(
                    f"\nEpoch {i + 1}, training loss: {train_loss:.3f}, developing loss: {del_loss:.3f}, "
                    f"testing loss: {test_loss:.3f}"
                )
            # Update learning rate
            if del_loss >= best_del_loss * 1.1:
                logging.info(f"\nEpoch: {i + 1}")
                logging.info(f"110% of Best developing loss: {best_del_loss * 1.1:.3f}")
                logging.info(
                    f"Current: developing loss: {del_loss:.3f}, training loss: {train_loss:.3f}, "
                    f"testing loss: {test_loss:.3f}")
                logging.info(
                    f"Current learning rate: {update_learning_rate:.6f}; "
                    f"Updated learning rate: {update_learning_rate * 0.99:.6f}")
                if VERBOSE:
                    print(f"\nEpoch: {i + 1}")
                    print(f"110% of Best developing loss: {best_del_loss * 1.1:.3f}")
                    print(
                        f"Current: developing loss: {del_loss:.3f}, training loss: {train_loss:.3f}, "
                        f"testing loss: {test_loss:.3f}")
                    print(
                        f"Current learning rate: {update_learning_rate:.6f}; "
                        f"Updated learning rate: {update_learning_rate * 0.99:.6f}")
                update_learning_rate = update_learning_rate * 0.99
                optimizer = optax.adam(learning_rate=update_learning_rate)
                state = TrainState.create(apply_fn=fcnn_model.apply, params=state.params, tx=optimizer)
            best_del_loss = min(best_del_loss, del_loss)
            if i == EPOCHS - 1:
                COMPLETE = True

    # Save result
    # ========================================================================================
    init_pools, init_params = state.apply_fn(state.params, predictors)
    results = dalec_model.step_spa_batch(init_pools, variant_forcings, invariant_forcings, init_params)
    save_path = os.path.join(
        OUT_DIR, "{}_{}_{}_ts_{}_bs_{}_lr_{}_epo_{}_it_{}_r_{}_unc_{}_p2g_{}_i_{}.pkl".format(
            SOC_TYPE, GE_TYPE, UNTM_TYPE, TIME_STEP, BATCH_SIZE, LEARNING_RATE, EPOCHS, ITERATIONS,
            RANDOM_SEED, UNCERTAINTY, POINT2GRID, RUN_INDEX))
    mask_ls = [train_mask, develop_mask, test_mask, valid_mask]
    VAR_IND_LS = [VAR_OBS_IND, INVAR_OBS_IND, PARAMS_IND, POOLS_IND]
    param_min = dalec_model.param_min
    param_max = dalec_model.param_max
    pool_min = dalec_model.pool_min
    pool_max = dalec_model.pool_max
    param_bound_ls = [param_min, param_max, pool_min, pool_max]
    result_ls = [results, invariant_obs, variant_obs, grid_weights, loss_ls, mask_ls, REGION, lon, lat, 
                 init_pools, init_params, VARNAME_LS, VAR_IND_LS, param_bound_ls]
    with open(save_path, "wb") as f:
        pickle.dump(result_ls, f)
    if VERBOSE:
        print(f"Results saved to {save_path}")
    logging.info(f"Results saved to {save_path}")

    # Visualization
    if VISUALIZATION:
        fig_dir_base = os.path.join(
            FIG_DIR, "{}_{}_{}_ts_{}_bs_{}_lr_{}_epo_{}_it_{}_r_{}_unc_{}_p2g_{}_i_{}".format(
                SOC_TYPE, GE_TYPE, UNTM_TYPE, TIME_STEP, BATCH_SIZE, LEARNING_RATE, EPOCHS, ITERATIONS,
                RANDOM_SEED, UNCERTAINTY, POINT2GRID, RUN_INDEX))
        visualization_global(result_ls, fig_dir_base, mask=MASK, metric=METRIC)

    return

