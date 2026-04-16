# Parse the arguments
import argparse
import jax
import jax.numpy as jnp
from functools import partial
import optax
import os
import numpy as np
import pandas as pd
import pickle
import xarray as xr
from geopy.distance import geodesic
import logging
import sys
import time

sys.path.insert(1, '../')

from DifferLand.util.preprocessing import get_train_test_sel, generate_met_matrix, generate_site_level_target_matrix
from DifferLand.model.DALEC990 import DALEC990
from DifferLand.util.init_mlp_params import init_mlp_params
from DifferLand.util.normalization import par2nor
from DifferLand.util.visualization import plot_site_figure
from DifferLand.model.DALEC_990_parinfo import dalec990_param_parmin, dalec990_pool_parmin
from DifferLand.optimization.loss_functions import compute_test_nnse, compute_nnse_eval

DRIVERS_DIR = "../drivers/"
SITE_SOM_FILENAME = "../drivers/site_coords_som.csv"
FIG_DIR = "./fig/"
OUTPUT_DIR = "./output/"
LOG_DIR = "./log/"
LEARNING_RATE = 5e-4
TOTAL_ITERATIONS = 10000


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog='DifferLand Calibration',
                                     description='Run the script to calibrate the physical and neural parameters in the DifferLand model.')
    parser.add_argument("-n", "--sitename", help="Enter the SITE_ID of FLUXNET site, e.g. US-Var", metavar="SITE_ID",
                        default="AU-DaS")
    parser.add_argument("-w", "--with_som", help="Enter whether use som in loss function, 0 no, 1 yes",
                        type=int, default=1)
    parser.add_argument("-c", "--model_configuration", metavar="CONFIG_IDX",
                        help="Enter the index for model configuration, range from 1-6."
                             + "(1) baseline; (2) beta-JS; (3) beta-nn; (4) GPP&ET(NN)_MET; (5) GPP&ET(NN)_MET+LAI; (6) GPP(ACM)_ET(NN)",
                        type=int, default=2)
    parser.add_argument("-m", "--scm_configuration", metavar="SCM_IDX",
                        help="Enter the index for soil carbon model configuration, range from 1-2. "
                             + "(1) One_pool; (2) Three_pool_Century",
                        type=int, default=2)
    parser.add_argument("-i", "--run_index", metavar="RUN_IDX",
                        help="Enter an index for the run, used to identify independent calibrations", type=int,
                        default=1)
    parser.add_argument("-s", "--silent", help="Silence message printing in the terminal.",
                        action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    return args


def get_with_som_type(model_configuration):
    """Get wether use som in loss function or not for the model_configuration index"""
    if model_configuration == 0:
        return "without_som"
    elif model_configuration == 1:
        return "with_som"
    else:
        print("ERROR: Model type must be between 0-1:")
        print("0) Without som in loss function")
        print("1) With som in loss function")
        sys.exit(1)


def get_stress_type(model_configuration):
    """Get stress type str for the model_configuration index"""
    if model_configuration == 1:
        return "baseline"
    elif model_configuration == 2:
        return "default"
    elif model_configuration == 3:
        return "nn_paw"
    elif model_configuration == 4:
        return "nn_whole_no_lai"
    elif model_configuration == 5:
        return "nn_whole"
    elif model_configuration == 6:
        return "gpp_acm_et_nn"
    else:
        print("ERROR: Model type must be between 1-6:")
        print("1) baseline")
        print("2) JS-beta")
        print("3) nn-beta")
        print("4) GPP&ET(NN)_MET")
        print("5) GPP&ET(NN)_MET+LAI")
        print("6) GPP(ACM)_ET(NN)")
        sys.exit(1)


def get_scm_type(scm_consiguration):
    """Get soil carbon model type str for the scm_type index"""
    if scm_consiguration == 1:
        return "onepool"
    elif scm_consiguration == 2:
        return "century"
    else:
        print("ERROR: Soil carbon model type must be between 1-2:")
        print("1) One_pool")
        print("2) Three_pool_Century")

        sys.exit(1)


def start_message(verbose, sitename, driver_name, run_index, stress_type, scm_type, with_som_type):
    if verbose:
        print("*" * 70)
        print("Welcome to DifferLand!")
        print("an automatically differentiable hybrid terrestrial biosphere model")
        print("*" * 70)
        print("Sitename: {}".format(sitename))
        print("Driver name: {}".format(driver_name))
        print("Run index: {}".format(run_index))
        print("PAW limitation type: {}".format(stress_type))
        print("Soil carbon model type: {}".format(scm_type))
        print("Loss function {}".format(with_som_type.replace('_', ' ')))
        print("LEARNING RATE: {}".format(LEARNING_RATE))
        print("TOTAL ITERATIONS: {}".format(TOTAL_ITERATIONS))
        print()


def create_folder_if_not_exists(folder_path, verbose=False):
    """Create a folder if the folder doesn't exist"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        if verbose:
            print("Created folder: {}".format(folder_path))
    else:
        if verbose:
            print("Folder path exists: {}".format(folder_path))
    return


def get_som(sitename, verbose=False):
    """Get total som for the site"""
    try:
        site_som_df = pd.read_csv(SITE_SOM_FILENAME)
    except FileNotFoundError:
        print("ERROR: SITE_LOC_FILENAME: {} not found".format(SITE_SOM_FILENAME))
        sys.exit(1)
    try:
        som = site_som_df[site_som_df.SITE_ID == sitename]["som"].values.item()
    except:
        print("ERROR: Cannot access the som for the site {} in the file {}".format(sitename, SITE_SOM_FILENAME))
    if verbose:
        print("Soil organic matter: {:.2f} gC/m^2".format(som))
    return som


def get_driver_ds(driver_name):
    """Load driver data from NetCDF file"""
    try:
        driver_ds = xr.open_dataset(os.path.join(DRIVERS_DIR, driver_name))
    except FileNotFoundError:
        print("Driver file {} not found!".format(os.path.join(DRIVERS_DIR, driver_name)))
        sys.exit(1)
    return driver_ds


def get_data_matrices(driver_ds, som, train_sel):
    """Get meterologial forcings and target variable matracies"""
    met_matrix_full = generate_met_matrix(driver_ds, train_sel, train_mode=False)
    target_matrix_full = generate_site_level_target_matrix(driver_ds, train_sel, som, train_mode=False, reco=True)
    return met_matrix_full, target_matrix_full


def initialize_nn_params(stress_type):
    """Initialize parameters for neural network subcomponents"""
    if stress_type == "baseline":
        gpp_params = init_mlp_params([1, 1, 1], n=np.random.randint(9e10))
    elif stress_type == "default":
        gpp_params = init_mlp_params([1, 1, 1], n=np.random.randint(9e10))
    elif stress_type == "nn_paw":
        gpp_params = init_mlp_params([1, 10, 1], n=np.random.randint(9e10))
    elif stress_type == "nn_whole_no_lai":
        gpp_params = init_mlp_params([5, 10, 10, 2], n=np.random.randint(9e10))
    elif stress_type == "nn_whole":
        gpp_params = init_mlp_params([6, 10, 10, 2], n=np.random.randint(9e10))
    elif stress_type == "gpp_acm_et_nn":
        beta_params = init_mlp_params([1, 10, 1], n=np.random.randint(9e10))
        et_params = init_mlp_params([7, 10, 10, 1], n=np.random.randint(9e10))
        gpp_params = (beta_params, et_params)
    return gpp_params


def initialize_physical_parameters(som, model, with_som_configuration):
    """Initialize parameters for the physical model parameters in DALEC"""
    # Independent on the range of params, except for the ce_opt & initial paw
    key = jax.random.PRNGKey(np.random.randint(9999999))
    key, subkey = jax.random.split(key)
    pool_initial = jax.random.normal(subkey, dalec990_pool_parmin.shape)
    del subkey
    key, subkey = jax.random.split(key)

    # random normal distribution, mean value = 0, std = 1
    param_initial = jax.random.normal(subkey, dalec990_param_parmin.shape)
    # The initial paw size was initialized empirically to 500 kg H2O m-2 to avoid convergence issues
    pool_initial = pool_initial.at[0].set(par2nor(500, model.parmin.initial_PAW, model.parmax.initial_PAW))
    pool_initial = pool_initial.at[6].set(par2nor(som, model.parmin.Csoc, model.parmax.Csoc))
    if with_som_configuration == 1:
        som1 = som * 0.03
        som2 = som * 0.301
        som3 = som * 0.669
        pool_initial = pool_initial.at[7].set(par2nor(som1, model.parmin.Csoc1, model.parmax.Csoc1))
        pool_initial = pool_initial.at[8].set(par2nor(som2, model.parmin.Csoc2, model.parmax.Csoc2))
        pool_initial = pool_initial.at[9].set(par2nor(som3, model.parmin.Csoc3, model.parmax.Csoc3))

    del subkey
    return pool_initial, param_initial


def save_calibrated_result(driver_ds, som, train_sel, test_sel, met_matrix_full, target_matrix_full, model, fig_save_name,
                           output_save_name, param_state, verbose=False):
    """Save calibrated results to a pickle file"""
    param_initial_cal, pool_initial_cal, gpp_params_cal = param_state
    output_matrix_full = model.forward(param_initial_cal, pool_initial_cal, gpp_params_cal, met_matrix_full)
    nnse_eval = compute_nnse_eval(output_matrix_full, target_matrix_full, train_sel, test_sel, reco=True)
    plot_site_figure(driver_ds, som, output_matrix_full, train_sel, test_sel, nnse_eval, fig_save_name, reco=True)
    if verbose:
        print("Calibrated figure saved to: {}".format(fig_save_name))
    output_dict = {"param_state": param_state,
                   "driver_ds": driver_ds,
                   "som":  som,
                   "output_matrix_full": output_matrix_full,
                   "train_sel": train_sel,
                   "test_sel": test_sel,
                   "nnse_eval": nnse_eval}

    with open(output_save_name, "wb") as fp:
        pickle.dump(output_dict, fp)
    if verbose:
        print("Calibrated parameters, performance metrics, and model output dict saved to: {}".format(output_save_name))


def main():
    
    t0 = time.time()
    
    # basic settings
    # ========================================================================================
    # get args info from keyboard
    args = parse_args()
    verbose = not args.silent
    sitename = args.sitename
    driver_name = f"{sitename}.nc"
    model_configuration = args.model_configuration
    with_som_configuration = args.with_som
    with_som_type = get_with_som_type(with_som_configuration)
    scm_configuration = args.scm_configuration
    run_index = args.run_index
    # convert stress type and soil carbon type to str
    stress_type = get_stress_type(model_configuration)
    scm_type = get_scm_type(scm_configuration)
    # print basic info
    start_message(verbose, sitename, driver_name, run_index, stress_type, scm_type, with_som_type)
    # create folder to save result
    create_folder_if_not_exists(FIG_DIR, verbose)
    create_folder_if_not_exists(LOG_DIR, verbose)
    create_folder_if_not_exists(OUTPUT_DIR, verbose)
    # print info for training loggings
    training_log_path = os.path.join(LOG_DIR, "daily_{}_{}_{}_{}_{}_{}_{}.log".format(sitename, with_som_type, stress_type, scm_type, LEARNING_RATE, TOTAL_ITERATIONS, run_index))
    logging.basicConfig(level=logging.INFO,
                        filename=training_log_path,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    if verbose:
        print("Training log will be stored in: {}".format(training_log_path))

    # load data
    # ========================================================================================
    if verbose:
        print("Loading datasets for model calibration...")
    # som data
    som = get_som(sitename, verbose)
    # driver data
    driver_ds = get_driver_ds(driver_name)
    # get the index of training and testing data
    train_sel, test_sel = get_train_test_sel(driver_ds)
    # select training and testing data based on index
    met_matrix_full, target_matrix_full = get_data_matrices(driver_ds, som, train_sel)

    # create model
    # ========================================================================================
    model = DALEC990(np.sum(train_sel), water_stress_type=stress_type, scm_type=scm_type, with_som=with_som_configuration, reco=True)
    # params optimization function based on gradient
    tx = optax.adam(learning_rate=LEARNING_RATE)
    loss_grad_fn = jax.value_and_grad(partial(model.compute_loss), [0, 1, 2])

    @jax.jit
    def update(params, met_matrix, target_matrix, opt_state, k):
        param_initial, pool_initial, gpp_params = params
        loss, grads = loss_grad_fn(param_initial, pool_initial, gpp_params, met_matrix, target_matrix, k)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    # start calibration
    COMPLETE = False
    if verbose:
        print("Calibration START!")
    init_num = 0
    while not COMPLETE:
        gpp_params = initialize_nn_params(stress_type)
        pool_initial, param_initial = initialize_physical_parameters(som, model, with_som_configuration)
        param_state = (param_initial, pool_initial, gpp_params)
        opt_state = tx.init(param_state)
        init_num += 1
        # if initial more than 100 times, break the loop
        if init_num > 100:
            break
        if verbose:
            print("Parameters initialized {}".format(init_num))
        for i in range(TOTAL_ITERATIONS):
            loss, param_state, opt_state = update(param_state, met_matrix_full, target_matrix_full, opt_state, np.minimum(i * (30 / 10000), 30))
            if jnp.isnan(loss):
                if verbose:
                    print("iter {}, loss: {:.3f}".format(i + 1, loss))
                    print("Numerical issue encountered, trying to reinitialize the model...\n")
                logging.info("Numerical issue encountered, trying to reinitialize the model...")
                break
            if i % 100 == 0:
                param_initial_cal, pool_initial_cal, gpp_params_cal = param_state
                output_matrix_full = model.forward(param_initial_cal, pool_initial_cal,
                                                   gpp_params_cal, met_matrix_full)
                test_nnse = compute_test_nnse(output_matrix_full, target_matrix_full, test_sel, reco=True)
                logging.info("iter {}, loss: {:.3f}, test nnse: {:.3f}".format(i + 1, loss, test_nnse))
                if verbose:
                    print("iter {}, loss: {:.3f}, test nnse: {:.3f}".format(i + 1, loss, test_nnse))
            if i == TOTAL_ITERATIONS - 1:
                COMPLETE = True
    if verbose:
        print("Training completed!")

    # save result
    # ========================================================================================
    fig_save_name = os.path.join(FIG_DIR, "daily_{}_{}_{}_{}_{}_{}_{}.png".format(sitename, with_som_type, stress_type, scm_type, LEARNING_RATE, TOTAL_ITERATIONS, run_index))
    output_save_name = os.path.join(OUTPUT_DIR, "daily_{}_{}_{}_{}_{}_{}_{}.pkl".format(sitename, with_som_type, stress_type, scm_type, LEARNING_RATE, TOTAL_ITERATIONS, run_index))
    save_calibrated_result(driver_ds, som, train_sel, test_sel, met_matrix_full, target_matrix_full, model, fig_save_name, output_save_name, param_state, verbose)
    
    t1 = time.time()
    thour = int((t1-t0)/3600)
    tmin = int(((t1-t0) - thour*3600)/60)
    tsec = round((t1-t0) - thour*3600 -  tmin*60, 2)
    print(f'All running time: {thour}h {tmin}min {tsec}s')
    logging.info(f'All running time: {thour}h {tmin}min {tsec}s')


if __name__ == "__main__":
    main()