# -*- coding: utf-8 -*-


import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import matplotlib.pyplot as plt
from functools import partial
import logging
from datetime import date, timedelta
import xarray as xr
import sys
from copy import deepcopy


# Basic set
STATION_NAME = 'AU-Dry'
DRIVERS_DIR = f"./drivers/{STATION_NAME}.nc"
OUTPUT_FILENAME = f"./results/{STATION_NAME}_ce_opt.csv"
OUTPUT_LOGFILE = f"./results/{STATION_NAME}_optimization.log"
OUTPUT_FIGFILE = f"./results/{STATION_NAME}_results.png"
os.makedirs("./results", exist_ok=True)
ce_min = 5
ce_max = 50
LEARNING_RATE = 0.5
TOTAL_ITERATIONS = 400
logging.basicConfig(filename=OUTPUT_LOGFILE, level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Optimization process started")


# Get training data
def get_training_data(DRIVERS_DIR):
    driver_ds = xr.open_dataset(DRIVERS_DIR)
    lat = driver_ds.LAT.values.item()
    doy = driver_ds.DOY.values
    t_max = driver_ds.T2M_MAX.values
    t_min = driver_ds.T2M_MIN.values
    lai = driver_ds.LAI.values
    rad = driver_ds.SSRD.values
    ca = driver_ds.CO2.values
    time_vals = driver_ds.time.values
    time = np.array([date(2001, 1, 1) + timedelta(days=int(i)) for i in time_vals])

    # Assuming daily temporal resolution
    n_years = np.int64(np.round(len(time) / 365.25))
    train_years = np.round(np.ceil(n_years / 2))
    train_end_date = date(int(time[0].year + train_years), 1, 1)
    train_sel = time < train_end_date
    test_sel = time >= train_end_date
    return (driver_ds, lat, doy, t_max, t_min, lai, rad, ca, train_sel, test_sel)
    

# Define the ACM function
def ACM(lat, doy, t_max, t_min, lai, rad, ca, ce):
    d1 = 0.0156935  # day length constant
    theta = 4.22273
    k = 208.868
    d2 = 0.0453194
    b2 = 0.37836
    c1 = 7.19298  # Fixed the typo 'a' to '*'
    a2 = 0.011136
    c2 = 2.1001
    eb1T = 0.789798
    phi_d = -2
    H = 1

    # compute daily canopy conductance, gc
    gc = jnp.abs(phi_d) ** eb1T / (b2 * H + 0.5 * (t_max - t_min))

    # compute p parameter needed for ci
    p = lai * 1 * ce * jnp.exp(a2 * t_max) / gc

    # compute the q parameter needed for ci
    q = theta - k

    # compute the internal CO2 concentration, ci
    ci = 0.5 * (ca + q - p + jnp.sqrt((ca + q - p) ** 2 - 4 * (ca * q - p * theta)))

    # compute canopy-level quantum yield, e0
    e0 = c1 * (lai ** 2) / (c2 + lai ** 2)

    # compute day length based on latitude and day of year
    dec = -23.4 * jnp.cos((360 * (doy + 10) / 365) * jnp.pi / 180) * jnp.pi / 180
    mult = jnp.tan(lat * jnp.pi / 180) * jnp.tan(dec)
    mult_valid = (mult < 1) * (mult > -1)
    mult_temp = mult * mult_valid
    dayl = 24 * jnp.arccos(-mult_temp) / jnp.pi
    mult_geq_one_sel = (mult < 1)
    dayl = dayl * mult_geq_one_sel + (1 - mult_geq_one_sel) * 24
    mult_leq_minus_one_sel = (mult > -1)
    dayl = dayl * mult_leq_minus_one_sel + (1 - mult_geq_one_sel) * 0

    # compute co2 rate of diffusion to the site of fixation, pd
    pd = gc * (ca - ci)

    # compute light limitation pi
    pi = e0 * rad * pd / (e0 * rad + pd)

    # compute gpp
    gpp = pi * (d1 * dayl + d2)

    return gpp


# Vectorized ACM model
ACM_V = jax.jit(jax.vmap(jax.jit(ACM), in_axes=[None, 0, 0, 0, 0, 0, 0, None]))


def compute_nse(observed, modeled, mask):
    modeled = jnp.squeeze(modeled)
    numerator = jnp.sum((observed - modeled) ** 2 * mask)
    observed_mean = jnp.sum(observed * mask) / jnp.sum(mask)
    denominator = jnp.sum((observed - observed_mean) ** 2 * mask)
    nse = 1 - (numerator / denominator)
    return nse


def compute_nnse(observed, modeled, mask):
    nse = compute_nse(observed, modeled, mask)
    #nse = jnp.clip(nse, a_min=0, a_max=1) 
    return 1 / (2 - nse)
    
    
# Compute loss function
def computer_loss(ce, lat, doy, t_max, t_min, lai, rad, ca, gpp_target, gpp_target_mask, train_sel):
    gpp_pred = ACM_V(lat, doy, t_max, t_min, lai, rad, ca, ce)
    gpp_loss = compute_nnse(gpp_target[train_sel], gpp_pred[train_sel], gpp_target_mask[train_sel])
    return -gpp_loss


# Plot results
def plot_results(ce, lat, doy, t_max, t_min, lai, rad, ca, gpp_target, gpp_target_mask, train_sel, test_sel):
    
    gpp_pred = ACM_V(lat, doy, t_max, t_min, lai, rad, ca, ce)
    train_nnse = compute_nnse(gpp_target[train_sel], gpp_pred[train_sel], gpp_target_mask[train_sel])
    test_nnse = compute_nnse(gpp_target[test_sel], gpp_pred[test_sel], gpp_target_mask[test_sel])
    
    # Create figure and axis
    plt.figure(figsize=(14, 5))
    
    # plot gpp time series
    gpp_target[gpp_target_mask == 0] = np.nan
    plt.plot(gpp_target[train_sel], alpha=0.5)
    plt.plot(np.arange(np.sum(train_sel), len(test_sel)), gpp_target[test_sel], alpha=0.5)
    plt.plot(gpp_pred)
    
    plt.ylabel("GPP (gC/m$^2$/day)")
    plt.text(100, np.nanmax(gpp_target) * 0.9, "Train NNSE: {:.2f}".format(train_nnse))
    plt.text(100, np.nanmax(gpp_target) * 0.8, "Test NNSE: {:.2f}".format(test_nnse))

    # Show the plot
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGFILE)
    print(OUTPUT_FIGFILE)
    
    return


def main():
    # Get the training and test data
    driver_ds, lat, doy, t_max, t_min, lai, rad, ca, train_sel, test_sel = get_training_data(DRIVERS_DIR)
    gpp_target = deepcopy(driver_ds.GPP.values)
    gpp_target_mask = np.invert(np.isnan(gpp_target)).astype(np.float32)
    gpp_target[gpp_target_mask == 0] = -9999
    
    # Initialize the optimizer
    tx = optax.adam(learning_rate=LEARNING_RATE)

    # loss_grad_fn computes the gradient of the loss with respect to 'ce'
    loss_grad_fn = jax.value_and_grad(partial(computer_loss), argnums=0)

    # Iterative optimization
    COMPLETE = False
    print("Calibration START!")
    
    while not COMPLETE:
        # Initialization of parameters (only ce is optimized here)
        ce = np.random.uniform(ce_min, ce_max, size=(1,))
        
        opt_state = tx.init(ce)
    
        print("Parameters initialized")
        
        for i in range(TOTAL_ITERATIONS):

            # Compute the gradient and loss (pass all required parameters here)
            loss, grads = loss_grad_fn(ce, lat, doy, t_max, t_min, lai, rad, ca, gpp_target, gpp_target_mask, train_sel)
    
            # Apply the gradient updates to optimize 'ce'
            updates, opt_state = tx.update(grads, opt_state)
            ce = optax.apply_updates(ce, updates)  # Update 'ce'
            
            if jnp.isnan(loss):
                print("Numerical issue encountered, trying to reinitialize the model...\n")
                logging.info("Numerical issue encountered, trying to reinitialize the model...")
                break
                
            # Store predictions for plotting after every 100 steps
            if i % 100 == 0:
                # Make predictions with the current 'ce'
                gpp_pred = ACM_V(lat, doy, t_max, t_min, lai, rad, ca, ce)
                
                test_nnse = compute_nnse(gpp_target[test_sel], gpp_pred[test_sel], gpp_target_mask[test_sel])
                logging.info("iter {}, ce: {:.1f}, loss: {:.3f}, test nnse: {:.3f}".format(i + 1, ce.item(), loss, test_nnse))
                print("iter {}, ce: {:.1f}, loss: {:.3f}, test nnse: {:.3f}".format(i + 1, ce.item(), loss, test_nnse))
            if i == TOTAL_ITERATIONS - 1:
                COMPLETE = True
    
    # save result
    plot_results(ce, lat, doy, t_max, t_min, lai, rad, ca, gpp_target, gpp_target_mask, train_sel, test_sel)

    # Save the optimized 'ce' to a CSV file
    daily_ce_df = pd.DataFrame({"ce": ce})  # Wrap 'ce' into a dataframe for saving
    daily_ce_df.to_csv(OUTPUT_FILENAME, index=False)
    print(OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
