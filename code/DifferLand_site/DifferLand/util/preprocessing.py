import xarray as xr
import numpy as np
import collections
import os
import jax.numpy as jnp
import pandas as pd
from datetime import date, timedelta
from copy import deepcopy


# function for dividing training and testing data, return index
def get_train_test_sel(driver_ds):
    time_vals = driver_ds.time.values
    if time_vals[1] - time_vals[0] == 1:
        time = np.array([date(2001, 1, 1) + timedelta(days=int(i)) for i in time_vals])
        n_years = np.int64(np.round(len(time) / 365.25))
        train_years = np.round(np.ceil(n_years / 2))
        train_end_date = date(int(time[0].year + train_years), 1, 1)
        train_sel = time < train_end_date
        test_sel = time >= train_end_date
        return train_sel, test_sel
    elif time_vals[1] - time_vals[0] == 7:
        time = np.array([date(2001, 1, 1) + timedelta(days=int(i)) for i in time_vals])
        n_years = np.int64(np.round(len(time) / 52))
        train_years = np.round(np.ceil(n_years / 2))
        train_end_date = date(int(time[0].year + train_years), 1, 1)
        train_sel = time < train_end_date
        test_sel = time >= train_end_date
        return train_sel, test_sel
    elif time_vals[1] - time_vals[0] >= 28:
        time = np.array([date(2001, 1, 1) + timedelta(days=int(i)) for i in time_vals])
        n_years = np.int64(np.round(len(time) / 12))
        train_years = np.round(np.ceil(n_years / 2))
        train_end_date = date(int(time[0].year + train_years), 1, 1)
        train_sel = time < train_end_date
        test_sel = time >= train_end_date
        return train_sel, test_sel


# get driver data
def generate_met_matrix(driver_ds, train_sel, train_mode=True):
    time = driver_ds.time.values
    t_min = driver_ds.T2M_MIN.values
    t_max = driver_ds.T2M_MAX.values
    rad = driver_ds.SSRD.values
    ca = driver_ds.CO2.values
    doy = driver_ds.DOY.values
    vpd = driver_ds.VPD.values
    precipitation = driver_ds.TOTAL_PREC.values
    burned_area = np.full(time.shape[0], 0)
    lat = np.full(time.shape, driver_ds.LAT.values)
    delta_t = np.full(time.shape[0], time[1] - time[0])
    t_mean = np.full(time.shape[0], np.mean((t_min[train_sel] + t_max[train_sel]) / 2))
    mean_precipitation = np.full(time.shape[0], np.mean(precipitation[train_sel]))
    t_norm = ((t_max + t_min) / 2 - t_mean) / np.std((t_max[train_sel] + t_min[train_sel]) / 2)
    rad_norm = (rad - np.mean(rad[train_sel])) / np.std(rad[train_sel])
    vpd_norm = (vpd - np.mean(vpd[train_sel])) / np.std(vpd[train_sel])
    ca_norm = (ca - np.mean(ca[train_sel])) / np.std(ca[train_sel])

    end_of_year_sel = jnp.where(driver_ds.DOY.values == driver_ds.DOY.values[0].item())[0][1:] - 1
    end_of_year_index = np.full(len(driver_ds.DOY.values), False)
    end_of_year_index[end_of_year_sel] = True
    end_of_year_arr = jnp.array(end_of_year_index).astype(np.float32)

    # add new params
    mean_time = np.full(time.shape[0], np.mean(time[train_sel]))
    gpp = driver_ds.GPP.values
    mean_gpp = np.full(time.shape[0], np.nanmean(gpp[train_sel]))

    met_matrix = jnp.array([lat,
                            doy,
                            t_max,
                            t_min,
                            rad,
                            ca,
                            vpd,
                            t_norm,
                            rad_norm,
                            vpd_norm,
                            ca_norm,
                            time,
                            delta_t,
                            t_mean,
                            precipitation,
                            mean_precipitation,
                            mean_gpp,
                            end_of_year_arr]).T

    if train_mode:
        return met_matrix[train_sel, :]
    else:
        return met_matrix


# get target data, adjust the value of som1-3 when necessary
# we set them 0 here, as we don't use them in the loss function at present
def generate_site_level_target_matrix(driver_ds, train_sel, som, som1=0, som2=0, som3=0, train_mode=True, reco=False):
    gpp_target = deepcopy(driver_ds.GPP.values)
    gpp_target_mask = np.invert(np.isnan(gpp_target)).astype(np.float32)
    gpp_target[gpp_target_mask == 0] = -9999

    nee_target = deepcopy(driver_ds.NBE.values)
    nee_target_mask = np.invert(np.isnan(nee_target)).astype(np.float32)
    nee_target[nee_target_mask == 0] = -9999

    et_target = deepcopy(driver_ds.ET.values)
    et_target_mask = np.invert(np.isnan(et_target)).astype(np.float32)
    et_target[et_target_mask == 0] = -9999

    lai_target = deepcopy(driver_ds.LAI.values)
    lai_target_mask = np.invert(np.isnan(lai_target)).astype(np.float32)
    lai_target[lai_target_mask == 0] = -9999

    som_target = np.full(gpp_target.shape[0], som)
    som1_target = np.full(gpp_target.shape[0], som1)
    som2_target = np.full(gpp_target.shape[0], som2)
    som3_target = np.full(gpp_target.shape[0], som3)
    som_target_mask = np.full(gpp_target.shape[0], 1.0)
    som1_target_mask = np.full(gpp_target.shape[0], 1.0)
    som2_target_mask = np.full(gpp_target.shape[0], 1.0)
    som3_target_mask = np.full(gpp_target.shape[0], 1.0)

    if reco:
        reco_target = deepcopy(driver_ds.RECO.values)
        reco_target_mask = np.invert(np.isnan(reco_target)).astype(np.float32)
        reco_target[reco_target_mask == 0] = -9999

        target_matrix = jnp.stack([gpp_target, gpp_target_mask, nee_target, nee_target_mask,
                                   et_target, et_target_mask, lai_target, lai_target_mask,
                                   som_target, som_target_mask, som1_target, som1_target_mask,
                                   som2_target, som2_target_mask, som3_target, som3_target_mask,
                                   reco_target, reco_target_mask]).T
    else:
        target_matrix = jnp.stack([gpp_target, gpp_target_mask, nee_target, nee_target_mask,
                                   et_target, et_target_mask, lai_target, lai_target_mask,
                                   som_target, som_target_mask, som1_target, som1_target_mask,
                                   som2_target, som2_target_mask, som3_target, som3_target_mask]).T
    if train_mode:
        return target_matrix[train_sel, :]
    else:
        return target_matrix
