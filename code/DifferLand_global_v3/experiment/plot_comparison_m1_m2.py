# 9/26/2025
# Fayong Liu
# This file is for comparing the results of two ET modules, wue v.s. pml2

import pickle
import os
import sys
import warnings
import numpy as np

sys.path.insert(1, '../')
from DifferLand.util.visualization import (plot_global_pattern_difference, plot_scatter,
                                           plot_temporal_series, plot_temporal_performance_grid)
from DifferLand.util.normalization import unnormalize_parameters
sys.path.insert(1, '/data1/fyliu/')
from ManualFunction.plotprocess import fig_basic_param

warnings.filterwarnings("ignore", category=RuntimeWarning)

# basic set
fig_basic_param(font_sz_=12, ax_title_sz_=14, ax_lab_sz_=12, ax_linewidth_=1, xy_tick_lab_sz_=10, xy_tick_major_sz_=2,
                leg_fontsz_=12)

# def a function to compare the results from two different ET module, 1v1
def compare1v1(result_dir, result_name, result_dir1, result_name1, var_index, num_var, fig_dir_base, metric='R2'):
    result_path = os.path.join(result_dir, result_name)
    with open(result_path, 'rb') as fp:
        result_ls = pickle.load(fp)
        results, invariant_obs, variant_obs, loss_ls, mask_ls, lon, lat, init_pools, init_params = result_ls
        results = results[:, :, var_index]
        LMA = init_params[:, 0]
        LMA = unnormalize_parameters(LMA, 10, 400)
        train_mask, develop_mask, test_mask, valid_mask = mask_ls

    result_path = os.path.join(result_dir1, result_name1)
    with open(result_path, 'rb') as fp:
        result_ls = pickle.load(fp)
        results1, invariant_obs, variant_obs, loss_ls, mask_ls, lon, lat, init_pools1, init_params1 = result_ls
        results1 = results1[:, :, var_index]
        LMA1 = init_params1[:, 0]
        LMA1 = unnormalize_parameters(LMA1, 10, 400)
        train_mask, develop_mask, test_mask, valid_mask = mask_ls

    test_mask = None
    fig_dir = os.path.join(fig_dir_base, os.path.splitext(result_name1)[0])
    var_name_ls = ['GPP', 'LAI', 'RECO', 'ET', 'SOC', 'Rh', 'LMA']
    var_unit_ls = ['g C/m$^2$/d', 'm$^2$/m$^2$', 'g C/m$^2$/d', 'mm/day', 'g C/m$^2$', 'g C/m$^2$/d', 'g C/m$^2$']
    obs_ar_all = np.nanmean(variant_obs, axis=1)
    obs_ar_all = np.hstack([obs_ar_all, invariant_obs])
    mod_ar_all = np.nanmean(results, axis=1)
    mod_ar_all = np.hstack([mod_ar_all, LMA.reshape(-1, 1)])
    mod_ar_all1 = np.nanmean(results1, axis=1)
    mod_ar_all1 = np.hstack([mod_ar_all1, LMA1.reshape(-1, 1)])
    category = 3
    for i in range (len(var_name_ls)):
        # plot spatial pattern
        mod_var_ar = mod_ar_all[:, i]
        obs_var_ar = obs_ar_all[:, i]
        mod_var_ar1 = mod_ar_all1[:, i]
        var_name = var_name_ls[i]
        var_unit = var_unit_ls[i]
        plot_global_pattern_difference(mod_var_ar, obs_var_ar, var_name, var_unit, category, valid_mask, lon, lat,
                                       fig_dir, mod_var_ar1)
        if i < num_var:
            mod_var_ar = results[:, :, i]
            mod_var_ar1 = results1[:, :, i]
            obs_var_ar = variant_obs[:, :, i]
            # plot temporal series
            # originla
            plot_temporal_series(mod_var_ar, obs_var_ar, var_name, var_unit, valid_mask, lon, lat,
                                 metric, False, 3, fig_dir, test_mask, mod_var_ar1)
            # anomaly = True
            plot_temporal_series(mod_var_ar, obs_var_ar, var_name, var_unit, valid_mask, lon, lat,
                                 metric, True, 3, fig_dir, test_mask, mod_var_ar1)

    # plot scatter
    plot_scatter(mod_ar_all, obs_ar_all, var_name_ls, category, test_mask, fig_dir, metric, mod_ar_all1)
    # plot grid performance
    mod_ar_all = results[:, :, :num_var]
    mod_ar_all1 = results1[:, :, :num_var]
    obs_ar_all = variant_obs
    plot_temporal_performance_grid(mod_ar_all, obs_ar_all, valid_mask, lon, lat, metric, fig_dir,
                                   test_mask, mod_ar_all1)

    return


def main():
    fig_dir_base = r'../../../fig/comparing'
    # load data1, wue module
    result_dir = r'../../../output/obs_et1'
    result_name = 'onepool_wue_original_all_time_ts_30_bs_400_lr_0.0005_epo_1000_it_1_db_0_r_42_i_2.pkl'
    # load data2, pml2 module
    result_dir1 = r'../../../output/obs_et2'
    result_name1 = 'onepool_pml2_original_all_time_ts_30_bs_400_lr_0.0005_epo_1000_it_1_db_0_r_42_i_3.pkl'

    var_index = [14, 20, 17, 12, 7, 19]
    num_var = 4
    compare1v1(result_dir, result_name, result_dir1, result_name1, var_index, num_var, fig_dir_base, metric='R2')

    return


if __name__ == "__main__":
    main()








