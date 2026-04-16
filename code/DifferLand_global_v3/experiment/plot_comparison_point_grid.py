# Code for comparison between point/grid-based observations and modeling results

import os
import pickle
import sys
import xarray as xr
import jax.numpy as jnp
from calibration import TIME_STEP, REGION
sys.path.insert(1, '../')
from DifferLand.util.visualization import plot_scatter, plot_global_pattern_difference, cal2polt_global
from DifferLand.util.load_data import load_time_invariant_data


def main():

    # Get point-based SOC and MAOC and modeling results
    mask = None
    metric = 'R2'

    result_path = r'../../../output/soc3_ge2/mesc_pml2_all_time_ts_30_bs_400_lr_0.0005_epo_400_it_1_r_42_unc_0_p2g_1_i_1.pkl'
    with open(result_path, 'rb') as fp:
        result_ls = pickle.load(fp)
    
    (_, _, _, mod_ar_all, obs_ar_all, grid_weights, var_name_ls, 
     var_unit_ls, new_test_mask, valid_mask, lon, lat, _) = cal2polt_global(result_ls, mask)

    var_name_ls_soil = ['SOC', 'SOC2']
    idx_ls = [var_name_ls.index(name) for name in var_name_ls_soil]
    mod_ar_soil = mod_ar_all[:, idx_ls]
    obs_ar_soil = obs_ar_all[:, idx_ls]
    grid_weights_soil = grid_weights[:, idx_ls]
    var_unit_ls_soil = [var_unit_ls[idx] for idx in idx_ls]

    # Get grid-based SOC and MAOC
    base_dir = r'/data1/fyliu/DifferLand_forcing/observation_monthly_05deg/'
    var_name_ls_grid = ['SOC_0.5deg_0_30cm', 'MAOC_0.5deg_0_100cm']
    invariant, _ = load_time_invariant_data(base_dir, var_name_ls_grid, False, TIME_STEP, REGION)
    obs_ar_soil1 = invariant[valid_mask]

    fig_dir = r'../../../fig'

    # Plot scatter
    plot_scatter(mod_ar_soil, obs_ar_soil, grid_weights_soil, var_name_ls_soil, var_unit_ls_soil, 
                 new_test_mask, fig_dir, metric, obs_ar_soil1)
    
    # Plot spatial pattern
    for i in range (len(var_name_ls_soil)):
        mod_var_ar = mod_ar_soil[:, i]
        obs_var_ar = obs_ar_soil[:, i]
        var_name = var_name_ls_soil[i]
        var_unit = var_unit_ls_soil[i]
        grid_weight = grid_weights_soil[:, i]
        mod_var_ar1 = obs_ar_soil1[:, i]
        plot_global_pattern_difference(mod_var_ar, obs_var_ar, var_name, var_unit, 
                                       valid_mask, lon, lat, grid_weight, fig_dir, mod_var_ar1)
    
    return


if __name__ == "__main__":
    main()

