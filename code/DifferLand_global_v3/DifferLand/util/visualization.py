# Function for visulization


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.ticker as mticker
import os
import numpy as np
import xarray as xr
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
import warnings
from scipy.stats import linregress
import sys

sys.path.insert(1, '../')
from DifferLand.model.DALEC990_info import (param_info, pool_info, target_info, target_unit)
from DifferLand.util.load_data import restore_to_original_length
from DifferLand.util.normalization import unnormalize

batched_unnormalize = jax.vmap(jax.jit(unnormalize), in_axes=([0, None, None]))

warnings.filterwarnings("ignore", category=RuntimeWarning)

def adjust_axis_label(axname, lat_min, lat_max, lon_min, lon_max):
    """ Function to adjust axes tick and ticklabels """

    axname.set_xlim(lon_min, lon_max)
    axname.set_ylim(lat_min, lat_max)
    ticks_x = np.arange(lon_min, lon_max, int((lon_max - lon_min) / 6))
    labels_x = [f'{abs(int(tick))}\xb0{"E" if tick > 0 else "W" if tick < 0 else " "}' for tick in ticks_x]
    labels_x[0] = ' '
    axname.set_xticks(ticks_x)
    axname.set_xticklabels(labels_x)
    ticks_y = np.arange(lat_min, lat_max, int((lat_max - lat_min) / 6))
    labels_y = [f'{abs(tick)}\xb0{"N" if tick > 0 else "S" if tick < 0 else " "}' for tick in ticks_y]
    axname.set_yticks(ticks_y)
    axname.set_yticklabels(labels_y)
    axname.set_xlabel(None)
    axname.set_ylabel(None)
    axname.set_title('')

    return axname


def add_fignum(axname, txt):
    """ Function to set fig num """

    xlim = axname.get_xlim()
    ylim = axname.get_ylim()
    xpos = xlim[0]  # (xlim[0] + xlim[1]) / 2
    ypos = ylim[1]
    axname.text(xpos, ypos, txt, va='bottom', ha='left', fontweight='bold')

    return axname


def plot_global_pattern(var_ar, var_name, var_unit, ismod, valid_mask, region, lon, lat, fig_dir=None):
    """
    Function to plot global pattern

    :param var_ar: np.array/jnp.array, (n_grid, );
    :param var_name: str;
    :param var_unit: str;
    :param ismod: bool;
    :param valid_mask: bool array;

    """

    var_ar = np.array(var_ar)
    fig = plt.figure(figsize=(8, 3.5), dpi=300)
    grid = GridSpec(1, 1, figure=fig, left=0.08, bottom=0.07, right=0.75, top=0.93, wspace=0, hspace=0)
    axname = plt.subplot(grid[0, 0])

    # set norm & cmap
    notnan_values = var_ar[~np.isnan(var_ar)]
    vmin, vmax = np.percentile(notnan_values, 1), np.percentile(notnan_values, 99)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)
    cmap = plt.get_cmap('Blues')

    # convert result to xarray
    var_ds = restore_to_original_length(var_ar, valid_mask, fill_value=np.nan)
    var_ds = var_ds.reshape(len(lat), len(lon))
    var_ds = xr.DataArray(var_ds, coords=[lat, lon], dims=["lat", "lon"])
    var_ds['lon'] = xr.where(var_ds['lon'] > 180, var_ds['lon'] - 360, var_ds['lon'])
    im = var_ds.plot(ax=axname, cmap=cmap, norm=norm, add_colorbar=False)
    lat_min, lat_max = region.get("lat_min", -90), region.get("lat_max", 90)
    lon_min, lon_max = region.get("lon_min", -180), region.get("lon_max", 180)
    axname = adjust_axis_label(axname, lat_min, lat_max, lon_min, lon_max)
    txt = f'Global pattern of {var_name}'
    axname = add_fignum(axname, txt)

    # add lat stats
    bin_edges = [-60, -50, -23.5, 23.5, 50, 80]
    bin_centers = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)])
    bin_height = np.array([(bin_edges[i + 1] - bin_edges[i]) for i in range(len(bin_edges) - 1)])
    clr1 = 'steelblue'
    clr2 = 'dimgray'
    lat_mean = var_ds.mean(dim='lon', skipna=True)
    lat_std = var_ds.std(dim='lon', skipna=True)
    ax_lat = fig.add_axes(
        [axname.get_position().x1, axname.get_position().y0, 0.1, axname.get_position().height]
    )
    ax_lat.plot(lat_mean, lat_mean.lat, color=clr1, linewidth=1.5, alpha=0.5)
    ax_lat.fill_betweenx(lat_mean.lat, lat_mean + lat_std, lat_mean - lat_std, color=clr1, alpha=0.3)
    new_xlim = ax_lat.get_xlim()
    ax_lat.set_xlim([0, new_xlim[1]])

    # add temperature zonal stats
    bin_values = []
    for j in range(len(bin_edges) - 1):
        reg_mask = var_ds.sel(lat=slice(bin_edges[j], bin_edges[j + 1]))
        reg_mean = reg_mask.mean(skipna=True).item()
        bin_values.append(reg_mean)
    ax_lat.barh(bin_centers, bin_values, bin_height, color='none', edgecolor=clr2)
    ax_lat.set_ylim(-60, 80)
    ax_lat.set_yticks(np.array([-50, -23.5, 0, 23.5, 50]))
    ax_lat.set_yticklabels([])
    ax_lat.set_xlabel(None)
    ax_lat.set_ylabel(None)

    # add colorbars
    lab = var_name if var_unit == '' else var_name + ' (' + var_unit + ')'
    cbar_ax = fig.add_axes(
        [ax_lat.get_position().x1 + 0.02, ax_lat.get_position().y0, 0.03, ax_lat.get_position().height]
    )
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', extend='both')
    cbar.set_label(lab)

    # save result
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        if ismod:
            fig_name = f'Global_pattern_{var_name}_mod.png'
        else:
            fig_name = f'Global_pattern_{var_name}_obs.png'
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path, dpi=300)
        print(f'Saving {fig_path}')
    plt.close()

    return


def plot_global_pattern_biome_distribution(var_ar, var_name, var_unit, ismod, valid_mask, lon, lat,
                                           fig_dir=None):
    """ Function to plot global pattern of each param, pool, observation, or model output, add zonal distribution at each biome """

    biome_ls = [
        "(b) Global", "(c) TroStMBF", "(d) TroStDBF", "(e) TroStCF", "(f) TemBMF",
        "(g) TemCF", "(h) BorF", "(i) TroStGSS", "(j) TemGSS", "(k) FloGS",
        "(l) MonGS", "(m) Tundra", "(n) MedFWS", "(o) DXerShrub", "(p) Mangroves",
        "(q) Lakes", "(r) RockIce"
    ]
    biome_id_ls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 98, 99]
    color_ls = [
        "#999999", "#266b5e", "#58c24b", "#20627d", "#e6e978",
        "#66b49e", "#94dadb", "#f8d03d", "#fcb053", "#efc3a0",
        "#a1763c", "#487fc4", "#dd3232", "#d7a6d7", "#a64fa5", "#4ed6fa", "#cfcfcf"
    ]
    biome_file = r"/data1/fyliu/e_soil_carbon_model/data/biome/Biome_0.5deg.nc"
    biome_ds = xr.open_dataset(biome_file)
    biome_ds = biome_ds.sel(lat=slice(-60, 80))
    biome_values = biome_ds['Biome'].values.ravel()
    valid_biome = biome_values[valid_mask]

    var_ar = np.array(var_ar)
    fig = plt.figure(figsize=(8, 12), dpi=300)
    grid = GridSpec(12, 8, figure=fig, left=0.08, bottom=0.04, right=0.98, top=0.98, wspace=0.1, hspace=0.65)
    ax1 = plt.subplot(grid[0:3, 0:6])
    ax_ls = [plt.subplot(grid[i, j:j + 4]) for i in range(3, 12) for j in [0, 4]]
    fig.delaxes(ax_ls[1])
    del ax_ls[1]

    # plot global patterns
    # ====================================================================
    notnan_values = var_ar[~np.isnan(var_ar)]
    vmin, vmax = np.percentile(notnan_values, 1), np.percentile(notnan_values, 99)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)
    cmap = plt.get_cmap('Blues')
    axname = ax1

    # convert result to xarray
    var_ds = restore_to_original_length(var_ar, valid_mask, fill_value=np.nan)
    var_ds = var_ds.reshape(len(lat), len(lon))
    var_ds = xr.DataArray(var_ds, coords=[lat, lon], dims=["lat", "lon"])
    var_ds['lon'] = xr.where(var_ds['lon'] > 180, var_ds['lon'] - 360, var_ds['lon'])
    im = var_ds.plot(ax=axname, cmap=cmap, norm=norm, add_colorbar=False)
    axname = adjust_axis_label(axname)
    txt = f'(a) Global pattern of {var_name}'
    axname = add_fignum(axname, txt)

    # add lat stats
    bin_edges = [-60, -50, -23.5, 23.5, 50, 80]
    bin_centers = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)])
    bin_height = np.array([(bin_edges[i + 1] - bin_edges[i]) for i in range(len(bin_edges) - 1)])
    clr1 = 'steelblue'
    clr2 = 'dimgray'
    lat_mean = var_ds.mean(dim='lon', skipna=True)
    lat_std = var_ds.std(dim='lon', skipna=True)
    ax_lat = fig.add_axes(
        [axname.get_position().x1, axname.get_position().y0, 0.1, axname.get_position().height]
    )
    ax_lat.plot(lat_mean, lat_mean.lat, color=clr1, linewidth=1.5, alpha=0.5)
    ax_lat.fill_betweenx(lat_mean.lat, lat_mean + lat_std, lat_mean - lat_std, color=clr1, alpha=0.3)
    new_xlim = ax_lat.get_xlim()
    ax_lat.set_xlim([0, new_xlim[1]])

    # add temperature zonal stats
    bin_values = []
    for j in range(len(bin_edges) - 1):
        reg_mask = var_ds.sel(lat=slice(bin_edges[j], bin_edges[j + 1]))
        reg_mean = reg_mask.mean(skipna=True).item()
        bin_values.append(reg_mean)
    ax_lat.barh(bin_centers, bin_values, bin_height, color='none', edgecolor=clr2)
    ax_lat.set_ylim(-60, 80)
    ax_lat.set_yticks(np.array([-50, -23.5, 0, 23.5, 50]))
    ax_lat.set_yticklabels([])
    ax_lat.set_xlabel(None)
    ax_lat.set_ylabel(None)

    # add colorbars
    lab = var_name if var_unit == '' else var_name + ' (' + var_unit + ')'
    cbar_ax = fig.add_axes(
        [ax_lat.get_position().x1 + 0.02, ax_lat.get_position().y0, 0.03, ax_lat.get_position().height]
    )
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', extend='both')
    cbar.set_label(lab)

    # plot biome distribution
    # ====================================================================
    for j in range(len(ax_ls)):
        axname = ax_ls[j]
        if j == 0:
            reg_mask = ~np.isnan(valid_biome)
        else:
            reg_mask = valid_biome == biome_id_ls[j - 1]
        biome_var = var_ar[reg_mask]
        sns.kdeplot(biome_var, ax=axname, color=color_ls[j], linewidth=2, fill=True, alpha=0.3)
        mean_value = np.mean(biome_var)
        std_value = np.std(biome_var)
        axname.axvline(mean_value, color=color_ls[j], linestyle="--", linewidth=1.5)
        for spine in axname.spines.values():
            spine.set_visible(False)
        if j < 15:
            axname.set_xticks([])
        axname.set_yticks([])
        axname.set_ylabel('')
        if j == 0:
            new_xlim = axname.get_xlim()
        else:
            axname.set_xlim(new_xlim[0], new_xlim[1])
        if mean_value >= 1000:
            str_mean = str(np.around(mean_value, 0))
            str_std = str(np.around(std_value, 0))
        elif mean_value >= 100:
            str_mean = str(np.around(mean_value, 1))
            str_std = str(np.around(std_value, 1))
        else:
            str_mean = str(np.around(mean_value, 2))
            str_std = str(np.around(std_value, 2))
        txt = f'{biome_ls[j]}, {str_mean}\u00B1{str_std}'
        axname = add_fignum(axname, txt)
    for ax in ax_ls[-2:]:
        ax.spines["bottom"].set_visible(True)
        ax.set_xlabel(lab)

    # save result
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        if ismod:
            fig_name = f'Global_pattern_biome_distribution_{var_name}_mod.png'
        else:
            fig_name = f'Global_pattern_biome_distribution_{var_name}_obs.png'
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path, dpi=300)
        print(f'Saving {fig_path}')
    plt.close()

    return


def get_bin_edges(lat_min, lat_max, ref_edges):
   
    inner_edges = [e for e in ref_edges if lat_min < e < lat_max]
    bin_edges = [lat_min] + inner_edges + [lat_max]
    bin_edges = sorted(set(bin_edges))
    
    return bin_edges


def plot_global_pattern_difference(mod_var_ar, obs_var_ar, var_name, var_unit, valid_mask, region, lon, lat, 
                                   grid_weight=None, fig_dir=None, mod_var_ar1=None):
    """ Function to plot global pattern of difference between mod and obs """

    if var_name == 'SOC1':
        var_name = 'POC'
    elif var_name == 'SOC2':
        var_name = 'MAOC'
    mod_var_ar = np.array(mod_var_ar)
    obs_var_ar = np.array(obs_var_ar)

    if mod_var_ar1 is not None:
        mod_var_ar1 = np.array(mod_var_ar1)
        fig = plt.figure(figsize=(16, 12), dpi=300)
        grid = GridSpec(3, 2, figure=fig, left=0.05, bottom=0.1, right=0.94, top=0.97, wspace=0.26, hspace=0.17)
        ax_ls = [plt.subplot(grid[i, j]) for j in range(2) for i in range(3)]
    else:
        fig = plt.figure(figsize=(9, 10), dpi=300)
        bottom = 0.03
        grid = GridSpec(3, 1, figure=fig, left=0.07, bottom=bottom, right=0.77, top=0.98, wspace=0, hspace=0.18)
        ax_ls = [plt.subplot(grid[j, 0]) for j in range(3)]

    all_values = np.concatenate(
        [mod_var_ar, mod_var_ar1, obs_var_ar]).flatten() if mod_var_ar1 is not None else np.concatenate(
        [mod_var_ar, obs_var_ar]).flatten()
    notnan_values = all_values[~np.isnan(all_values)]
    vmax1 = np.percentile(notnan_values, 99)
    norm1 = TwoSlopeNorm(vmin=0, vcenter=vmax1 / 2, vmax=vmax1)
    cmap1 = plt.get_cmap('Blues')

    if mod_var_ar1 is not None:
        all_values = np.concatenate(
            [mod_var_ar - obs_var_ar, mod_var_ar1 - obs_var_ar, mod_var_ar - mod_var_ar1]).flatten()
        notnan_values = all_values[~np.isnan(all_values)]
        vmin2, vmax2 = np.percentile(notnan_values, 1), np.percentile(notnan_values, 99)
        var_ar_ls = [mod_var_ar, mod_var_ar1, obs_var_ar, mod_var_ar - obs_var_ar, mod_var_ar1 - obs_var_ar,
                     mod_var_ar - mod_var_ar1]
        pnum_ls = ['(a) M1', '(b) M2', '(c) Observation', '(d) M1 - Observation',
                   '(e) M2 - Observation', '(f) M1 - M2']
    else:
        all_values = (mod_var_ar - obs_var_ar).flatten()
        notnan_values = all_values[~np.isnan(all_values)]
        vmin2, vmax2 = np.percentile(notnan_values, 1), np.percentile(notnan_values, 99)
        var_ar_ls = [mod_var_ar, obs_var_ar, mod_var_ar - obs_var_ar]
        pnum_ls = ['(a) Model', '(b) Observation', '(c) Model - Observation']
    if vmin2 < 0 < vmax2:
        norm2 = TwoSlopeNorm(vmin=vmin2, vcenter=0, vmax=vmax2)
        cmap2 = plt.get_cmap('coolwarm')
    elif vmax2 <= 0:
        norm2 = TwoSlopeNorm(vmin=vmin2, vcenter=vmin2 / 2, vmax=0)
        cmap2 = LinearSegmentedColormap.from_list(
            "warm_coolwarm", plt.get_cmap("coolwarm")(np.linspace(0, 0.5, 256))
        )
    elif vmin2 >= 0:
        norm2 = TwoSlopeNorm(vmin=0, vcenter=vmax2 / 2, vmax=vmax2)
        cmap2 = LinearSegmentedColormap.from_list(
            "warm_coolwarm", plt.get_cmap("coolwarm")(np.linspace(0.5, 1, 256))
        )

    # zonal statistic
    bin_edges_ref = [-60, -50, -23.5, 23.5, 50, 80]
    lat_min, lat_max = region.get("lat_min", -90), region.get("lat_max", 90)
    lon_min, lon_max = region.get("lon_min", -180), region.get("lon_max", 180)
    bin_edges = get_bin_edges(lat_min, lat_max, bin_edges_ref)
    bin_centers = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)])
    bin_height = np.array([(bin_edges[i + 1] - bin_edges[i]) for i in range(len(bin_edges) - 1)])
    clr1 = 'steelblue'
    clr2 = 'dimgray'
    differ_idx = 3 if mod_var_ar1 is not None else 2
    if grid_weight is not None:
        weight_ds = restore_to_original_length(grid_weight, valid_mask, fill_value=np.nan)
        weight_ds = weight_ds.reshape(len(lat), len(lon))
        weight_ds = xr.DataArray(weight_ds, coords=[lat, lon], dims=["lat", "lon"])
        weight_ds['lon'] = xr.where(weight_ds['lon'] > 180, weight_ds['lon'] - 360, weight_ds['lon'])
    for j in range(len(var_ar_ls)):
        axname = ax_ls[j]
        # convert result to xarray
        var_ds = restore_to_original_length(var_ar_ls[j], valid_mask, fill_value=np.nan)
        var_ds = var_ds.reshape(len(lat), len(lon))
        var_ds = xr.DataArray(var_ds, coords=[lat, lon], dims=["lat", "lon"])
        var_ds['lon'] = xr.where(var_ds['lon'] > 180, var_ds['lon'] - 360, var_ds['lon'])
        if grid_weight is not None:
            var_ds = var_ds.where(weight_ds>0, np.nan)
        if j < differ_idx:
            im1 = var_ds.plot(ax=axname, cmap=cmap1, norm=norm1, add_colorbar=False)
            txt = pnum_ls[j] + ' (' + var_name + ')'
        else:
            im2 = var_ds.plot(ax=axname, cmap=cmap2, norm=norm2, add_colorbar=False)
            global_mean = var_ds.mean(skipna=True).item()
            global_std = var_ds.std(skipna=True).item()
            if var_unit == '':
                txt = pnum_ls[j] + ' (' + str(round(global_mean, 2)) + '\u00b1' + str(round(global_std, 2)) + ')'
            else:
                txt = pnum_ls[j] + ' (' + str(round(global_mean, 2)) + '\u00b1' + str(round(global_std, 2)) + ' ' + \
                      var_unit + ')'
        axname = adjust_axis_label(axname, lat_min, lat_max, lon_min, lon_max)
        axname = add_fignum(axname, txt)

        # add lat stats
        lat_mean = var_ds.mean(dim='lon', skipna=True)
        lat_std = var_ds.std(dim='lon', skipna=True)
        ax_lat = fig.add_axes(
            [axname.get_position().x1, axname.get_position().y0, 0.05, axname.get_position().height]
        )
        ax_lat.plot(lat_mean, lat_mean.lat, color=clr1, linewidth=1.5, alpha=0.5)
        ax_lat.fill_betweenx(lat_mean.lat, lat_mean + lat_std, lat_mean - lat_std, color=clr1, alpha=0.3)
        xlim = ax_lat.get_xlim()
        if j < differ_idx:
            ax_lat.set_xlim([0, xlim[1]])

        # add temperature zonal stats
        bin_values = []
        for j in range(len(bin_edges) - 1):
            reg_mask = var_ds.sel(lat=slice(bin_edges[j], bin_edges[j + 1]))
            reg_mean = reg_mask.mean(skipna=True).item()
            bin_values.append(reg_mean)
        ax_lat.barh(bin_centers, bin_values, bin_height, color='none', edgecolor=clr2)
        ax_lat.set_ylim(lat_min, lat_max)
        ticks_y = np.arange(lat_min, lat_max, int((lat_max - lat_min) / 6))
        ax_lat.set_yticks(ticks_y)
        ax_lat.set_yticklabels([])
        ax_lat.set_xlabel(None)
        ax_lat.set_ylabel(None)

    # add colorbars
    lab = var_name if var_unit == '' else var_name + ' (' + var_unit + ')'
    if mod_var_ar1 is not None:
        cbar_ax1 = fig.add_axes([0.05, 0.04, 0.44, 0.02])
        cbar_ax2 = fig.add_axes([0.55, 0.04, 0.44, 0.02])
        orientation = 'horizontal'
    else:
        left = 0.88
        width = 0.02
        height = 0.29
        cbar_ax1 = fig.add_axes([left, 0.52, width, height])
        cbar_ax2 = fig.add_axes([left, bottom, width, height])
        orientation = 'vertical'
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation=orientation, extend='both')
    cbar1.set_label(lab)
    cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation=orientation, extend='both')
    cbar2.set_label(f'Difference in {lab}')
    if vmin2 < 0 < vmax2:
        t1 = np.ceil(np.linspace(vmin2, 0, num=3))
        t2 = np.floor(np.linspace(0, vmax2, num=3))
        tick = np.hstack((t1, t2[1:]))
        cbar2.set_ticks(tick)

    # save result
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f'Global_pattern_difference_{var_name}.png'
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path, dpi=300)
        print(f'Saving {fig_path}')
    plt.close()

    return


def plot_scatter(mod_ar_all, obs_ar_all, grid_weights, var_name_ls, var_unit_ls, 
                 test_mask=None, fig_dir=None, metric='R2', mod_ar_all1=None):
    """ Function to plot scatter """

    metric_lab = 'R$^2$' if metric == 'R2' else metric

    n_plots = len(var_name_ls)
    # Fixed number of columns
    ncols = 3
    # Calculate number of rows
    nrows = (n_plots + ncols - 1) // ncols
    # Calculate figure size based on number of rows
    figsize = (4*ncols, 4*nrows)
    # Create figure and grid
    fig = plt.figure(figsize=figsize, dpi=300)
    grid = GridSpec(nrows, ncols, figure=fig, left=0.05, bottom=0.12, right=0.98, top=0.94, wspace=0.22, hspace=0.25)
    # Create all subplots
    ax_ls = [plt.subplot(grid[i, j]) for i in range(nrows) for j in range(ncols)]
    # Remove unused subplots
    for i in range(n_plots, len(ax_ls)):
        fig.delaxes(ax_ls[i])

    # plot scatter
    alp = 0.2
    clr = ['steelblue', 'maroon', '#C9C780']
    pattern_index_ls = []
    for i in range(len(var_name_ls)):
        axname = ax_ls[i]
        mod = np.array(mod_ar_all[:, i])
        obs = np.array(obs_ar_all[:, i])
        mask_mod = mod[test_mask] if test_mask is not None else mod
        mask_obs = obs[test_mask] if test_mask is not None else obs

        weight_mask = grid_weights[:, i]
        weight_mask = weight_mask[test_mask] if test_mask is not None else weight_mask
        mask_mod = mask_mod[weight_mask > 0]
        mask_obs = mask_obs[weight_mask > 0]

        if metric == 'R2':
            index = r2_score(mask_obs, mask_mod)
        elif metric == 'NNSE':
            index = r2_score(mask_obs, mask_mod)
            index = 1 / (2 - index)
        elif metric == 'r':
            index = np.corrcoef(mask_obs, mask_mod)[0, 1]

        this_lab = metric_lab + '=' + str(np.around(index, 2))
        pattern_index_ls.append(index)

        if mod_ar_all1 is not None:
            mod1 = np.array(mod_ar_all1[:, i])
            mask_mod1 = mod1[test_mask] if test_mask is not None else mod1
            mask_mod1 = mask_mod1[weight_mask > 0]
            valid_idx = ~(np.isnan(mask_obs) | np.isnan(mask_mod1))
            mask_obs = mask_obs[valid_idx]
            mask_mod = mask_mod[valid_idx]
            mask_mod1 = mask_mod1[valid_idx]
            if metric == 'R2':
                index1 = r2_score(mask_obs, mask_mod1)
            elif metric == 'NNSE':
                index1 = r2_score(mask_obs, mask_mod1)
                index1 = 1 / (2 - index1)
            elif metric == 'r':
                index1 = np.corrcoef(mask_obs, mask_mod1)[0, 1]
            this_lab = 'M1 ' + metric_lab + ' = ' + str(np.around(index, 2))
            this_lab1 = 'M2 ' + metric_lab + ' = ' + str(np.around(index1, 2))
            axname.scatter(mask_obs, mask_mod1, color=clr[2], s=5, label=this_lab1, alpha=alp)
            pattern_index_ls.append(index1)
        axname.scatter(mask_obs, mask_mod, color=clr[0], s=5, label=this_lab, alpha=alp)
        # add 1:1 line
        xlim = axname.get_xlim()
        axname.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], color=clr[1],
                    linestyle='--', linewidth=2, label='1:1 Line', alpha=0.4)
        axname.set_xlabel(f'Observed ({var_unit_ls[i]})')
        axname.set_ylabel(f'Modeled ({var_unit_ls[i]})')
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 3))
        axname.xaxis.set_major_formatter(formatter)
        axname.yaxis.set_major_formatter(formatter)
        axname.legend(ncol=1, loc='upper left', frameon=False)
        title = var_name_ls[i]
        if title == 'SOC1':
            title = 'POC'
        elif title == 'SOC2':
            title = 'MAOC'
        axname.set_title(title, fontweight='bold')

    # save result
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f'Global_pattern_scatter.png'
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path, dpi=300)
        print(f'Saving {fig_path}')
    plt.close()

    return pattern_index_ls


def cal_monthly_interannual_anomalies(dataarray, time_series):
    """ Function to calculate monthly interannual anomalies """

    months = time_series.month
    # cal monthly average
    monthly_mean = np.array([np.nanmean(dataarray[months == m]) for m in range(1, 13)])

    # cal monthly anomaly
    anomaly_array = np.copy(dataarray)
    for m in range(1, 13):
        anomaly_array[months == m] = dataarray[months == m] - monthly_mean[m - 1]

    return anomaly_array


def plot_temporal_series(mod_var_ar, obs_var_ar, var_name, var_unit, valid_mask, region, lon, lat,
                         metric='R2', anomaly=False, zones=3, fig_dir=None, test_mask=None, mod_var_ar1=None):
    """
    Function to plot temporal series

    :param mod_var_ar: modeled time series, (n_grids, n_days)
    :param obs_var_ar: observed time series, (n_girds, n_days)
    :param var_name: str, variable name
    :param var_unit: str, variable unit
    :param valid_mask: bool, valid mask
    :param lon: 1d array, longitude
    :param lat: 1d array, latitude
    :param metric: str, R2, r or nnse
    :param anomaly: bool, if true, plotting monthly interannual anomaly series, else plotting original series
    :param zones: int, if == 1, global, tropical, temperate, boreal
    elif == 2, tropical, temperate, boreal, north and south seperately
    elif == 3, 11 reccap zones
    :param fig_dir: str, figure directory
    :param test_mask: bool, if test_mask is not None, only use testing samples
    :param mod_var_ar1: modeled time series, (n_grids, n_days), if not None, plot multiple modeled time series

    """

    mod_var_ar = np.array(mod_var_ar)
    obs_var_ar = np.array(obs_var_ar)
    mod_var_ar1 = np.array(mod_var_ar1) if mod_var_ar1 is not None else None

    reccap_file = r"/data1/fyliu/DifferLand_forcing/RECCAP/Reccap_0.5deg.nc"
    lat_min, lat_max = region.get("lat_min", -90), region.get("lat_max", 90)
    lon_min, lon_max = region.get("lon_min", -180), region.get("lon_max", 180)
    reccap_ds = xr.open_dataset(reccap_file)
    reccap_ds = reccap_ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    reccap_values = reccap_ds['Region_Map'].values.ravel()

    metric_lab = 'R$^2$' if metric == 'R2' else metric
    lat_values = np.tile(lat[:, None], (1, len(lon))).ravel()
    valid_lat = lat_values[valid_mask]
    valid_reccap = reccap_values[valid_mask]

    start_time = '2001-01'
    num_time_steps = mod_var_ar.shape[1]
    time_series = pd.date_range(start=start_time, periods=num_time_steps, freq='ME')
    index_ls = []
    linewd = 1.5
    lab = var_name if var_unit == '' else var_name + ' (' + var_unit + ')'

    # global, tropical, temperate, boreal
    if zones == 1:
        fig = plt.figure(figsize=(12, 6), dpi=300)
        grid = GridSpec(2, 2, figure=fig, left=0.07, bottom=0.05, right=0.99, top=0.96, wspace=0.16, hspace=0.27)
        ax_ls = [plt.subplot(grid[j, k]) for j in range(2) for k in range(2)]
        txt_ls = ['(a) Global', '(b) Tropical', '(c) Temperate', '(d) Boreal']
        reg_ls = [0, 23.5, 50, 90]
    elif zones == 2:
        fig = plt.figure(figsize=(12, 9), dpi=300)
        grid = GridSpec(3, 2, figure=fig, left=0.07, bottom=0.05, right=0.99, top=0.96, wspace=0.16, hspace=0.27)
        ax_ls = [plt.subplot(grid[j, k]) for j in range(3) for k in range(2)]
        txt_ls = ['(a) Boreal (S)', '(b) Temperate (S)', '(c) Tropical (S)', '(d) Tropical (N)',
                  '(e) Temperate (N)', '(f) Boreal (N)']
        reg_ls = [-90, -50, -23.5, 0, 23.5, 50, 90]
    elif zones == 3:
        fig = plt.figure(figsize=(18, 12), dpi=300)
        grid = GridSpec(4, 3, figure=fig, left=0.05, bottom=0.04, right=0.99, top=0.97, wspace=0.16, hspace=0.27)
        ax_ls = [plt.subplot(grid[j, k]) for j in range(4) for k in range(3)]
        txt_ls = ['(a) Global', '(b) North America', '(c) South America', '(d) Europe', '(e) North Africa',
                  '(f) South Africa', '(g) Russia', '(h) West Asia', '(i) East Asia', '(j) South Asia',
                  '(k) Southeast Asia', '(l) Australasia']

    for j in range(len(txt_ls)):
        axname = ax_ls[j]
        mod_mask = mod_var_ar[test_mask, :] if test_mask is not None else mod_var_ar
        obs_mask = obs_var_ar[test_mask, :] if test_mask is not None else obs_var_ar
        lat_mask = valid_lat[test_mask] if test_mask is not None else valid_lat
        reccap_mask = valid_reccap[test_mask] if test_mask is not None else valid_reccap

        if zones == 1:
            if j == 0:
                reg_lat_mask = ~np.isnan(lat_mask)
            else:
                reg_lat_mask = (np.abs(lat_mask) >= reg_ls[j - 1]) & (np.abs(lat_mask) < reg_ls[j])
        elif zones == 2:
            reg_lat_mask = (lat_mask > reg_ls[j]) & (lat_mask <= reg_ls[j + 1])
        elif zones == 3:
            if j == 0:
                reg_lat_mask = ~np.isnan(lat_mask)
            else:
                reg_lat_mask = reccap_mask == j

        reg_mod_mask = mod_mask[reg_lat_mask, :]
        reg_obs_mask = obs_mask[reg_lat_mask, :]
        mod_mean = np.nanmean(reg_mod_mask, axis=0)
        obs_mean = np.nanmean(reg_obs_mask, axis=0)

        if mod_var_ar1 is not None:
            mod1_mask = mod_var_ar1[test_mask, :] if test_mask is not None else mod_var_ar1
            reg_mod1_mask = mod1_mask[reg_lat_mask, :]
            mod1_mean = np.nanmean(reg_mod1_mask, axis=0)

        x = np.arange(num_time_steps)

        if anomaly:
            # cal anomaly
            mod_mean = cal_monthly_interannual_anomalies(mod_mean, time_series)
            obs_mean = cal_monthly_interannual_anomalies(obs_mean, time_series)
            # plot trend
            sns.regplot(x=x, y=obs_mean, ci=95, scatter=False, color='dimgray', ax=axname)
            sns.regplot(x=x, y=mod_mean, ci=95, scatter=False, color='steelblue', ax=axname)
            slope_mod, _, _, p_value_mod, _ = linregress(x=x, y=mod_mean)
            slope_obs, _, _, p_value_obs, _ = linregress(x=x, y=obs_mean)
            # add slope and p values
            axname.text(
                0.05, 0.95, f'Slope: {slope_obs:.4f}, $P$ = {p_value_obs:.2f}', transform=axname.transAxes,
                va='top', ha='left', fontsize=10, color='dimgray')
            axname.text(
                0.05, 0.85, f'Slope: {slope_mod:.4f}, $P$ = {p_value_mod:.2f}', transform=axname.transAxes,
                va='top', ha='left', fontsize=10, color='steelblue')

            if mod_var_ar1 is not None:
                mod1_mean = cal_monthly_interannual_anomalies(mod1_mean, time_series)
                sns.regplot(x=x, y=mod1_mean, ci=95, scatter=False, color='#C9C780', ax=axname)
                slope_mod1, _, _, p_value_mod1, _ = linregress(x=x, y=mod1_mean)
                axname.text(
                    0.05, 0.75, f'Slope: {slope_mod1:.4f}, $P$ = {p_value_mod1:.2f}', transform=axname.transAxes,
                    va='top', ha='left', fontsize=10, color='#C9C780')

        # plot time series
        lab = 'Model1' if mod_var_ar1 is not None else 'Model'
        axname.plot(x, obs_mean, color='dimgray', linewidth=linewd, alpha=0.5, label='Observation')
        axname.plot(x, mod_mean, color='steelblue', linewidth=linewd, alpha=0.5, label=lab)
        if metric == 'R2':
            try:
                index = r2_score(obs_mean, mod_mean)
            except:
                index = 0.0
        elif metric == 'NNSE':
            try:
                index = r2_score(obs_mean, mod_mean)
                index = 1 / (2 - index)
            except:
                index = 0.0
        elif metric == 'r':
            try:
                index = np.corrcoef(obs_mean, mod_mean)[0, 1]
            except:
                index = 0.0
        index = np.around(index, 2)
        index_ls.append(index)

        if mod_var_ar1 is not None:
            axname.plot(x, mod1_mean, color='#C9C780', linewidth=linewd, alpha=0.5, label='Model2')
            if metric == 'R2':
                index1 = r2_score(obs_mean, mod1_mean)
            elif metric == 'NNSE':
                index1 = r2_score(obs_mean, mod1_mean)
                index1 = 1 / (2 - index)
            elif metric == 'r':
                index1 = np.corrcoef(obs_mean, mod1_mean)[0, 1]
            index1 = np.around(index1, 2)
            index_ls.append(index1)

        axname.set_xticks(np.arange(0, num_time_steps + 1, 24))
        xticks = [int(xtick / 12) + 2001 for xtick in axname.get_xticks()]
        axname.set_xticklabels(xticks)

        axname.set_ylabel(lab)
        if mod_var_ar1 is not None:
            txt = f'{txt_ls[j]}, M1 {metric_lab} = {index}, M2 {metric_lab} = {index1}'
        else:
            txt = f'{txt_ls[j]}, {metric_lab} = {index}'
        axname = add_fignum(axname, txt)
    
    ncols, loc = (1, 'lower right') if anomaly else (2, 'upper left')
    if mod_var_ar1 is not None:
        ncols = 3
    ax_ls[-1].legend(ncols=ncols, loc=loc, frameon=False)

    # save result
    zones_ls = ['4_temp_zones', '6_temp_zones', 'reccap']
    if fig_dir:
        key_lab = 'anomaly' if anomaly else 'original'
        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f'Temporal_series_{var_name}_{key_lab}_{zones_ls[zones - 1]}.png'
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path, dpi=300)
        print(f'Saving {fig_path}')
    plt.close()

    # NNSE/R2/r of global and regional averaged monthly interannual anomaly/original gpp/lai/reco/..., each containing 4/6/11 regions
    return index_ls


def cal_temporal_performance_grid(mod_var_ar, obs_var_ar, metric='R2'):
    """ Function to calculate performance at each grid point"""

    mod_var_ar = np.array(mod_var_ar)
    obs_var_ar = np.array(obs_var_ar)
    # cal performance of gpp, lai, and reco ...
    index_ls = []
    for i in range(mod_var_ar.shape[2]):
        mod = np.array(mod_var_ar[:, :, i])
        obs = np.array(obs_var_ar[:, :, i])
        if metric == 'R2':
            index = r2_score(obs.T, mod.T, multioutput='raw_values')
            index = np.where(index > 0, index, 0)
        elif metric == 'NNSE':
            index = r2_score(obs.T, mod.T, multioutput='raw_values')
            index = 1 / (2 - index)
        elif metric == 'r':
            index = np.array([np.corrcoef(obs[j, :], mod[j, :])[0, 1] for j in range(obs.shape[0])])
        index_ls.append(index)

    return np.array(index_ls).T


def plot_temporal_performance_grid(mod_ar_all, obs_ar_all, var_name_ls, valid_mask, region, lon, lat, 
                                   metric='R2', fig_dir=None, test_mask=None, mod_ar_all1=None):
    """ Function to plot temporal performance at each grid point"""

    mod_ar_all = np.array(mod_ar_all)
    obs_ar_all = np.array(obs_ar_all)

    metric_lab = 'R$^2$' if metric == 'R2' else metric
    left_loc = 0.05
    right_loc = 0.94
    if mod_ar_all1 is not None:
        mod_ar_all1 = np.array(mod_ar_all1)
        fig = plt.figure(figsize=(16, 16), dpi=300)
        grid = GridSpec(4, 2, figure=fig, left=left_loc, bottom=0.1, right=right_loc, top=0.96, wspace=0.26,
                        hspace=0.17)
        ax_ls = [plt.subplot(grid[i, j]) for j in range(2) for i in range(4)]
        var_name_ls = ['(a) M1 GPP', '(b) M1 LAI', '(c) M1 Reco', '(d) M1 ET',
                  '(e) M2 GPP', '(f) M2 LAI', '(g) M2 Reco', '(h) M2 ET']
        # cal r2
        index_ls = cal_temporal_performance_grid(mod_ar_all, obs_ar_all, metric).T
        index_ls1 = cal_temporal_performance_grid(mod_ar_all1, obs_ar_all, metric).T
        index_ls = np.vstack([index_ls, index_ls1])

    else:
        n_plots = mod_ar_all.shape[2]
        # Fixed number of columns
        ncols = 2
        # Calculate number of rows
        nrows = (n_plots + ncols - 1) // ncols
        # Calculate figure size based on number of rows
        figsize = (8*ncols, 4*nrows)
        fig = plt.figure(figsize=figsize, dpi=300)
        grid = GridSpec(nrows, ncols, figure=fig, left=left_loc, bottom=0.1, right=right_loc, 
                        top=0.96, wspace=0.26, hspace=0.17)
        ax_ls = [plt.subplot(grid[i, j]) for i in range(nrows) for j in range(ncols)]
        # Remove unused subplots
        for i in range(n_plots, len(ax_ls)):
            fig.delaxes(ax_ls[i])

        # cal r2
        index_ls = cal_temporal_performance_grid(mod_ar_all, obs_ar_all, metric).T

    index_all = index_ls.flatten()
    index_notnan = index_all[~np.isnan(index_all)]
    vmin, vmax = np.percentile(index_notnan, 1), np.percentile(index_notnan, 99)
    if (vmin == 0) & (vmax == 0):
        norm = TwoSlopeNorm(vmin=-0.000001, vcenter=0, vmax=0.000001)
        cmap = plt.get_cmap('coolwarm')
    elif vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = plt.get_cmap('coolwarm')
    elif vmax <= 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vmin / 2, vmax=0)
        cmap = LinearSegmentedColormap.from_list(
            "warm_coolwarm", plt.get_cmap("coolwarm")(np.linspace(0, 0.5, 256))
        )
    elif vmin >= 0:
        norm = TwoSlopeNorm(vmin=0, vcenter=vmax / 2, vmax=vmax)
        cmap = plt.get_cmap('Blues')

    grid_mean_index_ls = []
    grid_std_index_ls = []
    # zonal statistic
    bin_edges_ref = [-60, -50, -23.5, 23.5, 50, 80]
    lat_min, lat_max = region.get("lat_min", -90), region.get("lat_max", 90)
    lon_min, lon_max = region.get("lon_min", -180), region.get("lon_max", 180)
    bin_edges = get_bin_edges(lat_min, lat_max, bin_edges_ref)
    bin_centers = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)])
    bin_height = np.array([(bin_edges[i + 1] - bin_edges[i]) for i in range(len(bin_edges) - 1)])
    clr1 = 'steelblue'
    clr2 = 'dimgray'
    for i in range(len(var_name_ls)):
        axname = ax_ls[i]
        index = index_ls[i]
        # cal mean value and std
        tempt_mean = np.nanmean(index[test_mask]) if test_mask is not None else np.nanmean(index)
        tempt_mean = np.around(tempt_mean, 2)
        grid_mean_index_ls.append(tempt_mean)
        tempt_std = np.nanstd(index[test_mask]) if test_mask is not None else np.nanstd(index)
        tempt_std = np.around(tempt_std, 2)
        grid_std_index_ls.append(tempt_std)

        index_grid = restore_to_original_length(index, valid_mask, fill_value=np.nan)
        index_grid = index_grid.reshape(len(lat), len(lon))
        index_grid = xr.DataArray(index_grid, coords=[lat, lon], dims=["lat", "lon"])
        index_grid['lon'] = xr.where(index_grid['lon'] > 180, index_grid['lon'] - 360, index_grid['lon'])
        im = index_grid.plot(ax=axname, cmap=cmap, norm=norm, add_colorbar=False)
        axname = adjust_axis_label(axname, lat_min, lat_max, lon_min, lon_max)
        txt = (var_name_ls[i] + ', ' + metric_lab + '=' + str(tempt_mean) + '\u00b1' + str(tempt_std))
        axname = add_fignum(axname, txt)

        # add lat stats
        lat_mean = index_grid.mean(dim='lon', skipna=True)
        lat_std = index_grid.std(dim='lon', skipna=True)
        ax_lat = fig.add_axes(
            [axname.get_position().x1, axname.get_position().y0, 0.05, axname.get_position().height]
        )
        ax_lat.plot(lat_mean, lat_mean.lat, color=clr1, linewidth=1.5, alpha=0.5)
        ax_lat.fill_betweenx(lat_mean.lat, lat_mean + lat_std, lat_mean - lat_std, color=clr1, alpha=0.3)
        xlim = ax_lat.get_xlim()
        ax_lat.set_xlim([0, xlim[1]])

        # add temperature zonal stats
        bin_values = []
        for j in range(len(bin_edges) - 1):
            reg_mask = index_grid.sel(lat=slice(bin_edges[j], bin_edges[j + 1]))
            reg_mean = reg_mask.mean(skipna=True).item()
            bin_values.append(reg_mean)
        ax_lat.barh(bin_centers, bin_values, bin_height, color='none', edgecolor=clr2)
        ax_lat.set_ylim(lat_min, lat_max)
        ticks_y = np.arange(lat_min, lat_max, int((lat_max - lat_min) / 6))
        ax_lat.set_yticks(ticks_y)
        ax_lat.set_yticklabels([])
        ax_lat.set_xlabel(None)
        ax_lat.set_ylabel(None)

    # add shared colorbars below the plot
    cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_label(metric_lab)
    if vmin < 0 < vmax:
        t1 = np.ceil(np.linspace(vmin, 0, num=3) * 10) / 10
        t2 = np.floor(np.linspace(0, vmax, num=3) * 10) / 10
        tick = np.hstack((t1, t2[1:]))
        cbar.set_ticks(tick)

    # save result
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f'Temporal_performance_grid.png'
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path, dpi=300)
        print(f'Saving {fig_path}')
    plt.close()

    # mean and std NNSE/R2/r of gpp, lai and reco ... in all grids
    return grid_mean_index_ls, grid_std_index_ls


def plot_temporal_loss(loss_ls, var_name_ls, fig_dir=None):
    """ Function to plot temporal loss """

    loss_ls = np.array(loss_ls)
    n_plots = len(loss_ls)

    # Fixed number of columns
    ncols = 3
    # Calculate number of rows
    nrows = (n_plots + ncols - 1) // ncols
    # Calculate figure size based on number of rows
    figsize = (4*ncols, 4*nrows)
    # Create figure and grid
    fig = plt.figure(figsize=figsize, dpi=300)
    grid = GridSpec(nrows, ncols, figure=fig, left=0.03, bottom=0.04, right=0.98, top=0.98, wspace=0.25, hspace=0.25)
    # Create all subplots
    ax_ls = [plt.subplot(grid[i, j]) for i in range(nrows) for j in range(ncols)]
    # Remove unused subplots
    for i in range(n_plots, len(ax_ls)):
        fig.delaxes(ax_ls[i])

    for i in range(n_plots):
        axname = ax_ls[i]
        axname.plot(loss_ls[i], marker='o', linestyle='-', color='steelblue')
        axname.set_xlabel('Epochs')
        axname.set_ylabel('Loss')
        axname.set_title(var_name_ls[i], fontweight='bold')

        # Add inset for the last 100 epochs (if there are at least 100 epochs)
        if len(loss_ls[i]) > 100:
            axname_inset = axname.inset_axes([0.3, 0.6, 0.65, 0.35])  # Define the position of the inset plot
            axname_inset.plot(range(len(loss_ls[i]) - 100, len(loss_ls[i])), loss_ls[i][-100:], marker='o',
                              linestyle='-',
                              color='steelblue')

    # Save result
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f'Temporal_loss.png'
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path, dpi=300)
        print(f'Saving {fig_path}')
        plt.close()

    return


def cal2polt_global(result_ls, mask=None):
    """ Function to preprocess the results for visualization """

    (results, invariant_obs, variant_obs, grid_weights, loss_ls, mask_ls, region, lon, lat, 
     init_pools, init_params, VARNAME_LS, VAR_IND_LS, param_bound_ls) = result_ls
    train_mask, _, test_mask, valid_mask = mask_ls
    var_obs_ls, invar_obs_ls, params_ls, pools_ls = VARNAME_LS
    var_obs_ind, invar_obs_ind, params_ind, pools_ind = VAR_IND_LS
    param_min, param_max, pool_min, pool_max = param_bound_ls
    results = results[:, :, jnp.concatenate([var_obs_ind, invar_obs_ind])]
    n_var = len(var_obs_ind)
    n_invar = len(invar_obs_ind)
    n_total = n_var + n_invar

    if mask is None:
        new_test_mask = None
    elif mask == "train":
        new_test_mask = train_mask
    elif mask == "test":
        new_test_mask = test_mask
    else:
        sys.exit("Wrong mask name, plese input 'train', 'test' or None")

    # Observations
    if variant_obs is None:
        obs_ar_all = invariant_obs
    elif invariant_obs is None:
        obs_ar_all = jnp.mean(variant_obs, axis=1)
    else:
        obs_ar_all = jnp.hstack([jnp.mean(variant_obs, axis=1), invariant_obs])

    # Models
    variant_mod = None if n_var == 0 else results[:, :, :n_var]
    invariant_flux = None if n_invar == 0 else jnp.mean(results[:, :, n_var:n_total], axis=1)
    unnorm_params = batched_unnormalize(init_params, param_min, param_max)
    unnorm_pools = batched_unnormalize(init_pools, pool_min, pool_max)
    invariant_params = None if len(params_ind) == 0 else unnorm_params[:, params_ind]
    invariant_pools = None if len(pools_ind) == 0 else unnorm_pools[:, pools_ind]
    stacked_result = []
    if invariant_flux is not None:
        stacked_result.append(invariant_flux)
    if invariant_params is not None:
        stacked_result.append(invariant_params)
    if invariant_pools is not None:
        stacked_result.append(invariant_pools)

    # Perform the hstack only if there are values to stack
    if stacked_result:
        invariant_mod = jnp.hstack(stacked_result)
    else:
        invariant_mod = None
    if variant_mod is None:
        mod_ar_all = invariant_mod
    elif invariant_mod is None:
        mod_ar_all = jnp.mean(variant_mod, axis=1)
    else:
        mod_ar_all = jnp.hstack([jnp.mean(variant_mod, axis=1), invariant_mod])
    var_name_ls = var_obs_ls + invar_obs_ls + params_ls + pools_ls

    # Get unit
    var_obs_unit = [target_unit[i] for i in var_obs_ind.tolist()]
    invar_obs_unit = [target_unit[i] for i in invar_obs_ind.tolist()]
    params_unit = [target_unit[i] for i in params_ind.tolist()]
    pools_unit = [target_unit[i] for i in pools_ind.tolist()]
    var_unit_ls = var_obs_unit + invar_obs_unit + params_unit + pools_unit

    return (loss_ls, results, variant_obs, mod_ar_all, obs_ar_all, grid_weights, var_name_ls, 
            var_unit_ls, new_test_mask, valid_mask, region, lon, lat, n_var)


def visualization_global(result_ls, fig_dir, mask=None, metric='R2'):
    """" Function to plot results at global scale """

    (loss_ls, results, variant_obs, mod_ar_all, obs_ar_all, grid_weights, var_name_ls, 
     var_unit_ls, new_test_mask, valid_mask, region, lon, lat, n_var) = cal2polt_global(result_ls, mask)

    # Plot loss
    var_name_ls_loss = ['Total'] + var_name_ls
    plot_temporal_loss(loss_ls, var_name_ls_loss, fig_dir)

    # Plot scatter
    scatter_idx = plot_scatter(mod_ar_all, obs_ar_all, grid_weights, var_name_ls, 
                            var_unit_ls, new_test_mask, fig_dir, metric=metric)
    
    # Plot spatial pattern and temporal series
    for i in range (len(var_name_ls)):
        # Plot spatial pattern
        mod_var_ar = mod_ar_all[:, i]
        obs_var_ar = obs_ar_all[:, i]
        var_name = var_name_ls[i]
        var_unit = var_unit_ls[i]
        grid_weight = grid_weights[:, i]
        plot_global_pattern_difference(mod_var_ar, obs_var_ar, var_name, var_unit, 
                                    valid_mask, region, lon, lat, grid_weight, fig_dir)
        if i < n_var:
            mod_var_ar = results[:, :, i]
            obs_var_ar = variant_obs[:, :, i]
            # Plot temporal series
            # Original
            tempo_idx_ori = plot_temporal_series(
                mod_var_ar, obs_var_ar, var_name, var_unit, valid_mask, region, lon, lat,
                metric, False, 3, fig_dir, new_test_mask
            )
            # # Anomaly
            tempo_idx_anom = plot_temporal_series(
                mod_var_ar, obs_var_ar, var_name, var_unit, valid_mask, region, lon, lat,
                metric, True, 3, fig_dir, new_test_mask
            )

    # Plot grid performance
    mod_ar_all_grid = results[:, :, :n_var]
    obs_ar_all_grid = variant_obs
    var_name_ls = var_name_ls[:n_var]
    grid_idx_mean, grid_idx_std = plot_temporal_performance_grid(
        mod_ar_all_grid, obs_ar_all_grid, var_name_ls, valid_mask, region, lon, lat, metric, fig_dir, new_test_mask
    )

    return scatter_idx, tempo_idx_ori, tempo_idx_anom, grid_idx_mean, grid_idx_std

