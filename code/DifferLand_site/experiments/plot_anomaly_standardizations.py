import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Fayong Liu
# 5/13/2025
# this file is for plotting anomaly and standardization of each var in each site


# def a function cal monthly interannual anomalies
def cal_monthly_interannual_anomalies(dataarray, time_series):
    months = time_series.month
    # cal monthly average
    monthly_mean = np.array([np.nanmean(dataarray[months == m]) for m in range(1, 13)])

    # cal monthly anomaly
    anomaly_array = np.copy(dataarray)
    for m in range(1, 13):
        anomaly_array[months == m] = dataarray[months == m] - monthly_mean[m - 1]

    return anomaly_array


# def a function plot
def plot_anomaly_std(sitename, varname, unit, outpath):
    # Load the dataset with automatic time decoding
    path = f'/data1/fyliu/e_soil_carbon_model/code/DifferLand_site/drivers/{sitename}.nc'
    ds = xr.open_dataset(path, decode_times=True)
    var = ds[varname]
    base_date = pd.to_datetime('2000-01')
    time_series = base_date + pd.to_timedelta(ds['time'].values + 1, unit='D')
    var['time'] = time_series
    var_monthly = var.resample(time='1ME').mean()

    # cal original anomaly
    x = pd.to_datetime(var_monthly.time)
    ds_anomaly_original = cal_monthly_interannual_anomalies(var_monthly.values, x)
    # cal standardization
    scaler = StandardScaler()
    ds_standardized = scaler.fit_transform(var_monthly.values.reshape(-1, 1)).reshape(-1)
    # cal standardized anomaly
    ds_anomaly_standardized = cal_monthly_interannual_anomalies(ds_standardized, x)

    # Plotting the 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axs = axs.flatten()

    y_ls = [var_monthly.values, ds_standardized, ds_anomaly_original, ds_anomaly_standardized]
    clr_ls = ['blue', 'green', 'red', 'orange']
    num_ls = [f'(a) Original {varname}', f'(b) Standardized {varname}',
              f'(c) Anomaly of original {varname}', f'(d) Anomaly of standardized {varname}']

    for i, ax in enumerate(axs):
        ax.plot(x, y_ls[i], color=clr_ls[i])
        ax.set_title(num_ls[i])
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{varname} ({unit})")

    # Display the plots
    plt.tight_layout()

    # save result
    outname = f"{sitename}_{varname}_annomaly_standardization.png"
    finalpath = os.path.join(outpath, outname)
    plt.savefig(finalpath, dpi=300)
    print(finalpath)

    return


def main():
    sitename = 'AU-DaS'
    varname_ls = ['GPP', 'LAI', 'RECO', 'ET', 'SOM']
    unit_ls = ['g C/m$^2$', 'm$^2$/m$^2$', 'g C/m$^2$', 'mm/day', 'g C/m$^2$']
    outpath = r'/data1/fyliu/e_soil_carbon_model/code/DifferLand_site/experiments/fig/anomaly_std'
    os.makedirs(outpath, exist_ok=True)
    for varname, unit in zip(varname_ls, unit_ls):
        plot_anomaly_std(sitename, varname, unit, outpath)

    return


if __name__ == "__main__":
    main()
