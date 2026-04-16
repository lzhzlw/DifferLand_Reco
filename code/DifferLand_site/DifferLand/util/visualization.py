import matplotlib.pyplot as plt
import numpy as np
from DifferLand.model.DALEC_990_parinfo import dalec990_pfn
from datetime import datetime, timedelta


def plot_site_figure(driver_ds, som, output_matrix_full, train_sel, test_sel, nnse_eval, save_name, n=0, reco=False):
    gpp_tr, nee_reco_tr, et_tr, lai_tr, som_tr, gpp_te, nee_reco_te, et_te, lai_te = nnse_eval
    fig, axs = plt.subplots(5, 1, figsize=(10, 10), dpi=300)
    ax = axs.flatten()

    # Create time sequences
    start_date = datetime(2000, 1, 1+n)  # Starting date
    days = np.arange(output_matrix_full.shape[0]-n)
    dates = [start_date + timedelta(days=int(day)) for day in days]

    # Train, Test, and Other sequences
    train_sel = train_sel[n:]
    test_sel = test_sel[n:]
    train_dates = [dates[i] for i in range(len(dates)) if train_sel[i]]
    test_dates = [dates[i] for i in range(len(dates)) if test_sel[i]]

    # Plot GPP
    ax[0].plot(train_dates, driver_ds.GPP.values[train_sel], alpha=0.5, label="Observations (Train)")
    ax[0].plot(test_dates, driver_ds.GPP.values[test_sel], alpha=0.5, label="Observations (Test)")
    ax[0].plot(dates, output_matrix_full[:, dalec990_pfn.gpp], label="Model")
    ax[0].set_ylabel("GPP (gC/m$^2$/day)")
    ax[0].text(dates[1], np.nanmax(driver_ds.GPP.values) * 0.9, f"Train NNSE: {gpp_tr:.3f}")
    ax[0].text(test_dates[365*2], np.nanmax(driver_ds.GPP.values) * 0.9, f"Test NNSE: {gpp_te:.3f}")
    #ax[0].legend(bbox_to_anchor=(0, 1.01), loc="lower left", ncol=3, frameon=False)

    # Plot RECO or NEE
    if not reco:
        ax[1].plot(train_dates, driver_ds.NBE.values[train_sel], alpha=0.5, label="Observations (Train)")
        ax[1].plot(test_dates, driver_ds.NBE.values[test_sel], alpha=0.5, label="Observations (Test)")
        ax[1].plot(dates, output_matrix_full[:, dalec990_pfn.nee], label="Model")
        ax[1].set_ylabel("NEE (gC/m$^2$/day)")
        ax[1].text(dates[1], np.nanmax(driver_ds.NBE.values) * 0.9, f"Train NNSE: {nee_reco_tr:.3f}")
        ax[1].text(test_dates[365*2], np.nanmax(driver_ds.NBE.values) * 0.9, f"Test NNSE: {nee_reco_te:.3f}")
    else:
        ax[1].plot(train_dates, driver_ds.RECO.values[train_sel], alpha=0.5, label="Observations (Train)")
        ax[1].plot(test_dates, driver_ds.RECO.values[test_sel], alpha=0.5, label="Observations (Test)")
        ax[1].plot(dates, output_matrix_full[:, dalec990_pfn.reco], label="Model")
        ax[1].set_ylabel("RECO (gC/m$^2$/day)")
        ax[1].text(dates[1], np.nanmax(driver_ds.RECO.values) * 0.9, f"Train NNSE: {nee_reco_tr:.3f}")
        ax[1].text(test_dates[365*2], np.nanmax(driver_ds.RECO.values) * 0.9, f"Test NNSE: {nee_reco_te:.3f}")

    # Plot ET
    ax[2].plot(train_dates, driver_ds.ET.values[train_sel], alpha=0.5, label="Observations (Train)")
    ax[2].plot(test_dates, driver_ds.ET.values[test_sel], alpha=0.5, label="Observations (Test)")
    ax[2].plot(dates, output_matrix_full[:, dalec990_pfn.ET], label="Model")
    ax[2].set_ylabel("ET (mm/day)")
    ax[2].text(dates[1], np.nanmax(driver_ds.ET.values) * 0.9, f"Train NNSE: {et_tr:.3f}")
    ax[2].text(test_dates[365*2], np.nanmax(driver_ds.ET.values) * 0.9, f"Test NNSE: {et_te:.3f}")

    # Plot LAI
    ax[3].plot(train_dates, driver_ds.LAI.values[train_sel], alpha=0.5, label="Observations (Train)")
    ax[3].plot(test_dates, driver_ds.LAI.values[test_sel], alpha=0.5, label="Observations (Test)")
    ax[3].plot(dates, output_matrix_full[:, dalec990_pfn.lai], label="Model")
    ax[3].set_ylabel("LAI (m$^2$/m$^2$)")
    ax[3].text(dates[1], np.nanmax(driver_ds.LAI.values) * 0.9, f"Train NNSE: {lai_tr:.3f}")
    ax[3].text(test_dates[365*2], np.nanmax(driver_ds.LAI.values) * 0.9, f"Test NNSE: {lai_te:.3f}")

    # SOC comparison
    model_mean_soc = np.mean(output_matrix_full[:, dalec990_pfn.soc_pool])
    ymax = max(som, model_mean_soc)*1.2
    ax[4].scatter([1], [som], alpha=0.8, facecolor='dimgray', edgecolor='black', s=100)
    ax[4].scatter([2], [model_mean_soc], alpha=0.8, facecolor='steelblue', edgecolor='black', s=100)
    ax[4].text(1.1, som, f"{som:.2f}", ha='left', va='center', fontsize=12, color='dimgray')
    ax[4].text(2.1, model_mean_soc, f"{model_mean_soc:.2f}", ha='left', va='center', fontsize=12, color='steelblue')
    ax[4].set_ylabel("SOC (gC/m$^2$)")
    ax[4].set_xlim([0, 3])
    ax[4].set_xticks([1, 2])
    ax[4].set_xticklabels(['Observation', 'Model'])
    ax[4].set_ylim([0, ymax])
    ax[4].set_ylim([0, ymax])
    ax[4].text(0.15, max(som, model_mean_soc), "MRE: {:.3f}".format(som_tr))

    # Layout adjustments
    fig.subplots_adjust(hspace=0.3, left=0.1, right=0.97, top=0.98, bottom=0.05)

    # Save the figure
    plt.savefig(save_name)
    plt.close()

    print(f"Figure saved: {save_name}")
