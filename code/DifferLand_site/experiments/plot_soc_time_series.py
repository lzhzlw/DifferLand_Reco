import pickle
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(1, '../')
from DifferLand.model.DALEC_990_parinfo import dalec990_pfn

# Paths for input data and output figures
out_path = r"/data1/fyliu/e_soil_carbon_model/code/DifferLand_v3_1_site_soil_pool/experiments/output/"
fig_path = r"/data1/fyliu/e_soil_carbon_model/code/DifferLand_v3_1_site_soil_pool/experiments/fig/"

# List of site names
sites = ["US-Var", "FR-Pue", "NL-Loo", "US-Ha1"]

# Define colors
colors = {
    "soc": "black",
    "soc1": "maroon",
    "soc2": "steelblue",
    "soc3": "#037F77"
}

# Set font sizes for plots
ftsz = 20
plt.rcParams.update({
    "font.size": ftsz,  # Set default font size for all text
    "axes.titlesize": ftsz+2,
    "axes.labelsize": ftsz,
    "legend.fontsize": ftsz,
    "xtick.labelsize": ftsz,
    "ytick.labelsize": ftsz
})

linewidth = 2
alp = 0.5

# Function to plot figures
def plot_figures(site, n=0, extra_suffix=""):
    out_name = f"daily_{site}_with_som_default_century_0.0005_25000_2.pkl"
    output_save_name = os.path.join(out_path, out_name)

    # Load data from the pickle file
    with open(output_save_name, "rb") as fp:
        data = pickle.load(fp)

    output_matrix_full = data["output_matrix_full"]

    # Extract soil carbon pools
    soc = output_matrix_full[n:, dalec990_pfn.soc_pool]
    soc1 = output_matrix_full[n:, dalec990_pfn.soc1_pool]
    soc2 = output_matrix_full[n:, dalec990_pfn.soc2_pool]
    soc3 = output_matrix_full[n:, dalec990_pfn.soc3_pool]

    # Extract soil respiration data
    respiration_soc = output_matrix_full[n:, dalec990_pfn.respiration_hetero_soc]
    respiration_soc1 = output_matrix_full[n:, dalec990_pfn.respiration_hetero_soc1]
    respiration_soc2 = output_matrix_full[n:, dalec990_pfn.respiration_hetero_soc2]
    respiration_soc3 = output_matrix_full[n:, dalec990_pfn.respiration_hetero_soc3]

    # Extract litter inputs
    litter_to_soc = output_matrix_full[n:, dalec990_pfn.litter_to_soc]
    litter_to_soc1 = output_matrix_full[n:, dalec990_pfn.litter_to_soc1]
    litter_to_soc2 = output_matrix_full[n:, dalec990_pfn.litter_to_soc2]
    litter_to_soc3 = output_matrix_full[n:, dalec990_pfn.litter_to_soc3]

    # Create a time axis (daily from 2000-01-01)
    start_date = datetime(2000, 1, n + 1)
    days = np.arange(output_matrix_full.shape[0] - n)
    dates = [start_date + timedelta(days=int(day)) for day in days]

    # Create the figure
    plt.figure(figsize=(15, 15))

    # (a) Litter Inputs
    plt.subplot(3, 1, 1)
    plt.plot(dates, litter_to_soc, label="Litter to Total SOC", color=colors["soc"], linewidth=linewidth, alpha=alp)
    plt.plot(dates, litter_to_soc1, label="Litter to Active SOC", color=colors["soc1"], linewidth=linewidth, alpha=alp)
    plt.plot(dates, litter_to_soc2, label="Litter to Slow SOC", color=colors["soc2"], linewidth=linewidth, alpha=alp)
    #plt.plot(dates, litter_to_soc3, label="Litter to Passive SOC", color=colors["soc3"], linewidth=linewidth, alpha=alp)
    plt.xlabel("Year")
    plt.ylabel("Litter Input (gC/m$^2$/day)")
    plt.title(f"(a) Litter Inputs to SOC ({site})")
    #plt.legend(loc="upper left", frameon=False)

    # (b) Soil Carbon Pool
    plt.subplot(3, 1, 2)
    plt.plot(dates, soc, label="SOC Total Pool", color=colors["soc"], linewidth=linewidth, alpha=alp)
    plt.plot(dates, soc1, label="Active SOC Pool", color=colors["soc1"], linewidth=linewidth, alpha=alp)
    plt.plot(dates, soc2, label="Slow SOC Pool", color=colors["soc2"], linewidth=linewidth, alpha=alp)
    plt.plot(dates, soc3, label="Passive SOC Pool", color=colors["soc3"], linewidth=linewidth, alpha=alp)
    plt.xlabel("Year")
    plt.ylabel("Soil Carbon (C g/m$^2$)")
    plt.title(f"(b) Soil Carbon Pool ({site})")

    # (c) Soil Respiration
    plt.subplot(3, 1, 3)
    plt.plot(dates, respiration_soc, label="Total SOC", color=colors["soc"], linewidth=linewidth, alpha=alp)
    plt.plot(dates, respiration_soc1, label="Active SOC ", color=colors["soc1"], linewidth=linewidth, alpha=alp)
    plt.plot(dates, respiration_soc2, label="Slow SOC", color=colors["soc2"], linewidth=linewidth, alpha=alp)
    plt.plot(dates, respiration_soc3, label="Passive SOC", color=colors["soc3"], linewidth=linewidth, alpha=alp)
    plt.xlabel("Year")
    plt.ylabel("Respiration (gC/m$^2$/day)")
    plt.title(f"(c) Soil Respiration ({site})")
    plt.legend(loc="upper left", frameon=False, ncol=4)

    # Save the figure
    base_name = os.path.splitext(os.path.basename(output_save_name))[0]
    fig_save_name = os.path.join(fig_path, f"soc_resp_litter_time_series_{base_name}{extra_suffix}.png")
    plt.tight_layout()
    plt.savefig(fig_save_name)
    plt.close()

    print(f"Figure saved: {fig_save_name}")


# Loop through sites and generate plots
for site in sites:
    # Plot with default n=0
    plot_figures(site, n=0)

