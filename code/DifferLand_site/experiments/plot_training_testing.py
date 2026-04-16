import pickle
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(1, '../')

from DifferLand.util.visualization import plot_site_figure

ftsz = 12
plt.rcParams.update({
    "font.size": ftsz,  # Set default font size for all text
    "axes.titlesize": ftsz+2,
    "axes.labelsize": ftsz+2,
    "legend.fontsize": ftsz+2,
    "xtick.labelsize": ftsz,
    "ytick.labelsize": ftsz
})

# Paths for input data and output figures
out_path = r"/data1/fyliu/e_soil_carbon_model/code/DifferLand_v3_1_site_soil_pool/experiments/output/"
fig_path = r"/data1/fyliu/e_soil_carbon_model/code/DifferLand_v3_1_site_soil_pool/experiments/fig/"

# List of site names
site_ls = ["US-Var", "FR-Pue", "NL-Loo", "US-Ha1"]
with_som_ls = ["with", "without"]
scm_type_ls = ["onepool", "century"]

for site in site_ls:
    for with_som  in with_som_ls:
        for scm_type in scm_type_ls:
            out_name = f"daily_{site}_{with_som}_som_default_{scm_type}_0.0005_25000_2.pkl"
            output_save_name = os.path.join(out_path, out_name)
            
            # Load data from the pickle file
            with open(output_save_name, "rb") as fp:
                data = pickle.load(fp)
            
            driver_ds = data["driver_ds"]
            som = data["som"]
            output_matrix_full = data["output_matrix_full"]
            train_sel = data["train_sel"]
            test_sel = data["test_sel"]
            nnse_eval = data["nnse_eval"]
            base_name = os.path.splitext(os.path.basename(output_save_name))[0]
            fig_save_name = os.path.join(fig_path, f"{base_name}.png")
            
            plot_site_figure(driver_ds, som, output_matrix_full, train_sel, test_sel, nnse_eval, fig_save_name, n=0, reco=True)



