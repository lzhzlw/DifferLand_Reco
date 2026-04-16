# Function to plot SOC maps


import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_tif_map(file_path, outdir, band=1, cmap='viridis', title=None, unit=None):
    
    with rasterio.open(file_path) as src:
        
        data = src.read(band)
        data = np.where((data >= 0) & (data < 32767), data, np.nan)
        bounds = src.bounds
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    im = ax.imshow(data, extent=[bounds.left, bounds.right,bounds.bottom, bounds.top], cmap=cmap)
    ax.set_title(f"{title}")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label(unit)
    plt.tight_layout()
    
    # Add basic info in the lower left corner
    ax.text(0.03, 0.05, f"Min: {np.nanmin(data)}, Max: {np.nanmax(data)}", transform=ax.transAxes, fontsize=8)
    ax.text(0.03, 0.1, f"Mean: {np.nanmean(data):.2f}, Std: {np.nanstd(data):.2f}", transform=ax.transAxes, fontsize=8)

    os.makedirs(outdir, exist_ok=True)
    outname = os.path.basename(file_path).replace('.tif', '.png')
    outpath = os.path.join(outdir, outname)
    plt.savefig(outpath, dpi=300)
    print(f"Plot saved to {outpath}")

    return

def main():

    var = []
    depths = ['0-5', '5-15', '15-30', '30-60', '60-100', '100-200']
    mean_paths = [f'/data4/dataset/belowground_topo/SoilGrids250m_v2.0/soilgrids_v2_0.5deg/ocd/ocd_{depth}cm_mean.tif' for depth in depths]
    unc_paths = [f'/data4/dataset/belowground_topo/SoilGrids250m_v2.0/soilgrids_v2_0.5deg/ocd/ocd_{depth}cm_uncertainty.tif' for depth in depths]
    outdir = r'../../../fig/soil'
    for i in range(len(mean_paths)):
        plot_tif_map(mean_paths[i], outdir, title=f'OCD mean ({depths[i]}cm)', unit='hg m-3')
        plot_tif_map(unc_paths[i], outdir, title=f'OCD uncertainty ({depths[i]}cm)', unit='')

    return

if __name__ == "__main__":

    main()

