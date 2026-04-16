import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_tif_map(file_path, band=1, cmap='viridis'):
    
    with rasterio.open(file_path) as src:
        
        data = src.read(band)
        data = np.where((data >= 0) & (data < 32767), data, np.nan)
        bounds = src.bounds
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    # 主图
    im = ax.imshow(data, extent=[bounds.left, bounds.right,bounds.bottom, bounds.top], cmap=cmap)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    # 打印基本信息
    print(f"Shape: {data.shape}")
    print(f"Min: {np.nanmin(data)}, Max: {np.nanmax(data)}")
    print(f"Mean: {np.nanmean(data):.2f}, Std: {np.nanstd(data):.2f}")

    outname = os.path.basename(file_path).replace('.tif', '.png')
    plt.savefig(outname, dpi=300)
    print(f"Plot saved to {outname}")

    return

# 使用
plot_tif_map('/data4/dataset/belowground_topo/SoilGrids250m_v2.0/soilgrids_v2_0.5deg/ocd/ocd_0-5cm_mean.tif')
plot_tif_map('/data4/dataset/belowground_topo/SoilGrids250m_v2.0/soilgrids_v2_0.5deg/ocd/ocd_0-5cm_uncertainty.tif')