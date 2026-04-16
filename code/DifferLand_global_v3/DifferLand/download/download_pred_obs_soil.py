# Code for downloading soil data


import os
from soilgrids import SoilGrids


def download_pred_obs_soilgrids(varname, depth, outdir_base, lon_size, lat_size, extent, crs, overwrite):
    ''' Function to download soilgrids data
    https://soilgrids.readthedocs.io/en/latest/index.html
    Gan, T. (2023). CSDMS SoilGrids Data Component. Zenodo. https://doi.org/10.5281/zenodo.10368882
    
    :param varname: variable name
    :param depth: depth
    :param outdir: output directory
    :param lon_size: longitude size
    :param lat_size: latitude size
    :param extent: extent, [west, east, south, north]
    :param crs: coordinate reference system
    :param overwrite: whether to overwrite existing files

    :return None

    '''

    outdir = os.path.join(outdir_base, varname)
    os.makedirs(outdir, exist_ok=True)
    outnames = []
    outpaths = []
    if varname == 'wrb':
        outname = 'MostProbable.tif'
        outnames.append(outname)
        outpaths.append(os.path.join(outdir, outname))
    else:
        for key in ['mean', 'uncertainty']:
            outname = f'{varname}_{depth}cm_{key}.tif'
            outnames.append(outname)
            outpaths.append(os.path.join(outdir, outname))
    
    for outname, outpath in zip(outnames, outpaths):
        if os.path.exists(outpath) and not overwrite:
            print(f'{outpath} already exists, skip downloading')
        else:
            if overwrite:
                print(f'Overwriting {outpath}...')
            else:
                print(f'{outpath} does not exist, downloading...')
            try:
                soil_grids = SoilGrids()
                soil_grids.get_coverage_data(service_id=varname, coverage_id=outname[:-4],
                                             west=extent[0], south=extent[2], east=extent[1], north=extent[3],
                                             width=lon_size, height=lat_size, crs=crs, output=outpath)
                print(f'Saving {outpath}')
            except Exception as e:
                print(f'Can not download {outpath}. Error: {str(e)}')
        
    return

