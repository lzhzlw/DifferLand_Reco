# Code for downloading soil temperature and soil moisture dataset


import cdsapi


def download_forc_ts_sm_era5l(var_name, start_year, end_year, outpath):
    ''' Function to download soil temperature and soil moisture of ERA5_Land dataset
    
    :param var_name: variable name to download
    :param start_year: start year to download
    :param end_year: end year to download
    :param outpath: output path
    
    :return None

    '''
    
    dataset = "reanalysis-era5-land-monthly-means"
    variable_ls = [f"{var_name}_layer_1", f"{var_name}_layer_2", f"{var_name}_layer_3", f"{var_name}_layer_4"]
    year_ls = list(range(start_year, end_year + 1))
    year_ls = [str(int(year)) for year in year_ls]
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": variable_ls,
        "year": year_ls,
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    
    client = cdsapi.Client()
    try:
        client.retrieve(dataset, request).download(outpath)
        print(f'Saving {outpath}')
    except Exception as e:
        print(f'Can not download {outpath}. Error: {str(e)}')

    return

    