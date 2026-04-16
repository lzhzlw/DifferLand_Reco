# Code for downloading CO2 of CAMS dataset


import cdsapi


def download_forc_co2_cams(start_year, end_year, outpath):
    ''' Function to download CO2 of CAMS dataset
    
    :param start_year: start year to download
    :param end_year: end year to download
    :param outpath: output path
    
    :return None

    '''

    dataset = "cams-global-greenhouse-gas-inversion"
    year_ls = list(range(start_year, end_year + 1))
    year_ls = [str(int(year)) for year in year_ls]
    request = {
        "variable": "carbon_dioxide",
        "quantity": "concentration",
        "input_observations": "surface",
        "time_aggregation": "instantaneous",
        "version": "latest",
        "year": year_ls,
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        'format': 'netcdf.zip'
    }

    client = cdsapi.Client(url='https://ads.atmosphere.copernicus.eu/api', key='7fe677c8-388f-4a96-8e7e-27e7551adf6b')
    try:
        client.retrieve(dataset, request).download(outpath)
        print(f'Saving {outpath}')
    except Exception as e:
        print(f'Can not download {outpath}. Error: {str(e)}')

    return

    