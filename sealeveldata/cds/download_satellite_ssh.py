#!/usr/bin/env python
import concurrent.futures
import os, sys
import cdsapi

from cmip6.cds import _extract_files
from cmip6.regrid import cdo

c = cdsapi.Client()

dataset = 'satellite-sea-level-global'

def filename(year):
    return f'{dataset}/{year}.zip'


def download_year(year):

    zipfile = filename(year)

    os.makedirs(os.path.dirname(zipfile), exist_ok=True)

    if os.path.exists(zipfile):
        print(zipfile, 'already exists')
        return

    c.retrieve(
        dataset,
        {
            'version': 'vDT2021',
            'variable': 'all',
            'format': 'zip',
            'year': year,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
        },
        zipfile)

workers = 1
years = list(range(1993, 2019+1)) # 2020 seems buggy, 2021 is not complete

target = f'{dataset}/satellite_sla_{years[0]}_{years[-1]}.nc'

if os.path.exists(target):
    sys.exit(0)


with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    futures = { executor.submit(download_year, year) : year for year in years}
    for future in concurrent.futures.as_completed(futures):
        year = futures[future]
        try:
            future.result()
        except Exception as exc:
            # print(exc)
            print(f'!! failed to download: {year}')
            raise


def yearmean(y):
    if os.path.exists(f'{dataset}/{y}.nc'):
        return
    subfiles = _extract_files(filename(y))
    cdo(f'yearmean -mergetime -select,name=sla,tpa_correction {" ".join(subfiles)} {dataset}/{y}.nc')


with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    futures = { executor.submit(yearmean, year) : year for year in years}
    for future in concurrent.futures.as_completed(futures):
        year = futures[future]
        try:
            future.result()
        except Exception as exc:
            print(f'!! failed to merge year: {year}')
            raise

# for y in years:
#     subfiles = _extract_files(filename(y))
#     cdo(f'yearmean -mergetime -select,name=sla,tpa_correction {" ".join(subfiles)} {y}.nc')

cdo(f"""mergetime {" ".join(f'{dataset}/{y}.nc' for y in years)} {target}""")
