import re
import os
import glob
from pathlib import Path
import xarray as xa
import concurrent.futures

from sealevelbayes.config import logger, CONFIG
from sealevelbayes.runutils import cdo
from sealevelbayes.datasets.manager import get_datapath
from sealevelbayes.datasets.garnerkopp2022 import load_cmip6

ISIMIPDIR = Path(CONFIG['isimipdir'])
ISIMIP_EXPERIMENTS = ['ssp126', 'ssp370', 'ssp585']
ISIMIP_MODELS = ["UKESM1-0-LL", "MPI-ESM1-2-HR", "IPSL-CM6A-LR", "MRI-ESM2-0", "GFDL-ESM4"]


def build_path(variable, experiment='*', model='*', frequency='daily', domain='global', year_start='*', year_end='*', realm='*', run_id = '*', 
    protocol='ISIMIP3b', input_data='InputData', correction='bias-adjusted', datatype='climate', obsclim='w5e5'):

    if protocol == 'ISIMIP3b':
        folder = ISIMIPDIR / f"{protocol}/{input_data}/{datatype}/{realm}/{correction}/{domain}/{frequency}/{experiment}/{model}/"
        file = f"{model.lower()}_{run_id}_{obsclim}_{experiment}_{variable}_{domain}_{frequency}_{year_start}_{year_end}.nc"
    elif protocol == 'ISIMIP3a':
        folder = ISIMIPDIR / f"{protocol}/{input_data}/{datatype}/{realm}/obsclim/{domain}/{frequency}/historical/{obsclim}/"
        file = f"{obsclim.lower()}_obsclim_{variable}_{domain}_{frequency}_{year_start}_{year_end}.nc"
    return folder / file

def list_files(variable, *args, **kw):
    f = build_path(variable, *args, **kw)
    return sorted(glob.glob(str(f)))

def parse_file(file, variable, **kwargs):
    regex = re.compile(str(
            # get model from path
            build_path(variable, experiment=r'\w+', model=r'(?P<model>.*)', realm=r'\w+', input_data=r'\w+', **kwargs).parent/ 
            # get the rest from name
            build_path(variable, experiment=r'(?P<experiment>\w+)', model='.*', 
                year_start=r'(?P<year_start>\w+)', year_end=r'(?P<year_end>\w+)', run_id=r'\w+', **kwargs).name))
    d = regex.match(file).groupdict()
    # d['model'] = Path(file).parent.name
    return d

def build_path_work(variable, experiment, model, frequency='yearly', domain='globalmean', obsclim='w5e5', timestamp="", protocol='ISIMIP3b'):
    folder = get_datapath(f"{protocol}")
    if protocol == 'ISIMIP3b':
        file = f"{model.lower()}_{obsclim}_{experiment}_{variable}_{domain}_{frequency}{timestamp}.nc"
    elif protocol == 'ISIMIP3a':
        file = f"{obsclim.lower()}_obsclim_{variable}_{domain}_{frequency}{timestamp}.nc"
    else:
        raise NotImplementedError(protocol)
    return folder / file

def load_global_annual_mean(variable, experiment, model, xarray_kwargs={}, max_workers=16, **kwargs):
    file = build_path_work(variable, experiment, model, **kwargs)

    if not file.exists():
        file.parent.mkdir(exist_ok=True, parents=True)
        files_in = list_files(variable, experiment, model, **kwargs)
        # cdo(f"-mergetime -selname,{variable} -fldmean -yearmean {' '.join(files_in)} {file}")  
        # for some reason the above does not work: https://code.mpimet.mpg.de/boards/1/topics/254

        # First pass: year and global mean
        def process_file(file_in):
            logger.debug(f"process {file_in}")
            d = parse_file(file_in, variable, **kwargs)
            file_tmp = build_path_work(variable, experiment, model, 
                timestamp="_{year_start}_{year_end}".format(**d), **kwargs)
            cdo(f"-fldmean -yearmean {file_in} {file_tmp}")
            return str(file_tmp)

        if max_workers > 1:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            files_tmp = [p for p in pool.map(process_file, files_in)]
        else:
            files_tmp = [process_file(f) for f in files_in]

        # Second path: concatenate
        cdo(f"mergetime -selname,{variable} {' '.join(files_tmp)} {file}")

        # Remove temporary files
        for file_tmp in files_tmp:
            os.remove(file_tmp)

    logger.info(f"Load {file}")

    # load and clean-up netCDF
    with xa.open_dataset(file, **xarray_kwargs) as ds:
        a = ds.load()[variable].squeeze()
        a.coords['time'] = a['time.year'].values
        return a.rename({"time": "year"})


def load_tuning_data_zostoga(model, experiment):
    years, annual_mean = load_cmip6(model, experiment, 'zostoga')
    a = xa.DataArray(annual_mean, coords={'year': years}).loc[:2100]
    a -= a.loc[1995:2014].mean() # projection baseline (not really necessary here)
    a = a.assign_coords({'experiment':experiment})
    return a.loc[1900:2100]


def load_forcing_data_tas(model, experiment, obsclim='GSWP3-W5E5', smooth=True, mergeobs=True):

    hist = load_global_annual_mean('tas', "historical", model)
    future = load_global_annual_mean('tas', experiment, model)
    gmt = xa.concat([hist, future], dim='year').squeeze()

    if mergeobs:
        obs = load_global_annual_mean("tas", "historical", model=None, protocol='ISIMIP3a', obsclim=obsclim) - 273.15

        # baseline for merging
        y1, y2 = 1993, obs.year.values[-1]
        baseline = obs.loc[y1:y2].mean().item()    

        gmt -= gmt.loc[y1:y2].mean()
        gmt += baseline

        gmt = gmt.loc[1900:2100]
        gmt.loc[1901:y2].values[:] = obs.loc[1901:y2].values
        gmt.loc[1900] = obs.loc[1901:1920].mean().item() # fill-in the first data point

    if smooth:
        from scipy.signal import savgol_filter
        gmt.values[:] = savgol_filter(gmt.values, 21, 1)

    return gmt
