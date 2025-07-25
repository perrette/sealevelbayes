import concurrent.futures
import glob, os, shutil
from pathlib import Path
import subprocess as sp
import numpy as np
import netCDF4 as nc
import logging
from .regrid import data, cdo, fmtmodel, get_path, get_all_models, get_all_experiments, logger
# from cmip6 import list_cmip6_files, parse_filename # alternative data source

DEBUG = False


def get_branching_timestep(model, variable, database=None):
    """Search the branching index in pre-industrial control run"""

    historical = get_path('regridded', model, 'historical', variable, database=database)
    control = get_path('regridded', model, 'piControl', variable, database=database)
    with nc.Dataset(historical) as h, nc.Dataset(control) as c:
        logger.debug(f"{model}: branch time in parent: {h.branch_time_in_parent}")
        logger.debug(f"{model}: branch time in child: {h.branch_time_in_child}")
        time = h.getncattr("branch_time_in_parent")
        if type(time) is str and time.endswith('D'): # found in ec_earth3, presumably for Double
            time = time[:-1]
        time = float(time)
        logger.debug(f"{model}: branch time: {time}")
        t = np.searchsorted(c["time"][:], time)
        logger.debug(f"{model}: branch index: {t}")

        if fmtmodel(model) == "kiost_esm":
            # There are issues in the time axis before that time
            hack = 64
            logger.info(f"{model}: branching time-step HACK: {t} => {hack} (total: {c['time'].size} time steps)")
            # Cause: issues before timestep 64)
            t = hack
        elif fmtmodel(model) == "icon_esm_lr":
            hack = 22
            logger.info(f"{model}: branching time-step HACK for {model}: {t} => {hack} (total: {c['time'].size} time steps)")
            t = hack
        elif fmtmodel(model) == "awi_esm_1_1_lr":
            hack = 0
            logger.info(f"{model}: branching time-step HACK for {model}: {t} => {hack} (total: {c['time'].size} time steps)")
            t = hack
        elif fmtmodel(model) == "norcpm1":
            hack = 0
            logger.info(f"{model}: branching time-step HACK for {model}: {t} => {hack} (total: {c['time'].size} time steps)")
            t = hack
        elif fmtmodel(model) == fmtmodel('EC-Earth3-Veg-LR'):
            hack = 0
            logger.info(f"{model}: branching time-step HACK for {model}: {t} => {hack} (total: {c['time'].size} time steps)")
            t = hack

        remaining = c['time'].size - t

        # We usually compute the trend from 1850 to 2100, i.e. 250 years
        if remaining < 250:
            logger.warning(f"{model}, {variable}: {remaining} < 250 years available for drift trend calculation")


    return t


def dedrift(model, experiment, variable, year1, year2, force=False, database=None):
    """Post-process the output from process_cmip6
    """
    file_out = get_path(f'dedrifted_{year1}-{year2}', model, experiment, variable, database=database)
    if not force and os.path.exists(file_out):
        logger.info(f"...{file_out} already exists")
        return

    file_in = get_path('regridded', model, experiment, variable, database=database)
    if not os.path.exists(file_in):
        # That is not really an error: the file simply does not exist: nothing to do
        logger.info(f"Input file {Path(file_in)} does not exist")
            # raise ValueError(f"Input file {Path(file_in)} does not exist")
        return

    file_pi = get_path('regridded', model, 'piControl', variable, database=database)
    if not os.path.exists(file_pi):
        raise ValueError(f"piControl input file {Path(file_pi)} does not exist")

    # Extract the linear trend if needed
    base_drift = Path(get_path(f'drifttrend_{year1}-{year2}', model, 'piControl', variable), database=database)
    file_a = Path(str(base_drift) + ".a")
    file_b = Path(str(base_drift) + ".b")
    if force or not file_a.exists() or not file_b.exists():
        os.makedirs(base_drift.parent, exist_ok=True)
        t0 = get_branching_timestep(model, variable, database=database)
        start_year = 1850 # we checked that all historical runs start in 1850
        i0, n = t0+year1-start_year, year2-year1+1
        # e.g. year1 = 1900, year2 = 2100 ==> i0 = t0 + 50, n = 201
        cdo(f'trend -seltimestep,{i0}/{i0+n-1} {file_pi} {file_a} {file_b}')

    # Retrieve the linear trend
    os.makedirs(Path(file_out).parent, exist_ok=True)
    # we add selname because otherwise there is an error with "area" variable for ipsl_cm6a_lr historical
    cdo(f'subtrend -selname,{variable} {file_in} {file_a} {file_b} {file_out}')

    # if the regridded file was properly demean, then no need to do that here


def dedrift_experiments(model, experiments, variable, year1, year2, force=False, database=None):
    """ serialize for one model and one variable, so that piControl drift is not computed several times
    """
    for experiment in experiments:
        try:
            dedrift(model, experiment, variable, year1, year2, force=force, database=database)
        except Exception as error:
            logger.error(str(error))
            if DEBUG:
                raise
            logger.error("=> Failed to dedrift: ", model, experiment, variable, database)
            continue


def main():
    global DEBUG
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", default="zos", choices=["zos"]) #, help=argparse.SUPPRESS)
    parser.add_argument("--model", nargs="+")
    parser.add_argument("--experiment", nargs="*")
    parser.add_argument("--database")
    parser.add_argument("--years", nargs=2, type=int, default=[1850, 2100])
    parser.add_argument("--max-workers", default=5, type=int)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--force", action='store_true', help='force recomputing even if final and intermediate files already exist')
    parser.add_argument("--check-branching", action='store_true', help='diagnose branching indices')
    # parser.add_argument("--ensemble", nargs="*")
    o = parser.parse_args()

    if o.debug:
        DEBUG = True
        logger.setLevel(logging.DEBUG)
        o.max_workers = 1

    if not o.experiment:
        o.experiment = get_all_experiments(o.variable, model, database=o.database)
        logger.info(f"experiments found: {' '.join(o.experiment)}")

    if not o.model:
        o.model = list(sorted(
            set(get_all_models(o.variable, experiments=['piControl'], database=o.database)).intersection(
                get_all_models(o.variable, experiments=['historical'], database=o.database) )))
        logger.info(f"models found: {' '.join(o.model)}")

    if o.check_branching:
        for model in o.model:
            # logger.setLevel(logging.DEBUG)
            get_branching_timestep(model, o.variable)
        parser.exit(0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=o.max_workers) as executor:
        futures = { executor.submit(dedrift_experiments, model, o.experiment, o.variable, *o.years, database=o.database, force=o.force) : model for model in o.model}

        for future in concurrent.futures.as_completed(futures):
            zipfile = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error(exc)
                logger.error(f'failed to process: {zipfile}')
                if DEBUG:
                    raise
                # print(f'failed to download: {model}: {experiment}')


if __name__ == "__main__":
    # test()
    main()