import concurrent.futures
import glob, os
from pathlib import Path
from .regrid import data, cdo, fmtmodel, get_path, get_all_experiments, get_all_models, logger
# from cmip6 import list_cmip6_files, parse_filename # alternative data source

DEBUG = False

def global_mean(model, experiment, variable, database=None, force=False):
    file_in = Path(get_path('regridded', model, experiment, variable, database=database))
    folder = file_in.parent.parent / 'globalmean'
    file_out = folder/file_in.name
    if not file_in.exists():
        raise ValueError(f"{model}, {experiment}: {file_in} does not exist")
    if file_out.exists():
        logger.info(f"{model}, {experiment}: {file_out} already exists")
        return
    os.makedirs(folder, exist_ok=True)
    fldmean(file_in, file_out, grid='r720x360')

def fldmean(file_in, file_out, grid=None, missval=None):
    setmissval= f' -setmissval,{missval}' if missval is not None else ""
    setgrid = f'-setgrid,{grid}' if grid is not None else ""
    cdo(f"fldmean {setgrid} {setmissval} {file_in} {file_out}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--variable")
    parser.add_argument("--model", nargs="+")
    parser.add_argument("--experiment", nargs="+")
    parser.add_argument("--database", "--db", default="climate_data_central")
    parser.add_argument("--max-workers", default=5, type=int)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--force", action='store_true')
    # parser.add_argument("--ensemble", nargs="*")
    o = parser.parse_args()

    if o.debug:
        global DEBUG
        DEBUG = True

    if not o.experiment:
        o.experiment = get_all_experiments(o.variable, database=o.database)
        print("experiments found:"," ".join(o.experiment))

    if not o.model:
        o.model = get_all_models(o.variable, o.experiment, database=o.database)
        print("model found:"," ".join(o.model))

    with concurrent.futures.ThreadPoolExecutor(max_workers=o.max_workers) as executor:
        futures = { executor.submit(global_mean, model, experiment, o.variable, database=o.database, force=o.force) : model
            for model in o.model for experiment in o.experiment}

        for future in concurrent.futures.as_completed(futures):
            zipfile = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(exc)
                print(f'failed to process: {zipfile}')
                if DEBUG:
                    raise
                # print(f'failed to download: {model}: {experiment}')


if __name__ == "__main__":
    # test()
    main()