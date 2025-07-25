import concurrent.futures
import glob, os, shutil
from pathlib import Path
import yaml
import subprocess as sp
import re
import logging
import netCDF4 as nc
from .cmip6 import list_cmip6_files as list_cdc_files, parse_filename # alternative data source
from .cmip6 import fmtmodel, all_experiments, all_models, MODELS_MAP, DATA as data, logger

ZOSMASK = f"{data}/cmip6/zos/masks"


DEBUG = False
VERBOSE = False

logger.setLevel(logging.INFO)

if DEBUG or VERBOSE:
    logger.setLevel(logging.DEBUG)


def list_source_files(database=None, **kwargs):
    return list_cdc_files(**kwargs)


def get_path(mode, model, experiment, variable, database=None):
    root = Path(f"{data}/cmip6")
    return f"{root}/{variable}/{mode}/{variable}_{fmtmodel(model)}_{experiment}.nc"


def get_all_experiments(variable, model=None, database=None):
    """Get all experiments in regridded folder"""
    return list(sorted({f[:-3].split('_')[-1] for f in glob.glob(get_path('regridded', model or '*','*', variable, database=database)) }))


def get_all_models(variable, experiments=None, database=None, method="intersection"):
    """Get all models in regridded folder, which include all of provided experiments"""
    if experiments is None:
        experiments = get_all_experiments(variable, database=database)
        method = "union"
    elif type(experiments) is str:
        experiments = [experiments]

    modelset = getattr(set, method)(*(
        { Path(f).name[len(variable)+1:-(3+len(experiment)+1)]
            for f in glob.glob(get_path('regridded', '*',experiment, variable, database=database)) } for experiment in experiments))

    return [MODELS_MAP[model] for model in sorted(modelset)]


def cdo(cmd):
    logger.info("cdo "+cmd)
    sp.check_call("module load cdo; OMP_NUM_THREADS=16 cdo "+cmd, shell=True)
    # sp.check_call("cdo "+cmd, shell=True)

def demean(file_in, file_out, grid=None, missval=None):
    setmissval= f' -setmissval,{missval}' if missval is not None else ""
    setgrid = f'-setgrid,{grid}' if grid is not None else ""
    cdo(f"sub {setmissval} {file_in} -enlarge,{file_in} -fldmean {setgrid} {setmissval} {file_in} {file_out}")

def yearmean(file1, file2):
    cdo(f"yearmean {file1} {file2}")

def regrid(file1, file2, like=None, options="", method="remapbil"):

    if like is None:
        # like = "r720x360"  # [0, 0.5, ..., 359.5]
        import frederikse2020
        like = f"{frederikse2020.root}/steric.nc" # [0.25, 0.75..., 359.75]

    cdo(f"{method},{like} {options} {file1} {file2}")


def _get_regrid_options(model, realm):

    assert model != fmtmodel(model), "model must be provided with its original name"

    method = "remapbil"
    options = ""

    if realm == "ocean":
        if model in ["CNRM-ESM2-1", "CNRM-CM6-1"]:
            # cnrm_cm6_1_hr looks fine
            # https://code.mpimet.mpg.de/boards/1/topics/8676
            options="-selindexbox,2,361,2,293" # to avoid the white line
        elif model in ["CMCC-CM2-SR5", "CMCC-ESM2"]:
            # https://code.mpimet.mpg.de/boards/1/topics/8676
            options="-selindexbox,2,359,2,290" # to avoid the white line
            logger.warning('Antarctica is partly covered with sea due to interpolation issue TODO: use landseamask')
        elif model in [
            "CanESM5",
            "EC-Earth3",
            "EC-Earth3-AerChem",
            "EC-Earth3-CC",
            "EC-Earth3-Veg",
            "EC-Earth3-Veg-LR",
            ]:
            logger.warning(f'{model} use NN interpolation')
            method="remapnn"
        # elif "ipsl" in model.lower():
        elif model in ["IPSL-CM6A-LR"]:
            options="-selname,zos -selindexbox,2,361,2,331"
        elif model in ["MPI-ESM-1-2-HAM", "MPI-ESM1-2-LR"]:
            options="-selindexbox,2,256,1,220"# remove first column
        elif model in ["MPI-ESM1-2-HR"]:
            # logger.warning(f'{model}: remapcon, some small NaN dots remain in the Arctic')
            # options="-selindexbox,1,801,1,404" # drop only the last x value  (full grid 802 x 404)
            method = "remapnn" # using remapbil results in infinite values, and remapcon
        elif model in ["TaiESM1"]:
            logger.warning(f'{model}: remapnn (white band with bil), but some spurious wave-like structures remain in the Arctic')
            method = "remapnn"
        # elif model in ["icon_esm_lr", "awi_esm_1_1_lr", "awi_cm_1_1_mr",""]:
        elif model in ["ICON-ESM-LR", "AWI-ESM-1-1-LR", "AWI-CM-1-1-MR"]:
            method = "remapcon"

    return method, options


def regrid_cmip6(file1, file2, like=None):
    """Read model and realm metadata to derive the appropriate method (based on trial and error)
    """

    with nc.Dataset(file1) as ds:
        model = ds.source_id
        # variable = ds.variable_id
        realm = ds.realm

    method, options = _get_regrid_options(model, realm)

    return regrid(file1, file2, like=like, options=options, method=method)



def files_and_folders(model, experiment, variable, database=None):
    root = Path(f"{data}/cmip6")
    base = f"{variable}_{model}_{experiment}"
    yearly = Path(f"{root}/{variable}/yearly/{base}")
    regridded = Path(f"{root}/{variable}/regridded_parts/{base}")
    yearly_merged = Path(f"{root}/{variable}/yearly/{base}.nc")
    file_merged = Path(f"{root}/{variable}/regridded/{variable}_{model}_{experiment}.nc")
    return {"yearly":yearly, "yearly_merged":yearly_merged, "regridded":regridded, "file_merged": file_merged}

def cleanup(model, experiment, variable):
    d = files_and_folders(model, experiment, variable)
    if os.path.exists(d['yearly']):
        shutil.rmtree(d["yearly"])
    if os.path.exists(d['yearly_merged']):
        os.remove(d["yearly_merged"])
    if os.path.exists(d['regridded']):
        shutil.rmtree(d["regridded"])
    if os.path.exists(d["file_merged"]):
        os.remove(d["file_merged"])


# def _yield_ensembles():
#     for r in range(1,110+1): # 1 to 110 found, : put last to avoid losing time
#         for i in range(1,1+1):
#             for p in range(1, 5+1): # 1, 2, 4, 5 found
#                 for f in range(1, 4+1): # 1, 2, 3, 4 found
#                         yield f"r{r}i{i}p{p}f{f}"

# sorted_ensembles = [e for i, e in enumerate(_yield_ensembles()) if e < 100] # that should cover our use case

ensemble_digits = re.compile(r"r(?P<r>\d+)i(?P<i>\d+)p(?P<p>\d+)f(?P<f>\d+)")

def _get_ensemble_tuple(e):
    "return r, i, p, f"
    return tuple(int(i) for i in ensemble_digits.match(e).groups())

def sorted_ensembles(ensembles):
    return [e for t,e in sorted([(_get_ensemble_tuple(e), e) for e in ensembles])]


def _read_variant_label(f):
    with nc.Dataset(f) as ds:
        return ds.variant_label, (ds.parent_variant_label if hasattr(ds, 'parent_variant_label') else None)

def check_branching(model, variable, raise_error=False):
    """check branching of existing runs
    """
    historical, _ = _read_variant_label(get_path('regridded', model, 'historical', variable))

    for experiment in all_experiments:
        if experiment == "historical":
            continue
        try:
            future, parent = _read_variant_label(get_path('regridded', model, experiment, variable))
        except FileNotFoundError:
            continue
        assert parent is not None, (model, experiment, variable)
        if parent != historical:
            logger.warning(f"{model}: {experiment}'s expected parent: {parent} but we got historical: {historical}")
            if raise_error:
                raise ValueError(f'branching error: {model}, {experiment}')


def select_source_files(model, experiment, variable, **kwargs):
    experiments = {
        "pi_control": "piControl",
    }

    experiment = experiments.get(experiment, experiment)

    # works best with original, upper-case model
    if model == fmtmodel(model):
        models = [parse_filename(f)['model']
            for f in list_source_files(experiment=experiment, variable=variable, **kwargs)]
        model0 = models[[fmtmodel(m) for m in models].index(model)] # that's the original name
        model = model0

    listing = list_source_files(experiment=experiment, variable=variable, model=model, **kwargs)

    if len(listing) == 0:
        raise ValueError(f"{model}, {experiment}, {variable}, {kwargs}: No file found")

    parents = list(set([str(Path(file).parent) for file in listing]))
    if len(parents) != 1:
        assert kwargs.get('database') != "cds"  # only foreseen for climate_data_central (cds has only one ensemble variant)

        # Several ensembles? (that needs to come before "versions" !!)
        ensembles = [parse_filename(file)["ensemble"] for file in listing]
        if len(set(ensembles)) > 1:
            logger.debug(f"{model}, {experiment}, {variable}: {len(set(ensembles))} ensembles found: {set(ensembles)}")

            # Default choice: just pick the first ensemble (sorted)
            ensemble = sorted_ensembles(set(ensembles))[0]

            # For the historical experiment, check which parent the SSP585 scenario indicates, if any...
            if experiment == "historical":
                try:
                    files = select_source_files(model, "ssp585", variable, **kwargs)
                    e, p = _read_variant_label(files[0])
                    # If not present, that is not something we can solve automatically. Raise an error:
                    assert p in ensembles, f'{model}: "ssp585 scenario indicates {p} as parent, but only found: {ensembles}'
                    if p != ensemble:
                        logger.info(f"{model}, {experiment}: smallest available ensemble is {ensemble}, but ssp585 indicates {p} as parent.")
                    ensemble = p
                except ValueError:
                    # No SSP experiment is provided: OK
                    pass

            # ensemble = list(sorted(set(ensembles)))[0]
            logger.info(f"{model}, {experiment}, {variable}: select {ensemble} out of {len(set(ensembles))} ensembles")
            return select_source_files(model, experiment, variable, ensemble=ensemble, **kwargs)

        # In some case both gn and gr are provided: use gr !
        grids = [parse_filename(file)["grid"] for file in listing]
        if len(set(grids)) > 1:
            logger.info(f"{model} use regular grid")
            return select_source_files(model, experiment, variable, grid="gr", **kwargs)
            # try:
            #     print("!!", model, experiment, variable, "process natural grid gn")
            #     return select_source_files(model, experiment, variable, grid="gn", **kwargs)
            # except Exception as error:
            #     print("!!! Failed to process natural grid, try from interpolated grid")
            #     print("==>", model, experiment, variable, "process interpolated grid gr")
            #     cleanup(model, experiment, variable)
            #     return select_source_files(model, experiment, variable, grid="gr", **kwargs)


        # Cases where different versions exist (this needs to come last !!)
        versions = [parse_filename(file)["version"] for file in listing]
        if len(set(versions)) > 1:
            version = list(sorted(set(versions)))[-1]
            logger.info(f"""{model}, {experiment}, {variable}: pick {version} out of {' '.join(set(versions))}""")
            return select_source_files(model, experiment, variable, version=version, **kwargs)

        # print("!! More than one folder matches the request:", folders)
        raise ValueError("More than one folder matches the request: " + " ".join(parents))

    return listing


def process_cmip6(model, experiment, variable, remove_mean=False, database=None, **kwargs):

    model0 = model
    model = fmtmodel(model)

    d = files_and_folders(model, experiment, variable, database=database)
    yearly = d["yearly"]
    yearly_merged = d["yearly_merged"]
    # regridded = d["regridded"]
    file_merged = d["file_merged"]

    if file_merged.exists():
        logger.info(f"{file_merged} already exists")
        return

    if yearly_merged.exists():
        logger.info(f"{yearly_merged} already exists")
    else:
        listing = select_source_files(model0, experiment, variable, database=database, **kwargs)

        parents = list(set([str(Path(file).parent) for file in listing]))
        assert len(parents) == 1

        input_files = []

        # sub_files = _extract_files(file)
        for f in listing:
            # subfolder, base = Path(f).parts[-2:]

            file_yearly = yearly / Path(f).name

            if not file_yearly.exists():
                os.makedirs(yearly, exist_ok=True)
                yearmean(f, file_yearly)

            input_files.append(str(file_yearly))

        # merge all yearly data prior to regridding
        cdo(f"mergetime {' '.join(input_files)} {yearly_merged}")

        # remove singular yearly files
        shutil.rmtree(yearly) # the whole directory

    # regrid all at once
    os.makedirs(file_merged.parent, exist_ok=True)
    regrid_cmip6(yearly_merged, f"{file_merged}.tmp")

    maskfile = Path(ZOSMASK) / ('mask_'+fmtmodel(model)+'.nc')
    if maskfile.exists():
        logger.info(f"{model}, {experiment}, {variable}, mask inner seas with {maskfile}")
        cdo(f"div -selname,zos {file_merged}.tmp {maskfile} {file_merged}.tmp.masked")
        rename(f"{file_merged}.tmp.masked", f"{file_merged}.tmp")

    if remove_mean:
        # os.rename(f"{file_merged}", f"{file_merged}.withmean")
        if variable != "zos":
            raise NotImplementedError('demean only implemented for zos, check !')

        # Remove mean
        kw = {}
        if model == "kiost_esm":
            kw['missval'] = "nan"
        demean(f"{file_merged}.tmp", f"{file_merged}.tmp.demean", grid="r720x360", **kw)
        rename(f"{file_merged}.tmp.demean", f"{file_merged}.tmp")

    rename(f"{file_merged}.tmp", f"{file_merged}")

def rename(file1, file2):
    logger.debug(f"mv {file1} to {file2}")
    os.rename(file1, file2)
    # os.remove(file2)
    # shutil.copy(file1, file2)
    # os.remove(file1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", "--db", choices=["climate_data_central"], default="climate_data_central")
    parser.add_argument("--variable", required=True, choices=["zos", "tas"]) #, help=argparse.SUPPRESS)
    parser.add_argument("--domain")
    parser.add_argument("--model", nargs="+")
    parser.add_argument("--experiment", nargs="+")
    # parser.add_argument("--demean", action='store_true')
    parser.add_argument("--max-workers", default=5, type=int)
    parser.add_argument("--debug", action='store_true')
    # parser.add_argument("--ensemble", nargs="*")
    o = parser.parse_args()

    if o.debug:
        global DEBUG
        DEBUG = True
        logger.setLevel(logging.DEBUG)
        o.max_workers = 1

    if not o.experiment:
        if not o.model:
            o.experiment = all_experiments
        else:
            x = set()
            for model in o.model:
                x = x.union(parse_filename(f)['experiment'] for f in list_source_files(variable=o.variable, domain=o.domain or '*', model=model))
            o.experiment = sorted(x.intersection(all_experiments))
        logger.info(f'experiment: {" ".join(o.experiment)}')

    if not o.model:
        x = set()
        for experiment in o.experiment:
            x = x.union(parse_filename(f)['model'] for f in list_source_files(variable=o.variable, domain=o.domain or '*', experiment=experiment))
        o.model = sorted(x)
        logger.info(f'model: {" ".join(o.model)}')

    with concurrent.futures.ThreadPoolExecutor(max_workers=o.max_workers) as executor:
        # futures = { executor.submit(process_zip, f"ssh_{model}_{experiment}.zip") : (model, experiment) for model in models for experiment in experiments}
        futures = { executor.submit(process_cmip6, model, experiment, o.variable, remove_mean=o.variable=="zos", domain=o.domain or '*', database=o.database) : (model, experiment)
            for experiment in o.experiment for model in o.model}

        for future in concurrent.futures.as_completed(futures):
            zipfile = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.debug(str(exc))
                logger.warning(f'failed to process: {zipfile}')
                # print(f'failed to cdsroot: {model}: {experiment}')


if __name__ == "__main__":
    # test()
    main()
