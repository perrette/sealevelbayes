#!/usr/bin/env python3
"""Same from existing trace, but request more output diagnostic and experiments
"""
from pathlib import Path
import shutil
import numpy as np
import xarray as xa
import gzip
import json
import tqdm
import subprocess
from contextlib import nullcontext
import zipfile
import pandas as pd
import io

import numpy as np
import pymc as pm
import concurrent.futures

from sealevelbayes.config import logger, get_runpath, get_webpath
from sealevelbayes.postproc.zenodo import upload_zenodo, DEPOSITION_ID

CIRUNS = get_runpath()
WEBDIR = get_webpath()
WEB_EXPERIMENTS = [
    "ssp126_mu", "ssp585_mu",
    "ssp126", "ssp585",
    'SP', 'GS', 'CurPol',
    'SP_mu', 'GS_mu', 'CurPol_mu',
    ]
SOURCES = ["steric", "glacier", "gis", "ais", "landwater", "vlm", "gia", "vlm_res", "total", "tas"]
DIAGS = ["change", "rate"]
FIELDS = ["rsl", "rad", "gsl", "global"]
QLEVS = [.05, .167, .5, .833, .95]

def iter_variables(diags, fields, sources):
    for diag in diags:
        for field in fields:
            for source in sources:
                if source == "tas" and field != "global":
                    continue
                if field == "global":
                    if source in ["vlm", "gia", "vlm_res"]:
                        continue
                yield {"diag":diag, "field":field, "source":source, "variable": f"{diag}_{field}_{source}"}

def get_variables(diags, fields, sources):
    variables = [r["variable"] for r in iter_variables(diags, fields, sources)]
    return variables

def get_model_variables(cirun=None, psmsl_ids=None, experiments=WEB_EXPERIMENTS, frequency=1,
        sources=SOURCES,
        diags=["change", "rate"],
        fields=["rsl", "rad", "gsl", "global"],
        tr=None,
        no_model=False,
        ):

    variables = get_variables(diags, fields, sources)

    if no_model:
        return nullcontext(), None, variables

    if tr is None:
        from sealevelbayes.postproc.run import ExperimentTrace
        tr = ExperimentTrace.load(cirun)
    else:
        cirun = tr.cirun

    if cirun is None:
        raise ValueError("No cirun provided")

    model = tr.get_model(diags=diags, experiments=experiments, fields=fields,
                    sources=sources, year_frequency=frequency, psmsl_ids=psmsl_ids)
    trace_free_RVs = tr.get_free_RVs_trace(model)

    return model, trace_free_RVs, variables

def filecheck(filename, variable):
    actions = []
    with xa.open_dataset(filename, decode_times=False) as ds:
        if variable not in ds.variables:
            actions.append("delete")

        elif len(ds) > 1:
            actions.append("clean")

    if "delete" in actions:
        logger.warning(f"Variable {variable} not found in {filename}, deleting")
        filename.unlink()
        return

    if "clean" in actions:
        logger.warning(f"File {filename} has multiple variables, cleaning up")
        filetmp = filename.with_suffix(".nc.tmp")
        shutil.copyfile(filename, filetmp)
        cmd = ["ncks", "-v", variable, str(filetmp), str(filename)]
        logger.info(" ".join(cmd))
        subprocess.run(cmd, check=True)
        remove_attrs(str(filename), [("global", "history"), ("global", "history_of_appended_files")])

    # remove _FillValue in case it exists
    remove_attrs(filename, [("quantile", "_FillValue")])

def filescheck(variables, postfolder, check=True):

    for variable in variables:
        filename = postfolder/f"quantiles_{variable}.nc"
        if check:
            if filename.exists():
                filecheck(filename, variable)

        if not filename.exists():
            yield variable

def edit_attrs(filename, attribute_vars):
    cmd = ["ncatted"]
    for attr,var,mode,type,value in attribute_vars:
        cmd.extend(["-a", f"{attr},{var},{mode},{type},{value}"])
    cmd.append(str(filename))
    logger.info(" ".join(cmd))
    subprocess.run(cmd, check=True)

def remove_attrs(filename, attribute_vars):
    return edit_attrs(filename, [(attr, var, "d", "", "") for var, attr in attribute_vars])

def check_final_attrs(filename, **kwargs):
    actions = [[a, "global", "d", "", ""] for a in ["history", "history_of_appended_files"]]
    for k, v in kwargs.items():
        actions.append([k, "global", "c", "c", v])
    edit_attrs(filename, actions)


def resample_quantiles(model, trace, variables, postfolder=None, overwrite=False, return_inferencedata=True, qlevs=QLEVS, metadata={}, **kwargs):

    datasets = {}

    if postfolder is not None:
        filename_all = postfolder / "quantiles_all.nc"
        if filename_all.exists() and not overwrite:
            logger.info(f"File {filename_all} already exists, skipping")
            check_final_attrs(filename_all, **metadata)
            return filename_all

    with model:
        for variable in variables:
            if postfolder is not None:
                filename = postfolder/f"quantiles_{variable}.nc"
                datasets[variable] = filename
                if filename.exists():
                    filecheck(filename, variable)
                if filename.exists() and not overwrite:
                    logger.info(f"File {filename} already exists, skipping")
                    continue
                logger.info(f"Produce... {filename}")

            idata = pm.sample_posterior_predictive(trace=trace, var_names=[variable], return_inferencedata=return_inferencedata, **kwargs)
            if return_inferencedata:
                data = idata.posterior_predictive[variable].quantile(qlevs, dim=["draw", "chain"])
            else:
                data = np.quantile(idata[variable], qlevs, axis=(0, 1))
                data = xa.DataArray(data, dims=["quantile", "experiment", "year", "station"],
                                    coords={"quantile": qlevs,
                                            "experiment": list(idata.coords["experiment"]),
                                            "year": list(model.coords["year"]),
                                            "station": list(model.coords["station"])})
            del idata
            if postfolder is None:
                datasets[variable] = data
            else:
                logger.info(f"Saving {filename}")
                encoding = {
                    variable: {
                        "zlib": True,
                        "complevel": 4,
                    },
                    "quantile": { "_FillValue": None }, # for some reason, this causes conflict when appending with ncks
                }
                data.to_netcdf(filename, encoding=encoding)

    if postfolder is None:
        return xa.Dataset(datasets)

    if datasets:
        filenames = list(datasets.values())

        logger.info("Merging files with ncks")
        shutil.copyfile(filenames[0], filename_all)
        for fn in filenames[1:]:
            cmd = ["ncks", "-A", str(fn), str(filename_all)]
            logger.info(" ".join(cmd))
            subprocess.run(cmd, check=True)

        check_final_attrs(filename_all, **metadata)

    else:
        logger.warning("No files to merge")

    logger.info("Finished successfully, exit")

    return filename


def model_is_needed(o):
    variables = get_variables(o.diags, o.fields, o.sources)

    if not o.overwrite:
        postfolder = get_runpath(o.cirun) / "postproc"
        if postfolder.exists():
            todo = list(filescheck(variables, postfolder, check=False))
            if not todo:
                return False

    return True

def create_csv(ds, station, i, diags=["change"], fields=["rsl", "gsl", "rad"], sources=SOURCES, experiments=None, cirun=None, percentiles=[50, 5, 95], digits=1):
    # CSV with columns: station_id,source,scenario,diagnostic,field,runid,1900,1910,...,2100
    if cirun is None:
        cirun = ds.attrs["runid"]
    iquantiles = np.array([ds.coords["quantile"].values.tolist().index(p/100) for p in percentiles])
    allyears = ds.coords["year"].values.tolist()
    decades = list(range(1900, 2101, 10))
    iyears = np.array([allyears.index(y) if y in allyears else -1 for y in decades])
    miss = (iyears == -1)
    nmiss = miss.sum()
    if not miss.any():
        select_years = lambda data: data[iyears]

    # last year is missing
    elif nmiss == 1:
        assert allyears[-1] == 2099
        assert iyears[-1] == -1
        def select_years(data):
            dat = data[iyears]
            dat[-1] += data[-1] - data[-2]
            return dat

    else:
        raise ValueError(f"Missing years {np.array(decades)[miss]} in {station} ({i})")

    metadata = list(iter_variables(diags, fields, sources))

    if experiments is None:
        experiments = ds.coords["experiment"].values.tolist()

    n = len(metadata) * len(experiments) * len(percentiles)
    dtypes = {
        "runid": object,
        "station_id": int if station != "global" else object,
        "source": object,
        "scenario": object,
        "diag": object,
        "field": object,
        "percentile": int,
        **{i: float for i in decades},
    }

    columns = {c:np.empty(n, dtype=dtypes[c]) for c in dtypes}

    index = -1
    for meta in metadata:
        v = ds[meta["variable"]]

        if meta["field"] != "global":
            v = v.isel(station=i)

        for j, experiment in enumerate(experiments):
            for k, pct in enumerate(percentiles):
                index += 1
                columns["runid"][index] = cirun
                columns["station_id"][index] = station
                columns["scenario"][index] = experiment
                columns["diag"][index] = meta["diag"]
                columns["field"][index] = meta["field"]
                columns["source"][index] = meta["source"]
                columns["percentile"][index] = int(pct)
                # values[np.arange(index+1, index+1+len(experiments)), :] = station
                data = select_years(v.isel(experiment=j, quantile=iquantiles[k]).values)
                for year, value in zip(decades, data):
                    columns[year][index] = value

    df = pd.DataFrame(columns)
    for year in decades:
        df[year] = df[year].round(digits)

    return df


def netcdf_to_csv_archive(cirun, overwrite=False, prefix=""):
    # ncfile = get_runpath(cirun) / "postproc" / "quantiles_all.nc"
    filename2 = get_runpath(cirun) / "postproc" / "quantiles_per_station.zip"

    if filename2.exists() and not overwrite:
        logger.info(f"File {filename2} already exists, skipping")
        return filename2

    # Create a zip archive and write CSV inside it
    with zipfile.ZipFile(filename2, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:

        for filename in netcdf_to_csv(cirun, overwrite=overwrite):
            logger.info(f"Adding {filename} to {filename2}")
            entry = prefix + filename.name
            zf.write(filename, arcname=entry)

    return filename2


def netcdf_to_csv(cirun, overwrite=False):
    ncfile = get_runpath(cirun) / "postproc" / "quantiles_all.nc"
    folder = get_webpath(cirun) / "data"

    with xa.open_dataset(ncfile, decode_times=False) as ds:

        entry = f"quantiles_global.csv"
        filename = folder / entry

        if filename.exists() and not overwrite:
            logger.info(f"File {filename} already exists, skipping")

        else:
            df = create_csv(ds, "global", 0, fields=["global"])
            logger.info(f"Writing global to {filename}")
            df.to_csv(filename, index=False)

        yield filename

        for i, station in enumerate(ds.coords["station"].values):
            entry = f"quantiles_{station}.csv"
            filename = folder / entry

            if filename.exists() and not overwrite:
                logger.info(f"File {filename} already exists, skipping")

            else:
                df = create_csv(ds, station, i)
                logger.info(f"Writing {station} to {filename}")
                df.to_csv(filename, index=False)

            yield filename


def _run(o):
    # don't define the model if all files are already present
    o.no_model = not model_is_needed(o)
    model, trace, variables = get_model_variables(o.cirun, psmsl_ids=o.psmsl_ids, experiments=o.experiments,
        diags=o.diags, fields=o.fields, sources=o.sources,
        frequency=o.frequency, no_model=o.no_model)
    postfolder = get_runpath(o.cirun) / "postproc"
    postfolder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "author" : "Mahé Perrette (mahe.perrette@gmail.com)",
        "creation_date": str(np.datetime64("now")),
        "runid": o.cirun,
    }
    filename = resample_quantiles(model, trace, variables, postfolder=postfolder, overwrite=o.overwrite, metadata=metadata)

    filename_csv = netcdf_to_csv_archive(o.cirun, overwrite=o.overwrite)

    # create the csv files (one per station)
    if o.upload_zenodo_csv or o.upload_zenodo:
        upload_zenodo(str(filename_csv), deposit_id=o.zenodo_deposit_id, target=f"webapp/{o.cirun}/quantiles_per_station.zip".replace("/", "__"))

    if o.upload_zenodo:
        upload_zenodo(str(filename), deposit_id=o.zenodo_deposit_id, target=f"netcdf/{o.cirun}/quantiles_all.nc".replace("/", "__"))


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cirun")
    parser.add_argument("--experiments", default=WEB_EXPERIMENTS, nargs="+")
    parser.add_argument("--diags", default=DIAGS, nargs="+")
    parser.add_argument("--fields", default=FIELDS, nargs="+")
    parser.add_argument("--sources", default=SOURCES, nargs="+")
    parser.add_argument("--quantiles", default=QLEVS, type=float, nargs="+")
    parser.add_argument("--psmsl-ids", nargs="*", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--no-save-timeseries", action="store_false", dest="save_timeseries")
    parser.add_argument("--batch", default=10, type=int)
    parser.add_argument("--raise-error", action='store_true')
    parser.add_argument("--save-trace", action='store_true')
    parser.add_argument("--no-save-json", action='store_false', dest="save_json")
    parser.add_argument("--frequency", default=1, type=int)
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--upload-zenodo", action='store_true')
    parser.add_argument("--upload-zenodo-csv", action='store_true')
    parser.add_argument("--zenodo-deposit-id", default=DEPOSITION_ID)
    return parser


def run(cirun, **kwargs):
    parser = get_parser()
    o = parser.parse_args([str(cirun)])
    vars(o).update(kwargs)
    _run(o)

def main(*args):
    parser = get_parser()
    o = parser.parse_args(*args)
    _run(o)

if __name__ == "__main__":
    main()