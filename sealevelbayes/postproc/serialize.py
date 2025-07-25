import itertools
import tqdm
import copy
from itertools import groupby, product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xa
import arviz
from scipy.stats import norm

from sealevelbayes.logs import logger
import sealevelbayes.datasets.frederikse2020 as frederikse2020
import sealevelbayes.datasets.ar6.supp
import sealevelbayes.datasets.ar6 as ar6
from sealevelbayes.datasets.shared import MAP_FRED, MAP_AR6
from sealevelbayes.models.localslr import SOURCES, FIELDS, DIAGS
from sealevelbayes.models.domain import get_stations_from_psmsl_ids

# median numbers in a dataframe
fred_df = frederikse2020.load_global()

ar6_85 = ar6.supp.load_global('ssp585')
ar6_26 = ar6.supp.load_global('ssp126')



def obsjs_from_trace(trace,
        model_names = ["tidegauge", "satellite", "gps"],
        obs_names = ["tidegauge_obs", "satellite_obs", "gps_obs"],
        mu_names = ["tidegauge_mu", "satellite_mu", "gps_mu"],
        sd_names = ["tidegauge_sd", "satellite_sd", "gps_sd"],
        sample_data=None, # posterior or prior or any such dataset
        constant_data=None,
        log_likelihood=None,
        sample_stats=None,
    ):

    obs = []

    if constant_data is None:
        constant_data = trace.constant_data

    if sample_data is None:
        for attr in ["posterior", "prior", "posterior_predictive", "prior_predictive"]:
            if attr in trace:
                if attr != "posterior":
                    logger.warning(f"Use trace {attr} field for observations. Pass sample_data explicitly to suppress this warning.")
                sample_data = trace[attr]
                break
        if sample_data is None:
            raise ValueError("sample_data is not provided (posterior, prior etc..)")

    if log_likelihood is None and trace is not None and 'log_likelihood' in trace:
        log_likelihood = trace.log_likelihood

    if sample_stats is None and trace is not None and 'sample_stats' in trace:
        sample_stats = trace.sample_stats

    if sample_stats is not None:
        total_loglik = arviz.extract(sample_stats['lp'])['lp']
        i_loglik = total_loglik.argsort()[::-1]
        i_mode = i_loglik[0]
        loglik = total_loglik[i_mode].item()

    else:
        i_mode = None
        loglik = None

    for name, obs_name, mu_name, sd_name in zip(model_names, obs_names, mu_names, sd_names):

        if name not in sample_data:
            logger.warning(f"{name} not in dataset. Skip.")
            continue

        try:
            obs_mu = constant_data[mu_name]

        except Exception as error:
            print(error)
            logger.warning(name)
            continue

        try:
            obs_sd = constant_data[sd_name]
        except Exception as error:
            print(error)
            logger.warning(name)
            obs_sd = obs_mu*0 + 1e-6  # 0

        obs_model = arviz.extract(sample_data[[name]])[name]
        obs_dist = norm(obs_mu.values, obs_sd.values)  # also works for array params
        med, lo, hi, lo99, hi99 = obs_model.quantile([.5, .05, .95, 0.005, 0.995], dim='sample')

        if log_likelihood is not None and obs_name in log_likelihood and i_mode is not None:
            obs_like = arviz.extract(log_likelihood[[obs_name]])[obs_name].isel(sample=i_mode).values
        else:
            obs_like = 0

        obs.append({
            'name': name if not name.endswith('_trend') else name[:-len('_trend')],
            'obs': obs_mu.values,
            'obs_sd': obs_sd.values,
            'obs_lower': obs_dist.ppf(0.05),
            'obs_upper': obs_dist.ppf(0.95),
            'gof': 0.5*((obs_model - obs_mu)**2/obs_sd**2).mean('sample').values,
            'log_likelihood': obs_like,
            'mean': obs_model.mean(dim='sample').values,
            'sd': obs_model.std(dim='sample').values,
            'median': med.values,
            'upper': hi.values,
            'lower': lo.values,
            'upper_99%': hi99.values,
            'lower_99%': lo99.values,
            **({'mode' : obs_model.isel(sample=i_mode).values} if i_mode is not None else {})
            # 'best10': obs_model.isel(sample=i_best10).values,
            # 'worst10': obs_model.isel(sample=i_worst10).values,
        })

    # return result as a list of serializable obs instance (one for each station) for back-compatibility

    obs2 = []
    for o in obs:
        for i in range(trace.posterior.station.size):
            o2 = {k:o[k] if np.ndim(o[k]) == 0 else o[k][i].item() for k in o}
            obs2.append({"station_index": i, **o2})


    key = lambda r: r["station_index"]
    return [{"obs": list(group), "log_likelihood": loglik} for k, group in itertools.groupby(sorted(obs2, key=key), key=key)]



def serialize_trace(trace=None, posterior=None,
                 sources=SOURCES+[f"{s}_trend" for s in SOURCES]+[f"{s}_obs" for s in SOURCES], fields=FIELDS, diags=DIAGS,
                 meta={}, key=None, posterior_predictive=None, prior_predictive=None, constant_data=None, obs_kw={},
                 quantiles=[.05, .167, .5, .833, .95], low=.05, high=.95,):

    assert low in quantiles, f"low quantile {low} must be in quantiles: {quantiles}"
    assert high in quantiles, f"high quantile {high} must be in quantiles: {quantiles}"

    if posterior is None and hasattr(trace, "posterior"):
        print("Merge chain and draw and posterior etc...")
        # if hasattr(trace, "prior") and not hasattr(trace, "posterior"):
        #     posterior = arviz.extract(trace.prior)
        # elif hasattr(trace, "posterior_predictive") and not hasattr(trace, "posterior"):
        #     posterior = arviz.extract(trace.posterior_predictive)
        # else:
        posterior = arviz.extract(trace.posterior)
        print("done")

    # also add any data from posterior_predictive (not so much for observations, but for time-series data or experiments sampled afterwards)
    if posterior_predictive is None and hasattr(trace, "posterior_predictive"):
        posterior_predictive = arviz.extract(trace.posterior_predictive)
    if prior_predictive is None and hasattr(trace, "prior_predictive"):
        prior_predictive = arviz.extract(trace.prior_predictive)

    psmsl_ids = None
    for dataset in [posterior, constant_data, posterior_predictive, prior_predictive] + (
        [getattr(trace, attr, None) for attr in ["posterior", "prior", "posterior_predictive", "prior_predictive"]] if trace is not None else []):
        if dataset is not None and "station" in dataset.coords:
            psmsl_ids = dataset.station.values
            break
    if psmsl_ids is None:
        logger.warning("No station dimension found in any dataset")
        stations = [-1]
    else:
        stations = get_stations_from_psmsl_ids(psmsl_ids)

    getyears = lambda ds: ds.year_output.values.tolist() if "year_output" in ds.dims else (ds.year.values.tolist() if 'year' in ds.dims else [])

    js = {}
    # js['stations'] = {k:v for k,v in stations.items() if not (isinstance(v, float) and np.isnan(v))}
    js['stations'] = stations
    js['years'] = getyears(posterior) if posterior is not None else []
    # js['sources'] = sources
    js['experiments'] = experiments = [] if posterior is None else posterior.experiment.values.tolist()
    js['samples'] = posterior.sample.size if posterior is not None else None
    js['records'] = records = []
    js.update({k: v for k, v in meta.items() if np.isfinite(v)}) # do not include NaNs (not handled in json)


    if hasattr(trace, "sample_stats"):
        i_loglik = arviz.extract(trace.sample_stats["lp"])["lp"].argsort().values[::-1]
        i_mode = i_loglik[0]
    else:
        i_loglik = None
        i_mode = None

    quantiles = list(quantiles)

    all_variables = [(f"{diag}_{field}_{source}", diag, field, source)
        for field in fields
            for source in sources
                for diag in diags]

    if posterior is None:
        missing_variables = []
    else:
        missing_variables = [k for k, diag, field, source in all_variables if k not in posterior]
    if missing_variables:
        missing_diag = set(diags).difference(diag for k, diag, field, source in all_variables if k in posterior)
        missing_field = set(fields).difference(field for k, diag, field, source in all_variables if k in posterior)
        missing_source = set(sources).difference(source for k, diag, field, source in all_variables if k in posterior)
        print(f"!! {len(missing_variables)} missing variables") #: {', '.join(missing_variables)}")
        if missing_source: print(f"!! missing source(s): {missing_source}")
        if missing_field: print(f"!! missing field(s): {missing_field}")
        if missing_diag: print(f"!! missing diag(s): {missing_diag}")

    def _get_precision(key=None, field=None, source=None, diag=None):
        if (source and source == "tas") or (key is not None and "tas" in key):
            return 3
        elif (diag and diag in ("rate", )) or (key is not None and "rate" in key):
            return 2
        else:
            return 1

    if posterior is not None:
        for field, source, diag in tqdm.tqdm(list(itertools.product(fields, sources, diags))):
            # precision = 0 if diag in ("change", "proj2100") else 1
            precision = _get_precision(None, field, source, diag)
            fmt = lambda v: v.round(precision).tolist()
            key = f"{diag}_{field}_{source}"
            if key not in posterior:
                continue
            array = posterior[key]

            # xarray indexing is too slow (180 ms for quantile, and indexing takes 0.5s per station)
            # use numpy instead for optimization
            # mid, low, high = array.quantile([.5, .05, .95], dim="sample")
            quantiles_values = np.percentile(array.values, np.asarray(quantiles)*100, axis=array.dims.index("sample"));
            midval, lowval, highval = (quantiles_values[quantiles.index(q)] for q in (.5, low, high))

            if i_mode is not None:
                mode = array.isel(sample=i_mode).values

            def select(values, i_station, i_experiment):
                idx = _build_mdindex(i_station, i_experiment)
                return values[idx]

            def _build_mdindex(i_station, i_experiment):
                idx = ()
                for dim in array.dims:
                    if dim == "sample": continue # was squeezed out via np.percentile
                    elif dim == "station": idx = idx + (i_station,)
                    elif dim == "experiment": idx = idx + (i_experiment,)
                    else: idx = idx + (slice(None),)
                return idx

            for experiment in experiments:

                i_experiment = array.experiment.values.tolist().index(experiment)

                for i_station, station in enumerate(stations):
                    if "station" not in array.dims and i_station > 0:
                        continue  # global record

                    records.append({
                        'station': station.get("PSMSL IDs", f"#{i_station}") if field != "global" else -1,
                        'station_index': i_station if field != "global" else -1,
                        'diag': diag,
                        'field': field or "global",  # None means global
                        'source': source,
                        'experiment': experiment,
                        'type': 'posterior',
                        # 'mean': fmt(mean),
                        'median': fmt(select(midval, i_station, i_experiment)),
                        'lower': fmt(select(lowval, i_station, i_experiment)),
                        'upper': fmt(select(highval, i_station, i_experiment)),
                        'quantile_levels': quantiles,
                        'quantile_values': [fmt(select(qval, i_station, i_experiment)) for qval in quantiles_values],
                        })

                    if i_mode is not None:
                        records[-1]['mode'] = fmt(select(mode, i_station, i_experiment)) # single best run


    # Here we pack satellite_obs, tidegauge_obs and gps_obs's posterior predictive samples
    def do_predictive(predictive, type):
        records = []
        for k in tqdm.tqdm(predictive):
            array = predictive[k]
            if "station" not in array.dims:
                logger.warning("only posterior or prior predictive with 'Station' dimension can be defined here")
                continue  # global record

            # all_q = [.5, .05, .95]
            # mid, low, high = array.quantile([.5, .05, .95], dim="sample")
            quantiles_values = np.quantile(array.values, quantiles, axis=array.dims.index("sample"));
            midval, lowval, highval = (quantiles_values[quantiles.index(q)] for q in (.5, low, high))

            # select = lambda array, i_station: array.isel(station=i_station)
            def select(values, i_station): return values[i_station]
                # return np.take(values, i_station, axis=array.dims.index("station"))

            precision = _get_precision(k)
            fmt = lambda v: v.round(precision).tolist()

            for i_station, station in enumerate(stations):

                records.append({
                    'station': station.get("PSMSL IDs", f"#{i_station}"),
                    'station_index': i_station,
                    'name': k,
                    'type': type,
                    'diag': None,
                    'field': None,
                    'source': None,
                    'experiment': None,
                    'median': fmt(select(midval, i_station)),
                    'lower': fmt(select(lowval, i_station)),
                    'upper': fmt(select(highval, i_station)),
                    'quantile_levels': quantiles,
                    'quantile_values': [fmt(select(qval, i_station)) for qval in quantiles_values],

                    })

                if i_mode is not None:
                    records[-1]['mode'] = fmt(select(array.isel(sample=i_mode).values, i_station)) # single best run
        return records

    if posterior_predictive is not None:
        js["posterior_predictive"] = do_predictive(posterior_predictive, type="posterior_predictive")

    if prior_predictive is not None:
        js["prior_predictive"] = do_predictive(prior_predictive, type="prior_predictive")

    js['fields'] = list(sorted(set(r['field'] for r in records)))
    js['sources'] = list(sorted(set(r['source'] for r in records)))
    js['diags'] = list(sorted(set(r['diag'] for r in records)))

    js["global"] = [r for r in js["records"] if r["field"] == "global"]
    js["records"] = [r for r in js["records"] if r["field"] != "global"]

    try:
        js['obs'] = obsjs_from_trace(trace, constant_data=constant_data, **obs_kw)
        for o, station in zip(js['obs'], stations):
            for oo in o['obs']:
                del oo["station_index"]
                oo['station'] = station["PSMSL IDs"]
    except AssertionError:
        raise
    except Exception as error:
        # raise
        logger.warning(str(error))
        logger.warning("=> Cannot derive 'obs' field")
        # raise

    return js


def split_stations(js):
    all_records = []
    key = lambda r: r["station_index"]
    global_js = None

    for i, group0 in groupby(sorted(js['records']+js['global']+js.get("posterior_predictive",[])+js.get("prior_predictive", []), key=key), key=key):
        j2 = {}
        j2['station'] = js["stations"][i]

        # split in records, posterior_predictive, prior_predictive
        kk = lambda r: r.get("type", "records")
        for type, group in groupby(sorted(group0, key=kk), key=kk):
            key = "records" if type == "posterior" else type
            j2[key] = list(group)


        j2['fields'] = js['fields']
        j2['sources'] = js['sources']
        j2['years'] = js['years']
        j2['experiments'] = js['experiments']
        j2['samples'] = js['samples']
        j2['diags'] = js['diags']

        if "obs" in js and js['obs']:
            j2["obs"] = js["obs"][i]

        if i != -1:
            all_records.append(j2)
        else:
            global_js = j2
            global_js["station"] = {"Station names": "global"}

    return all_records, global_js


def trace_to_json(trace, **kwargs):
    return split_stations(serialize_trace(trace, **kwargs))


def _concat_json(list_of_stations_js):

    all_stations_js = [{"records": [], "years": r["years"]} for r in list_of_stations_js[0]]

    for stations_js in list_of_stations_js:
        for js, newjs in zip(all_stations_js, stations_js):
            js["records"].extend(newjs["records"])

    for js in all_stations_js:
        js["experiments"] = sorted(set(r['experiment'] for r in js["records"]))
        js["fields"] = sorted(set(r['field'] for r in js["records"]))
        js["sources"] = sorted(set(r['source'] for r in js["records"]))

    return all_stations_js

def resample_as_json(tr, psmsl_ids, experiments, fields, sources, split_resampling=False, **kw):
    if split_resampling:
        return _concat_json([
            resample_as_json(tr, [psmsl_id], experiments, fields, sources)
                for psmsl_id in psmsl_ids], **kw)

    idata = tr.resample_posterior(experiments=experiments, fields=fields, sources=sources, psmsl_ids=psmsl_ids, **kw)
    stations_js, _ = trace_to_json(tr.trace.sel(station=psmsl_ids), posterior=arviz.extract(idata.posterior))
    return stations_js


def js_to_array(js, source='total', field='global', experiment='ssp585', diag='change', year_dim='year'):
    recs = [r for r in js['records'] if r['diag'] == diag and r['field'] == field and r['experiment'] == experiment and r['source'] == source]
    assert len(recs) == 1, repr(recs)
    r = recs[0]
    return xa.DataArray([r['median'], r['lower'], r['upper']], coords={'quantile': [.5, .05, .95], year_dim: js['years']}, dims=('quantile', year_dim))



def interpolate_js(js, years=None, inplace=False):
    years0 = js['years']
    if years is None:
        years = np.arange(1900, 2100+1)

    if len(years0) == len(years) and years[0] == years0[0] and years[-1] == years0[-1]:
        return js

    if not inplace:
        js = copy.deepcopy(js)

    for r in js['records']:
        for field in r:
            if type(r[field]) is list:
                if len(r[field]) == len(years0):
                    r[field] = np.interp(years, years0, r[field]).tolist()
                else:
                     v = np.asarray(r[field])
                     if v.ndim == 2 and v.shape[1] == len(years0):
                        r[field] = [np.interp(years, years0, vq).tolist() for vq in v]


    js['years'] = np.asarray(years).tolist()

    return js




def get_record(recs, experiment, source, field, diag):
    assert len(recs) > 0, "Please provide a valid record list."
    filtered_recs = [r for r in recs if r['diag'] == diag and r['field'] == field and r['experiment'] == experiment and r['source'] == source]
    if len(filtered_recs) == 0:
        for field, value in [('diag', diag), ('field', field), ('experiment', experiment), ('source', source)]:
            values = sorted(set(r[field] for r in recs))
            if value not in values:
                raise ValueError(f"Record {field}: {repr(value)} not found. Existing values are: {', '.join(repr(v) for v in values)}")
        raise ValueError(f"Record not found for {experiment}, {source}, {field}, {diag} (each of these exist in the records though, just not together)")

    if len(filtered_recs) != 1:
        for rec in filtered_recs:
            logger.info(f"Record: {rec}")
        raise ValueError(f"Expected 1 record for {experiment}, {source}, {field}, {diag}, but found {len(filtered_recs)}")

    return filtered_recs[0]


def get_model_quantiles(data, experiment, source, field, diag, quantiles=[.05, .5, .95]) -> np.ndarray:
    if isinstance(data, dict):
        data = data['records']

    if isinstance(data, list):
        record = get_record(data, experiment, source, field, diag)
        if "quantile_levels" not in record:
            quantile_levels = [.05, .5, .95]
            quantile_values = [record["lower"], record["median"], record["upper"]]
        else:
            quantile_levels = record["quantile_levels"]
            quantile_values = record["quantile_values"]

        try:
            return np.array([quantile_values[quantile_levels.index(q)] for q in quantiles])
        except ValueError:
            logger.error(f"Available quantile levels: {quantile_levels}")
            raise

    if isinstance(data, xa.Dataset):
        variable = f"{diag}_{field}_{source}"
        data = data[variable]

    data = data.sel(experiment=experiment)

    sample_dim = [d for d in ["sample", "draw", "chain"] if d in data.dims]
    return data.quantile(quantiles, dim=sample_dim).squeeze().values
    # raise NotImplementedError(f"Got {type(data)}. get_quantiles only implemented for dict or list of records currently.")