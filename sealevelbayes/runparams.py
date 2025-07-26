#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import json
from pathlib import Path

from sealevelbayes.runutils import get_default_kwargs
from sealevelbayes.config import config_parser, get_version, get_runpath
from sealevelbayes.logs import log_parser, logger
# from sealevelbayes.models.domain import coords_to_ids


def coords_to_ids(coordinates, round=1):
    """transform an arbitrary coordinate arrays into an ID

    e.g. lon = -23.19, lat = 76.00
    =>   lon = 336.8, colat = 90 - lat = 14.0   (wrap lon and use colat to avoid negative values, and round)
    =>   ID = 33680140
    """
    import numpy as np
    coordinates = np.asarray(coordinates)
    if coordinates.size == 0: return []
    lons, lats = coordinates.T
    lons = np.where(lons < 0, lons + 360, lons)
    colats = 90 - lats
    f = lambda l : (l.round(round)*10**round).astype(int)
    lons_scale = int(10**(round + 3))
    return f(1000+lons)*lons_scale + f(colats)

def id_to_coord(id):
    sid = str(id)[1:]
    n = len(sid)
    round = n//2 - 3
    return (int(sid[:n//2])*10**(-round), 90 - int(sid[n//2:])*10**(-round))


AVAILABLE_EXPERIMENTS = [
    'ssp119_mu',
  'ssp126_mu',
  'ssp245_mu',
  'ssp370_mu',
  'ssp585_mu',
  'Ren_mu',
  'SP_mu',
  'LD_mu',
  'Ren-2.0_mu',
  'GS_mu',
  'Neg-2.0_mu',
  'Neg_mu',
  'ModAct_mu',
  'CurPol_mu',
  'ssp119',
  'ssp126',
  'ssp245',
  'ssp370',
  'ssp585',
  'Ren',
  'SP',
  'LD',
  'Ren-2.0',
  'GS',
  'Neg-2.0',
  'Neg',
  'ModAct',
  'CurPol',
  'isimip_ssp126',
  'isimip_ssp370',
  'isimip_ssp585']

MM21_FORCING = ['CRU TS 4.03', 'ERA20C', '20CRV3', 'CFSR', 'JRA55', 'ERA5', 'MERRA2', 'ERA_Interim', '(Mean input)', '(Median input)']

# get default args programatically from globalslr to avoid duplicates
def get_all_defaults(static=False):
    defs = {}
    defs["antarctica"] = get_default_kwargs("sealevelbayes.models.globalslr", "get_antarctica_model", static=static)
    defs["greenland"] = get_default_kwargs("sealevelbayes.models.globalslr", "get_greenland_model", static=static)
    defs["steric"] = get_default_kwargs("sealevelbayes.models.globalslr", "get_steric_model", static=static)
    defs["glacier"] = get_default_kwargs("sealevelbayes.models.globalslr", "get_regional_glacier_models", static=static)
    defs["noise"] = get_default_kwargs("sealevelbayes.models.generic", "SourceModel._generate_noise", static=static)
    return defs


SOURCES_ALIAS_MAPPING = {"greenland":"gis", "antarctica":"ais"}

BASINS_MAP = {
    1: 'South Atlantic',
    2: 'Indian Ocean - South Pacific',
    3: 'East Pacific',
    4: 'Northeast Atlantic',
    5: 'Northwest Atlantic',
    6: 'Northwest Pacific',
    8: 'Mediterranean'}

GLOBAL_SOURCES = ['gis', 'ais', 'steric', 'glacier', 'landwater']

def get_watermark():
    import watermark
    return str(watermark.watermark(packages=",".join([
        "sealevelbayes", "pymc", "pytensor", "arviz", "numpy", "xarray", "pandas", "cloudpickle",
        "netCDF4", "h5py", "scipy", "statsmodels", "bottleneck", "flatdict"])))

def _source_short(name):
    return SOURCES_ALIAS_MAPPING.get(name, name)

def _source_long(name):
    return {v:k for k,v in SOURCES_ALIAS_MAPPING.items()}.get(name, name)

def _get_param(o, key, source):
    " get the parameter for one source, with a fallback to the global parameter "
    source_val = getattr(o, f"{source}_{key}", None)
    global_val = getattr(o, f"{key}", None)
    return source_val if source_val is not None else global_val

def random_hex(obj, n=6):
    obj = hashlib.sha256(json.dumps(obj, sort_keys=True).encode('utf-8'))
    return obj.hexdigest()[:n]

dist_mapping = {
        "HalfNormal": "HN", "Exponential": "E", "Normal": "N", "ConstantData": "cst", "Data": "cst",
        "Uniform": "U", "HalfCauchy": "HC", "Cauchy": "C", "StudentT": "T", "LogNormal": "LN",
        "ExpQuad": "EQ", "Matern32": "M32", "Matern52": "M52", "Matern12": "M12", "WhiteNoise": "WN"}

def get_noise_tag_parts(o, source):
    """Return tag parts for noise attribution (keep order to unique tag)
    """
    parts = []

    # rate or cumulative SLR?
    if getattr(o, f"{source}_noise_on_rate", None) or getattr(o, "noise_on_rate"):
        parts.append((1, "rate"))

    gp_cov = getattr(o, f"{source}_noise_gp_cov", None) or o.noise_gp_cov
    gp_cov_params = getattr(o, f"{source}_noise_gp_cov_params", None) or o.noise_gp_cov_params
    if gp_cov_params:
        gp_cov_params_tag = "-".join(map(str, gp_cov_params))
    else:
        ls_dist = getattr(o, f"{source}_noise_ls_dist", None) or o.noise_ls_dist
        ls_dist_params = getattr(o, f"{source}_noise_ls_dist_params", None) or o.noise_ls_dist_params
        gp_cov_params_tag = f"-{dist_mapping.get(ls_dist, ls_dist.lower())}{'-'.join(map(str, ls_dist_params))}"

    gp_tag = f"{dist_mapping.get(gp_cov, gp_cov.lower())}{gp_cov_params_tag}"
    parts.append(((2, gp_tag)))

    # Noise magnitude
    sigma_dist = getattr(o, f"{source}_noise_sigma_dist", None) or o.noise_sigma_dist
    sigma_dist_params = _get_param(o, "noise_sigma_dist_params", source)
    sigma_tag = f"{dist_mapping.get(sigma_dist, sigma_dist.lower())}{'-'.join(map(str, sigma_dist_params))}"
    parts.append((4, sigma_tag))

    # Likelihood: autocorrel ?
    if getattr(o, f"{source}_noise_obs_autocorrel", None) or getattr(o, "noise_obs_autocorrel"):
        parts.append((5, "obsmv"))

    if getattr(o, f"{source}_noise_intercept", None) or getattr(o, "noise_intercept"):
        parts.append((9, "intercept"))

    return parts


def get_noise_tag(o, source):
    parts = get_noise_tag_parts(o, source)
    return "-".join([tag for _, tag in sorted(parts)])


def set_bool_tag(o, name, alias=None, odef=None):
    if alias is None:
        alias = name
    if odef is None:
        return [alias] if getattr(o, name) else []

    value = alias if getattr(o, name) else "no-"+alias
    value_def = alias if getattr(odef, name) else "no-"+alias
    if value != value_def:
        return [value]
    else:
        return []

def set_value_tag(o, func, odef=None):
    tags = func(o)
    if odef is None:
        return tags
    tags_def = func(odef)
    return tags if tags != tags_def else []

def _get_constraints_tag(o, odef=None):

    tags = []

    if o.skip_constraints:
        if o.global_slr_only:
            skip_constraints = [c for c in o.skip_constraints if c not in ['tidegauge', 'satellite', 'gps']]
        else:
            skip_constraints = o.skip_constraints
        tags += [f'skip-'+"-".join(skip_constraints)]

    if o.skip_past_constraints:
        tags += [f'skip-past-'+"-".join(o.skip_past_constraints)]

    if o.skip_slr20_constraints:
        tags += [f'skip-slr20-'+"-".join(o.skip_slr20_constraints)]

    if o.skip_rate2000_constraints:
        tags += [f'skip-rate2000-'+"-".join(o.skip_rate2000_constraints)]

    if o.skip_future_constraints:
        tags += [f'skip-future-'+"-".join(o.skip_future_constraints)]

    if o.greenland_constraints:
        tags += set_value_tag(o, lambda o: [f'gis-{"-".join(sorted(o.greenland_constraints))}'], odef)

    if o.antarctica_constraints:
        tags += set_value_tag(o, lambda o: [f'ais-{"-".join(sorted(o.antarctica_constraints))}'], odef)

    if o.steric_constraints:
        tags += set_value_tag(o, lambda o: [f'steric-{"-".join(sorted(o.steric_constraints))}'], odef)

    if o.glacier_constraints:
        tags += set_value_tag(o, lambda o: [f'glacier-{"-".join(sorted(o.glacier_constraints))}'], odef)

    return tags

def _get_ice_g_tag(o):
    tags = []
    if o.greenland_exclude_fred_peripheral_glaciers:
        tags += [f'gis-g-{o.greenland_exclude_fred_peripheral_glaciers_method}']
        if o.greenland_exclude_fred_peripheral_glaciers_method == "resampled":
            tags += [f'{"-".join(name[:3] for name in o.greenland_exclude_fred_peripheral_glaciers_datasets)}']

    if o.antarctica_exclude_fred_peripheral_glaciers:
        tags += [f'ais-g-{o.antarctica_exclude_fred_peripheral_glaciers_method}']
        if o.antarctica_exclude_fred_peripheral_glaciers_method == "resampled":
            tags += [f'{"-".join(name[:3] for name in o.antarctica_exclude_fred_peripheral_glaciers_datasets)}']

    return tags


def get_glacier_tags(o, odef=None):

    glacier_tags = []

    if not o.regional_glacier:
        glacier_tags.append(f'_global-glacier')
        return glacier_tags

    if odef is None or set(o.glacier_regions) != set(odef.glacier_regions):
        glacier_tags += list(map(str, sorted(o.glacier_regions)))

    if odef is None or o.glacier_exponent != odef.glacier_exponent:
        glacier_tags.append(f"n{o.glacier_exponent}")
    glacier_tags += set_bool_tag(o, 'glacier_dimensionless', 'adim', odef)

    if o.glacier_exclude_icesheets:
        glacier_tags += ['no-icesheets']

    def _get_uncharted_tag(o):
        tags = []
        if not o.add_uncharted_glaciers:
            tags += ['no-uncharted']
        elif o.glacier_uncharted_distribution:
            tags += [f"uncharted-{o.glacier_uncharted_distribution}"]
            if o.glacier_uncharted_exclude_antarctica:
                tags += [f"noaa"]
        return tags

    glacier_tags += set_value_tag(o, _get_uncharted_tag, odef)


    if "mm21" in "".join(o.glacier_constraints):
        glacier_tags += set_bool_tag(o, 'glacier_regress_on_mm21', 'regress', odef)
        glacier_tags += set_bool_tag(o, 'glacier_mm21_drop_20CRV3_17', 'drop20CRV3-17', odef)
        glacier_tags += set_bool_tag(o, 'glacier_mm21_drop_20CRV3', 'drop20CRV3', odef)
        glacier_tags += set_bool_tag(o, 'glacier_include_mm21_error', 'mm21-error', odef)

    if (odef is None) or set(o.glacier_future_constraint_experiments) != set(odef.glacier_future_constraint_experiments):
        glacier_tags.extend(sorted(o.glacier_future_constraint_experiments))

    glacier_tags += set_bool_tag(o, 'glacier_noise_on_forcing', 'noise-on-forcing', odef)

    if "proj2100" in "".join(o.glacier_future_constraint_experiments):
        glacier_tags += set_bool_tag(o, 'glacier_future_constraint_on_trend', 'proj2100-on-trend', odef)

    if odef is None or o.glacier_volume_source != odef.glacier_volume_source:
        glacier_tags.append(f"V{o.glacier_volume_source}")

    return glacier_tags


def get_experiment_name(o, defaults={}):
    if o.name:
        return o.name

    odef = argparse.Namespace(**defaults)

    basename = f'run'

    if o.number == 0:
        o.global_slr_only = True

    if o.global_slr_only:
        basename += '_global-slr-only'

    if o.preset:
        basename += "_" + '-'.join(o.preset)

    if o.sources != odef.sources:
        basename += f'_only-{"-".join(o.sources)}'

    if o.isimip_mode:
        assert o.isimip_model is not None
        basename += f'_ISIMIP-{o.isimip_model}'
        if o.isimip_all_scaling_patterns:
            basename += f'-all-scaling-patterns'
        if o.isimip_steric_sigma != 10:
            basename += f'-steric-sigma-{o.isimip_steric_sigma}'
        if o.isimip_tas_noise:
            basename += f'_no-smooth-tas'
        if o.isimip_tas_no_obs:
            basename += f'_no-obs-tas'

    constraints_tags = set_value_tag(o, lambda oo: _get_constraints_tag(oo, odef), odef)
    if constraints_tags:
        basename += f'_{"-".join(constraints_tags)}'

    ice_g_tag = set_value_tag(o, _get_ice_g_tag, odef)
    if ice_g_tag:
        basename += f'_{"-".join(ice_g_tag)}'

    glacier_tags = get_glacier_tags(o, odef)
    if glacier_tags:
        glacier_tag = "-".join(glacier_tags)
        basename += f'_glacier-{glacier_tag}'

    long_sources = ["greenland", "antarctica", "steric", "glacier"]

    def _get_prior_dist_tag(o, source, p):
        dist_name = getattr(o, f"{source}_prior_dist_{p}")
        return dist_mapping.get(dist_name, dist_name)

    def _get_prior_dist_params_tag(o, source, p):
        value = getattr(o, f"{source}_prior_dist_params_{p}")
        if value is None:
            return ""
        return "-".join(map(str, value))

    def _get_prior_tag(o, source, p):
        return [p] + [_get_prior_dist_tag(o, source, p) + _get_prior_dist_params_tag(o, source, p)]


    for source in long_sources:
        name_parts = []
        for p in ["q", "a", "aT0", "b", "V0"]:
            if hasattr(odef, f"prior_dist_{p}"):
                p_parts = set_value_tag(o, lambda o: [source] + _get_prior_tag(o, source, p), odef)
                if p_parts:
                    name_parts += p_parts[1:]

        if source == "glacier" and o.glacier_prior_inflate_V0 and o.glacier_prior_inflate_V0 != 1:
            name_parts += [f"V0x{o.glacier_prior_inflate_V0}"]

        noise_parts = set_bool_tag(o, f"add_{source}_noise", "noise", odef)
        if noise_parts:
            name_parts += noise_parts

        if getattr(o, f"add_{source}_noise", False):
            noise_parts = set_value_tag(o, lambda o: [source] + [get_noise_tag(o, source)], odef)
            if noise_parts:
                name_parts += noise_parts[1:]

        if getattr(o, f"{source}_future_constraint", None) == "ar6-low-confidence":
            name_parts += [f"{source}-2e"]

        if name_parts:
            basename += f'_{_source_short(source)}-{"-".join(name_parts)}'

        if source != "glacier": # already written elsewhere for glaciers
            if "proj2100" in "".join(getattr(o, f"{source}_constraints") or (["proj2100"] if "proj2100" not in o.skip_future_constraints+o.skip_constraints else [])):
                tag = set_bool_tag(o, f'{source}_future_constraint_on_trend', '-on-trend', odef)
                if tag:
                    basename += f'_{_source_short(source)}-{"-".join(tag)}'

    if o.static_antarctica:
        basename += '_static-antarctica-' + "-".join(sorted(o.antarctica_ar6_method))

    if o.static_greenland:
        basename += '_static-greenland'

    if o.static_steric:
        basename += '_static-steric'

    if o.static_glacier:
        basename += '_static-glacier'

    # nuts_sampler_suffix = "_"+o.nuts_sampler if o.nuts_sampler != "pymc" else ""
    # slversion = "_" + get_version()
    # pymc_version = f"_{pm.__version__}"
    # python_version = f"-py{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    # # suffix = nuts_sampler_suffix + slversion + pymc_version + python_version
    # suffix = nuts_sampler_suffix + pymc_version + python_version
    # suffix = nuts_sampler_suffix
    # suffix = ""

    if o.global_slr_only:
        o.name = basename + o.suffix
        return o.name

    # Args for regional SLR simulations
    # ---------------------------------

    if o.method != "trend":
        basename += f'-{o.method}'
    # if o.legacy_april_2023:
    #     basename += "-legacy"
    if not o.mixed_covariance:
        basename += f'_no-mixed-cov'

    else:
        if o.independent_constraints:
            basename += f'mixed-except-'+"-".join(o.independent_constraints)
    if o.model_mean_rate_instead_of_lintrend:
        basename += f'_model-mean-rate'
    if o.scale_tidegauge_oceandyn:
        basename += f'_tg-error-scaled-oceandyn+{o.tidegauge_measurement_error}'
    elif not o.estimate_tidegauge_measurement_error:
        basename += f'_tg-error-{o.tidegauge_measurement_error}'
    elif o.tidegauge_measurement_error != 0.1:
        assert o.estimate_tidegauge_measurement_error
        basename += f'_tg-error-above-{o.tidegauge_measurement_error}'
    else:
        assert o.estimate_tidegauge_measurement_error

    if o.satellite_measurement_error_method != "prandi2021":
        if o.satellite_measurement_error_method == "constant":
            basename += f'_satellite-error-{o.satellite_measurement_error}'
        else:
            basename += f'_satellite-error-{o.satellite_measurement_error_method}'
    if o.covariance_method:
        basename += f'_{o.covariance_method}'
    if o.split_tidegauges:
        basename += f'_split-1990'
    if o.mask_pre_1990:
        basename += f'_mask-pre-1990'
    if o.mask_post_1990:
        basename += f'_mask-post-1990'

    if o.tidegauge_mask and o.tidegauge_mask != ['psmsl']:
        basename += '_tidegauge-'+"-".join(o.tidegauge_mask)
    if o.satellite_mask and o.satellite_mask != ['psmsl']:
        basename += '_satellite-'+"-".join(o.satellite_mask)
    if o.gps_mask and o.gps_mask != ['psmsl']:
        basename += '_gps-'+"-".join(o.gps_mask)
    # if o.skip_tidegauge:
    #     basename += '_tidegauge-skip-'+"-".join([str(ID) for ID in o.skip_tidegauge])
    # if o.skip_tidegauge_file:
    #     basename += f'_tidegauge-skip-{o.skip_tidegauge_file}'

    if o.covariance_source and (o.covariance_source != 'cmip6' or not o.rescale_cmip6):
        basename += f'_{o.covariance_source}'
    if not o.rescale_cmip6:
        basename += '-no-rescale'
    if o.psmsl_label != 'psmsl_rlr_1900_2018_subset':
        basename += f'_{o.psmsl_label}'

    if o.min_years != 20:
        basename += f'-min{o.min_years}years'

    if o.flagged:
        basename += "-flagged"
    if not o.remove_meteo:
        basename += '-meteo'
    if o.wind_correction:
        basename += '-wind'
    if not o.frederikse_only:
        basename += '-all-psmsl'
    if o.classical_formula_for_tides:
        basename += '-classicaltides'
    if o.linear_icesheet:
        basename += "_linear-icesheet"
    if o.global_slr_method != "estimate_coupled":
        basename += '-' + o.global_slr_method
    if o.historical_mass_fingerprints:
        basename += f'_historical-mass-fingerprints-{"-".join(sorted(o.historical_mass_fingerprints))}-{o.historical_window}-years'
    if o.gia_eof:
        if o.gia_eof_num != 30:
            basename += f'_gia-{o.gia_eof_num}-eof'
    else:
        basename += f'_gia-mvnormal'

    vlmres_tags = []
    if o.vlm_res_mode != "constant":
        vlmres_tags.append(o.vlm_res_mode)
        if o.vlm_res_mode == "split":
            vlmres_tags.append(f'{o.vlm_res_split_year}')
        if o.vlm_res_mode in ("split", "decadal") and o.vlm_res_autocorrel:
            vlmres_tags.append(f'{o.vlm_res_autocorrel}')
    if not o.vlm_res_normal:
        vlmres_tags.append("cauchy")
        vlmres_tags.append(str(o.vlm_res_sd))
    elif o.vlm_res_sd != 2:
        vlmres_tags.append(str(o.vlm_res_sd))
    if o.vlm_res_domain != "psmsl":
        vlmres_tags.append(o.vlm_res_domain)
    if o.vlm_res_spatial_scale != 100:
        vlmres_tags.append(f'spatial-scale-{o.vlm_res_spatial_scale}km')
    if vlmres_tags:
        basename += f'_vlm-res-{"-".join(vlmres_tags)}'

    if o.gps_distance_tol != 0.1:
        basename += f'_gps-tol-{o.gps_distance_tol}km'
    if o.gps_formal_error != 'roughness-no-filtering':
        basename += f'_gps-{o.gps_formal_error}'
    if o.gps_dist != 'normal':
        basename += f'-'+o.gps_dist
    if o.gps_quality and set(o.gps_quality) != {'good', 'medium', 'bad'}:
        basename += f'_gps-{"-".join(sorted(o.gps_quality))}'
    if o.gps_scale_error != 1:
        basename += f'_gps-scale-{o.gps_scale_error}'

    if not o.oceandyn:
        basename += f'_no-oceandyn'
    elif o.mean_oceandyn:
        basename += f'_mean-oceandyn'
    elif not o.steric_coef_eof:
        basename += f'_steric-coef-mvnormal'
    if o.oceandyn_models:
        basename += f'-' + '-'.join(o.oceandyn_models)
    if o.steric_coef_scale:
        basename += f'_steric-coef-scale-{o.steric_coef_scale}km'

    # else:
        # basename += f'_steric-coef-eof'

    if o.psmsl_ids:
        basename += f'_psmsl-{"-".join(str(ID) for ID in o.psmsl_ids)}'
    if o.coords:
        basename += f'_coords-{"-".join(str(ID) for ID in coords_to_ids(o.coords))}'
    if o.grid:
        basename += f'_grid-{o.grid}'
        if o.grid_bbox:
            basename += '-{:.0f}-{:.0f}E-{:.0f}-{:.0f}N'.format(*o.grid_bbox)
    if o.include_gps_stations:
        basename += f'_gps-' + '-'.join(o.include_gps_stations)

    obstag = "-".join(sorted(o.leave_out_obs_type)).replace("tidegauge", "tg").replace("satellite", "sat")

    # --leave-out-tidegauge experiments: do some rudimentary naming so that data does not get overwritten (duplicates may appear that's OK)
    if o.leave_out_tidegauge or o.leave_out_tidegauge_ordinal or o.leave_out_tidegauge_fraction:
        obj = {k:v for k,v in vars(o).items() if k.startswith('leave_out_tidegauge')}
        if o.leave_out_tidegauge_fraction:
            import numpy as np
            obj['randomness'] = float(np.random.default_rng(o.random_seed).normal())
        basename += f"_leave-out-{obstag}-{random_hex(obj)}"

    if o.leave_out_tidegauge_basin_id:
        basename += f"_leave-out-{obstag}-basin-{'-'.join(*(str(id) for id in sorted(o.leave_out_tidegauge_basin_id)))}"

    if o.number:
        basename += f'_first-{o.number}'

    o.name = basename + o.suffix
    return o.name




def set_presets(o, presets=None, check_fields=False, all_option_names=None):
    # in the code we use full versions as v...
    # patches that constitute the version are prefixed by "-"
    # patches that add features on top of a version are prefixed by "+"

    if all_option_names is None:
        all_option_names = list(vars(o))

    for preset in (presets or o.preset):

        if preset == "v0.3.4":  # update to glacier model v2 -- same parameters otherwise besides a Truncated Gaussian
            # nothing to do
            pass

        elif preset == "v0.3.4-dev-split-mm21-zemp": # apply MM21 and Zemp separately
            set_presets(o, ["v0.3.4"])
            o.glacier_constraints = ["proj2100", "mm21", "zemp19"]
            o.glacier_mm21_forcing = ['CRU TS 4.03', 'ERA20C', '20CRV3', 'CFSR', 'JRA55', 'ERA5', 'MERRA2', 'ERA_Interim']
            o.glacier_include_mm21_error = True

        elif preset == "v0.3.4-dev-split-mm21-zemp+ar6": # apply MM21 and Zemp separately
            set_presets(o, ["v0.3.4-dev-split-mm21-zemp"])
            o.glacier_constraints = ["proj2100", "mm21", "zemp19", "ar6-present"]

        else:
            raise ValueError(f"Unknown preset: {preset}")

    # check we didn't mistype the options
    if check_fields:
        new_options_names = list(vars(o))
        if not set(new_options_names).issubset(set(all_option_names)):
            raise ValueError(f"Unknown options: {sorted(set(new_options_names) - set(all_option_names))}")


def get_runid(_parser=None, cirun=None, ref=None, dirname=None, **kwargs):

    CIDIR = get_runpath()

    if cirun:
        return cirun

    if dirname:
        # Also derive cirun, which is used with --figs and --json, and possibly elsewhere (futureproof)
        try:
            return str(Path(dirname).relative_to(CIDIR))
        except ValueError:
            logger.warning(f"{dirname} is not a subpath of CIDIR: {CIDIR}")
            raise ValueError("Failed to derive CIRUN from --dirname: please provide via --cirun")

    args = argparse.Namespace(**kwargs)

    if not ref:
        ref = get_version()
        logger.warning(f"--ref not provided. {ref} is assumed.")

    if _parser is not None:
        defaults={a.dest:_parser.get_default(a.dest) for a in _parser._actions}

    name = get_experiment_name(args, defaults=defaults if _parser else {})

    return os.path.join(ref, name)

preparser = argparse.ArgumentParser(add_help=False)
group = preparser.add_argument_group("Meta options")
group.add_argument('--param-file', help="preset options")
group.add_argument('--preset', help="preset options", nargs="*", default=[])

def get_parser(static=True):
    parser = argparse.ArgumentParser(parents=[log_parser, config_parser, preparser])
    DEFAULTS_KW = get_all_defaults(static)
    # print(DEFAULTS_KW)
    # import sys
    # sys.exit(1)

    group = parser.add_argument_group("Sampler and run")
    group.add_argument('--nuts-sampler', choices=["pymc", "nutpie", "blackjax", "numpyro"], default="numpyro")
    group.add_argument('--resume', action="store_true", help='if True, read existing config / model and sample from there')
    group.add_argument('--no-resume', action="store_false", dest='resume')
    group.add_argument('--ref', help="git reference (commit, etc) to know where to save the trace under ci/runs. IF not providede, it will be inquired via git.")
    group.add_argument('--name', help="Experiment name. By default determined from command-line arguments.")
    group.add_argument('--suffix', default="", help="Suffix to add to the experiment name (if name is not provided)")
    group.add_argument('--cirun', help="same as {ref}/{cirun} (will take precedence on either of {ref} and {cirun})")
    group.add_argument('--dirname', help="Useful in case run folders are outside the typical workflow {CIRUN}/{ref}/{cirun} ==> --cirun might be required")
    group.add_argument("--no-sampling", action='store_true')
    group.add_argument("--sampling", action='store_false', dest='no_sampling')
    group.add_argument('--draws', type=int, default=1000)
    group.add_argument('--tune', type=int, default=1000)
    group.add_argument('--chains', type=int, default=4)
    group.add_argument('--cpus', type=int, default=4)
    group.add_argument('--target_accept', type=float)

    group.add_argument('--experiments', nargs="+", choices=AVAILABLE_EXPERIMENTS)
    group.add_argument('--global-slr-method', choices=['estimate_coupled', 'two-step'], default="estimate_coupled")
    group.add_argument('--sources', choices=GLOBAL_SOURCES, default=GLOBAL_SOURCES, nargs="+")
    group.add_argument('--standalone-global-trace', help='in the two-step mode, use this instead of re-running')
    group.add_argument('--global-slr-only', action='store_true', help='only estimate global SLR (no local SLR calculations)')
    group.add_argument('--no-global-slr-only', action='store_false', dest='global_slr_only')
    group.add_argument('--add-global-slr-only', action='store_true', help='also estimate a standalone trace')
    group.add_argument('--no-add-global-slr-only', action='store_false', dest='add_global_slr_only')
    group.add_argument('--save-timeseries', action="store_true", help="will save output time-series with 10-year time-step to trace. Otherwise skip, and left for sample_posterior_predictive as a postprocessing task")
    group.add_argument('--no-save-timeseries', action="store_false", dest='save_timeseries')
    group.add_argument('--random-seed', type=int)

    # group.add_argument('--steric-coef-eof', action='store_true')
    group.add_argument('--json', action='store_true', help="also crunch json files")
    group.add_argument('--no-json', action='store_false', dest='json')
    group.add_argument('--figs', action='store_true', help="also run the figures notebook (experimental)")
    group.add_argument('--no-figs', action='store_false', dest='figs')

    group = parser.add_argument_group("Global contributions")
    group.add_argument('--linear-icesheet', action="store_true")
    group.add_argument('--no-linear-icesheet', action="store_false", dest='linear_icesheet')
    # parser.add_argument('--regional-glacier', action="store_true")
    group.add_argument('--global-glacier', action="store_false", dest='regional_glacier', help='so that regional_glacier is True by default')
    group.add_argument('--glacier-model-version', default="v2")
    group.add_argument('--regional-glacier', action="store_true")
    group.add_argument('--no-uncharted-glaciers', action="store_false", dest="add_uncharted_glaciers")
    group.add_argument('--uncharted-glaciers', action="store_true", dest="add_uncharted_glaciers")
    # group.add_argument('--add-uncharted-glaciers', action="store_true")
    # group.add_argument('--add-glacier-residuals', action="store_true")
    group.add_argument('--no-add-glacier-noise', action="store_false", dest='add_glacier_noise')
    group.add_argument('--add-glacier-noise', action="store_true")
    group.add_argument('--no-add-antarctica-noise', action="store_false", dest='add_antarctica_noise')
    group.add_argument('--add-antarctica-noise', action="store_true")
    group.add_argument('--no-add-greenland-noise', action="store_false", dest='add_greenland_noise')
    group.add_argument('--add-greenland-noise', action="store_true")
    group.add_argument('--no-add-steric-noise', action="store_false", dest='add_steric_noise')
    group.add_argument('--add-steric-noise', action="store_true")
    group.add_argument('--glacier-noise-crosscorrel', action="store_true", help="glacier noise correlated across regions")
    group.add_argument('--no-glacier-noise-crosscorrel', action="store_false", dest='glacier_noise_crosscorrel')
    group.add_argument('--no-noise-obs-autocorrel', action="store_false", dest="noise_obs_autocorrel",
                       help="use obs auto-covariance structure to constrain the noise")
    group.add_argument('--noise-obs-autocorrel', action="store_true")
    group.add_argument('--glacier-noise-apply-constraints', action="store_true",
                       help="""apply the constraints in _generate_noise (that is not compatible with --glacier-constraint).
                       By default in the new v2 version the annual glacier constraints are applied via --glacier-constraint. """)
    group.add_argument('--no-glacier-noise-apply-constraints', action="store_false", dest='glacier_noise_crosscorrel')
    group.add_argument('--glacier-noise-on-forcing', action="store_true")
    group.add_argument('--no-glacier-noise-clip-volume', action="store_false", dest='glacier_noise_clip_volume',)
    group.add_argument('--glacier-noise-clip-volume', action="store_true") # true by default

    group.add_argument('--no-steric-noise-obs-autocorrel', action="store_false", dest="steric_noise_obs_autocorrel")
    group.add_argument('--steric-noise-obs-autocorrel', action="store_true")
    group.add_argument('--no-greenland-noise-obs-autocorrel', action="store_false", dest="greenland_noise_obs_autocorrel")
    group.add_argument('--greenland-noise-obs-autocorrel', action="store_true")
    group.add_argument('--no-glacier-noise-obs-autocorrel', action="store_false", dest="glacier_noise_obs_autocorrel")
    group.add_argument('--glacier-noise-obs-autocorrel', action="store_true")
    group.add_argument('--no-antarctica-noise-obs-autocorrel', action="store_false", dest="antarctica_noise_obs_autocorrel")
    group.add_argument('--antarctica-noise-obs-autocorrel', action="store_true")

    group.add_argument('--noise-gp-cov', default=DEFAULTS_KW["noise"]["gp_cov"],
                       help=argparse.SUPPRESS)
                    #    help="GP cov kernel")
    group.add_argument('--noise-gp-cov-params', type=float, nargs='*',
                       default=DEFAULTS_KW["noise"]["gp_cov_params"],
                       help=argparse.SUPPRESS)
                    #    help="Parameters for the GP cov kernel. If provided, this override noise-ls parameter. Note the parameters are then passed directly without hyperprior.")
    group.add_argument('--noise-ls-dist', default=DEFAULTS_KW["noise"]["ls_dist"],
                       help=argparse.SUPPRESS)
                    #    help="""Distribution for the length scale of the GP cov kernel. (default: %(default)s).""")
    group.add_argument('--noise-ls-dist-params', type=float, nargs="*", default=DEFAULTS_KW["noise"]["ls_dist_params"],
                       help=argparse.SUPPRESS)
                    #    help="""Length scale of the GP (default: %(default)s).""")
    group.add_argument('--noise-sigma-dist', default=DEFAULTS_KW["noise"]["sigma_dist"], help=argparse.SUPPRESS)
    group.add_argument('--noise-sigma-dist-params', type=float, nargs="*", default=DEFAULTS_KW["noise"].get("sigma_dist_params"), help=argparse.SUPPRESS)

    group.add_argument('--steric-noise-gp-cov', default="WhiteNoise")
    group.add_argument('--steric-noise-gp-cov-params', type=float, nargs='*', default=[1.0])
    group.add_argument('--steric-noise-ls-dist')
    group.add_argument('--steric-noise-ls-dist-params', type=float, nargs="*", default=[1.])
    group.add_argument('--steric-noise-sigma-dist', default="Exponential")
    group.add_argument('--steric-noise-sigma-dist-params', nargs="*", type=float, default=[1.0])

    group.add_argument('--glacier-noise-gp-cov', default="Matern12")
    group.add_argument('--glacier-noise-gp-cov-params', type=float, nargs='*')
    group.add_argument('--glacier-noise-ls-dist', default="ConstantData")
    group.add_argument('--glacier-noise-ls-dist-auto', action="store_true", help="use the glacier length scale from the data (default: %(default)s)")
    group.add_argument('--glacier-noise-ls-dist-params', type=float, nargs="*", default=[5.])
    group.add_argument('--glacier-noise-sigma-dist', default="ConstantData")
    group.add_argument('--glacier-noise-sigma-dist-params', nargs="*", type=float, default=[1.0])

    group.add_argument('--greenland-noise-gp-cov', default="Matern12")
    group.add_argument('--greenland-noise-gp-cov-params', type=float, nargs='*')
    group.add_argument('--greenland-noise-ls-dist', default="ConstantData")
    group.add_argument('--greenland-noise-ls-dist-params', type=float, nargs="*", default=[5.])
    group.add_argument('--greenland-noise-sigma-dist', default="Exponential")
    group.add_argument('--greenland-noise-sigma-dist-params', nargs="*", type=float, default=[1.0])

    group.add_argument('--antarctica-noise-gp-cov', default="WhiteNoise")
    group.add_argument('--antarctica-noise-gp-cov-params', type=float, nargs='*', default=[1.0])
    group.add_argument('--antarctica-noise-ls-dist')
    group.add_argument('--antarctica-noise-ls-dist-params', type=float, nargs="*", default=[1.0])
    group.add_argument('--antarctica-noise-sigma-dist', default="ConstantData")
    group.add_argument('--antarctica-noise-sigma-dist-params', nargs="*", type=float, default=[0.41])

    group.add_argument('--steric-noise-on-cumul', action="store_false", dest="steric_noise_on_rate", help="noise on cumulative SLR")
    group.add_argument('--steric-noise-on-rate', action="store_true")
    group.add_argument('--greenland-noise-on-cumul', action="store_false", dest="greenland_noise_on_rate", help="noise on cumulative SLR")
    group.add_argument('--greenland-noise-on-rate', action="store_true")
    group.add_argument('--glacier-noise-on-cumul', action="store_false", dest="glacier_noise_on_rate", help="noise on cumulative SLR")
    group.add_argument('--glacier-noise-on-rate', action="store_true")
    group.add_argument('--antarctica-noise-on-cumul', action="store_false", dest="antarctica_noise_on_rate", help="noise on cumulative SLR")
    group.add_argument('--antarctica-noise-on-rate', action="store_true")
    group.add_argument('--noise-on-rate', action="store_true", help=argparse.SUPPRESS) # DEPRECATED -> by default globally set noise_on_rate if False (otherwise all sources become True)
    group.add_argument('--noise-on-cumul', action="store_false", dest='noise_on_rate')
    # group.add_argument('--steric-noise-on-rate', action="store_true", help=argparse.SUPPRESS)
    # group.add_argument('--greenland-noise-on-rate', action="store_true", help=argparse.SUPPRESS)
    # group.add_argument('--glacier-noise-on-rate', action="store_true", help=argparse.SUPPRESS)
    # group.add_argument('--antarctica-noise-on-rate', action="store_true", help=argparse.SUPPRESS)
    group.add_argument('--noise-intercept', action="store_true", help=argparse.SUPPRESS)
    group.add_argument('--no-noise-intercept', action="store_false", dest='noise_intercept')

    group.add_argument('--antarctica-prior-dist-q', default=DEFAULTS_KW["antarctica"].get("prior_dist_q"))
    group.add_argument('--antarctica-prior-dist-params-q', default=DEFAULTS_KW["antarctica"].get("prior_dist_params_q"), nargs="*", type=float)
    group.add_argument('--antarctica-prior-dist-a', default=DEFAULTS_KW["antarctica"].get("prior_dist_a"))
    group.add_argument('--antarctica-prior-dist-params-a', default=DEFAULTS_KW["antarctica"].get("prior_dist_params_a"), nargs="*", type=float)
    group.add_argument('--antarctica-prior-dist-aT0', default=DEFAULTS_KW["antarctica"].get("prior_dist_aT0"))
    group.add_argument('--antarctica-prior-dist-params-aT0', default=DEFAULTS_KW["antarctica"].get("prior_dist_params_aT0"), nargs="*", type=float)

    group.add_argument('--greenland-prior-dist-q', default=DEFAULTS_KW["greenland"].get("prior_dist_q"))
    group.add_argument('--greenland-prior-dist-params-q', default=DEFAULTS_KW["greenland"].get("prior_dist_params_q"), nargs="*", type=float)
    group.add_argument('--greenland-prior-dist-a', default=DEFAULTS_KW["greenland"].get("prior_dist_a"))
    group.add_argument('--greenland-prior-dist-params-a', default=DEFAULTS_KW["greenland"].get("prior_dist_params_a"), nargs="*", type=float)
    group.add_argument('--greenland-prior-dist-aT0', default=DEFAULTS_KW["greenland"].get("prior_dist_aT0"))
    group.add_argument('--greenland-prior-dist-params-aT0', default=DEFAULTS_KW["greenland"].get("prior_dist_params_aT0"), nargs="*", type=float)

    group.add_argument('--steric-prior-dist-b', default=DEFAULTS_KW["steric"].get("prior_dist_b"))
    group.add_argument('--steric-prior-dist-params-b', default=DEFAULTS_KW["steric"].get("prior_dist_params_b"), nargs="*", type=float)
    group.add_argument('--steric-prior-dist-a', default=DEFAULTS_KW["steric"].get("prior_dist_a"))
    group.add_argument('--steric-prior-dist-params-a', default=DEFAULTS_KW["steric"].get("prior_dist_params_a"), nargs="*", type=float)
    group.add_argument('--steric-prior-dist-aT0', default=DEFAULTS_KW["steric"].get("prior_dist_aT0"))
    group.add_argument('--steric-prior-dist-params-aT0', default=DEFAULTS_KW["steric"].get("prior_dist_params_aT0"), nargs="*", type=float)

    group.add_argument('--glacier-prior-dist-a', default=DEFAULTS_KW["glacier"].get("prior_dist_a"))
    group.add_argument('--glacier-prior-dist-params-a', default=DEFAULTS_KW["glacier"].get("prior_dist_params_a"), nargs="*", type=float)
    group.add_argument('--glacier-prior-dist-aT0', default=DEFAULTS_KW["glacier"].get("prior_dist_aT0"))
    group.add_argument('--glacier-prior-dist-params-aT0', default=DEFAULTS_KW["glacier"].get("prior_dist_params_aT0"), nargs="*", type=float)
    group.add_argument('--glacier-prior-dist-V0', default=DEFAULTS_KW["glacier"].get("prior_dist_V0"))
    group.add_argument('--glacier-prior-dist-params-V0', default=DEFAULTS_KW["glacier"].get("prior_dist_params_V0"), nargs="*", type=float)
    group.add_argument('--glacier-prior-inflate-V0', default=1, type=float)
    group.add_argument('--glacier-exponent', default=DEFAULTS_KW["glacier"]["n"], type=float)
    group.add_argument('--no-glacier-dimensionless', action="store_false", dest='glacier_dimensionless')
    group.add_argument('--glacier-dimensionless', action="store_true", help="if True, glacier parameters are dimensionless")

    group.add_argument('--antarctica-ar6-method', choices=["ismipemu", "larmip", "bamber"], default=["ismipemu"], nargs="+")

    group.add_argument('--antarctica-future-constraint', choices=["ar6-medium-confidence", "ar6-low-confidence"], default="ar6-medium-confidence",)
    group.add_argument('--greenland-future-constraint', choices=["ar6-medium-confidence", "ar6-low-confidence"], default="ar6-medium-confidence",)

    group.add_argument('--static-antarctica', action="store_true", help="If true, Antarctica is taken from Fred and IPCC like landwater")
    group.add_argument('--no-static-antarctica', action="store_false", dest='static_antarctica')
    group.add_argument('--static-greenland', action="store_true", help="If true, greenland is taken from Fred and IPCC like landwater")
    group.add_argument('--no-static-greenland', action="store_false", dest='static_greenland')
    group.add_argument('--static-steric', action="store_true", help="If true, steric is taken from Fred and IPCC like landwater")
    group.add_argument('--no-static-steric', action="store_false", dest='static_steric')
    group.add_argument('--static-glacier', action="store_true", help="If true, glacier is taken from Fred and IPCC like landwater")
    group.add_argument('--no-static-glacier', action="store_false", dest='static_glacier')

    group.add_argument('--no-greenland-exclude-fred-peripheral-glaciers', action="store_false",
                       dest="greenland_exclude_fred_peripheral_glaciers")
    group.add_argument('--greenland-exclude-fred-peripheral-glaciers', action="store_true")
    group.add_argument('--greenland-exclude-fred-peripheral-glaciers-method', default="offset", choices=["resampled", "offset"],
                       help="resampled: resampled from source data, offset: offset the mean glacier contribution")
    group.add_argument('--greenland-exclude-fred-peripheral-glaciers-datasets',
                       choices=["kjeldsen2015", "mouginot2019", "bamber2018"],
                       default=["kjeldsen2015", "mouginot2019", "bamber2018"],
                       nargs="+")
    group.add_argument('--no-antarctica-exclude-fred-peripheral-glaciers', action="store_false",
                       dest="antarctica_exclude_fred_peripheral_glaciers")
    group.add_argument('--antarctica-exclude-fred-peripheral-glaciers', action="store_true")
    # ogroup.add_argument('--antarctica-exclude-fred-peripheral-glaciers', action="store_true", help="If true, resampled fred source data")
    group.add_argument('--antarctica-exclude-fred-peripheral-glaciers-method', default="offset", choices=["offset"],
                       help="offset: offset the mean glacier contribution")
    # group.add_argument('--icesheets-exclude-fred-peripheral-glaciers', action="store_true", help="alias for --antarctica-exclude-fred-peripheral-glaciers and --greenland-exclude-fred-peripheral-glaciers")

    # group.add_argument('-no-glacier-exclude-icesheets', dest="glacier_exclude_icesheets", action="store_false")
    group.add_argument('--glacier-exclude-icesheets', action="store_true",
                       help="If true, exclude glaciers at the periphery of the ice sheets")
    group.add_argument('--glacier-regions', nargs="*", default=list(range(1, 19+1)), type=int, choices=list(range(1, 19+1)))
    group.add_argument('--glacier-normalize-future', action="store_true", help="If true, normalize glacier future to 2018")
    group.add_argument('--no-glacier-normalize-future', action="store_false", dest='glacier_normalize_future')
    group.add_argument('--no-glacier-exclude-icesheets', action="store_false", dest='glacier_exclude_icesheets')
    group.add_argument('--glacier-uncharted-distribution', choices=["mm21-v1900", "mm21-slr20", "rgi", "rgi-small-2", "rgi-small-2-10"],
                       default="rgi", help="distribution of uncharted glaciers: e.g. https://gitlab.pik-potsdam.de/dcm-impacts/slr-tidegauges-future/-/issues/76#note_53420")

    group.add_argument('--no-glacier-uncharted-exclude-antarctica',
                       action="store_false", dest="glacier_uncharted_exclude_antarctica", help="If true, distribute uncharted glaciers away from AA")
    group.add_argument('--glacier-uncharted-exclude-antarctica', action="store_true")
    # group.add_argument('--glacier-uncharted-exclude-antarctica',
    #                    action="store_true", help="If true, distribute uncharted glaciers away from AA")

    # mechanism to pass additional AR6 constraints like 1970-2018
    group.add_argument('--greenland-constraints', nargs='+')
    group.add_argument('--antarctica-constraints', nargs='+')
    group.add_argument('--glacier-constraints', nargs='*', choices=["ar6-present", "proj2100", "mm21+zemp19", "mm21", "zemp19", "mm21-indiv", "mm21-2000"], default=["proj2100", "mm21+zemp19"])
    # ogroup.add_argument("--glacier-include-zemp", action="store_true")
    group.add_argument('--steric-constraints', nargs='+')

    group.add_argument('--steric-future-constraint-experiments', nargs='*', default=["ssp126", "ssp585"], choices=["ssp126", "ssp585"],)
    group.add_argument('--greenland-future-constraint-experiments', nargs='*', default=["ssp126", "ssp585"], choices=["ssp126", "ssp585"],)
    group.add_argument('--antarctica-future-constraint-experiments', nargs='*', default=["ssp126", "ssp585"], choices=["ssp126", "ssp585"],)
    # group.add_argument("--glacier-obs-resampled", action="store_true")
    # ogroup = group.add_mutually_exclusive_group()
    group.add_argument("--no-glacier-include-zemp", dest="glacier_include_zemp", action="store_false")
    group.add_argument("--glacier-include-zemp", action="store_true")
    group.add_argument("--glacier-include-mm21-error", action="store_true")
    group.add_argument("--no-glacier-include-mm21-error", dest="glacier_include_mm21_error", action="store_false")
    group.add_argument("--glacier-regress-on-mm21", action="store_true")
    group.add_argument("--no-glacier-regress-on-mm21", action="store_false", dest='glacier_regress_on_mm21')
    # group.add_argument("--no-glacier-regress-on-mm21", dest="glacier_regress_on_mm21", action="store_false")
    group.add_argument("--glacier-volume-source", choices=["ar6", "mm21"], default="ar6")
    group.add_argument("--glacier-future-constraint-experiments", default=["ssp585"], choices=["ssp585", "ssp1256"], nargs="*")
    group.add_argument("--glacier-mm21-forcing", default=MM21_FORCING, choices=MM21_FORCING, nargs="+")
    group.add_argument("--glacier-mm21-indiv-weighting", default='scale_region', choices=["none", "scale_region", "add_region"])
    group.add_argument("--glacier-mm21-drop-20CRV3-17", action="store_true", help="drop 20CRV3 from the mm21 forcing in region 17 (Southern Andes)") # see MM21
    group.add_argument("--glacier-mm21-drop-20CRV3", action="store_true", help="drop 20CRV3 from the mm21 forcing") # see MM21

    # apply future constraint on full trend + noise instead of just the trend
    group.add_argument("--glacier-future-constraint-on-trend", action="store_true")
    group.add_argument("--no-glacier-future-constraint-on-trend", dest="glacier_future_constraint_on_trend", action="store_false")
    group.add_argument("--debug-glacier-future-constraint-no-clip", action="store_true")
    group.add_argument("--debug-glacier-future-constraint-scaled", action="store_true")
    # group.add_argument("--debug-glacier-future-constraint-not-scaled", action="store_false", dest="debug_glacier_future_constraint_scaled")

    group.add_argument("--no-greenland-future-constraint-on-trend", dest="greenland_future_constraint_on_trend", action="store_false")
    group.add_argument("--greenland-future-constraint-on-trend", action="store_true")

    group.add_argument("--no-antarctica-future-constraint-on-trend", dest="antarctica_future_constraint_on_trend", action="store_false")
    group.add_argument("--antarctica-future-constraint-on-trend", action="store_true")

    group.add_argument("--no-steric-future-constraint-on-trend", dest="steric_future_constraint_on_trend", action="store_false")
    group.add_argument("--steric-future-constraint-on-trend", action="store_true")

    group = parser.add_argument_group("Global or generic constraints")
    group.add_argument('--add-constraints', nargs='+', default=[])
    group.add_argument('--skip-constraints', nargs='+', default=[],
        choices=["tidegauge", "satellite", "gps", "steric", "glacier", "ais", "gis"],
        help='name of constraints to skip (diagnostic only)')
    group.add_argument('--skip-past-constraints', nargs='+', default=[],
        choices=["steric", "glacier", "ais", "gis"],
        help='name past of constraints to skip (diagnostic only)')
    group.add_argument('--skip-slr20-constraints', nargs='+', default=[],
        choices=["steric", "glacier", "ais", "gis"],
        help='name past of constraints to skip (diagnostic only)')
    group.add_argument('--skip-rate2000-constraints', nargs='+', default=[],
        choices=["steric", "glacier", "ais", "gis"],
        help='name past of constraints to skip (diagnostic only)')
    group.add_argument('--skip-future-constraints', nargs='+', default=[],
        choices=["steric", "glacier", "ais", "gis"],
        help='name future of constraints to skip (diagnostic only)')
    group.add_argument('--skip-obs-20c', action="store_false", dest="obs_20c")
    group.add_argument('--obs-20c', action="store_true")
    group.add_argument('--skip-obs-21c', action="store_false", dest="obs_21c", help="this skip entering in the routine ==> will avoid issues with experiment choices and future AR6 constraints")
    group.add_argument('--obs-21c', action="store_true")
    group.add_argument('--skip-all-constraints', action="store_true", help="this will skip constraints, definition, used to speed-up model def for resampling")
    group.add_argument('--no-skip-all-constraints', action="store_false", dest='skip_all_constraints')

    group = parser.add_argument_group("Tidegauges dataset")
    group.add_argument('--psmsl-label', default="psmsl_rlr_1900_2018_subset", help=argparse.SUPPRESS)
    group.add_argument('--psmsl-ids', type=int, nargs="+", help=argparse.SUPPRESS)
    group.add_argument('--station-ids', type=int, nargs="+", help=argparse.SUPPRESS)
    group.add_argument('-n', '--number', type=int, help=argparse.SUPPRESS) #, help='limit the number of tide gauges')
    # parser.add_argument('--skip-tidegauge', type=int, nargs="+")
    group.add_argument('--min-years', default=20, type=int)
    group.add_argument('--flagged', action="store_true", help=argparse.SUPPRESS) #, help='include flagged stations')
    group.add_argument('--no-flagged', action="store_false", dest='flagged', help=argparse.SUPPRESS)
    group.add_argument('--no-remove-meteo', action='store_false', dest='remove_meteo', help=argparse.SUPPRESS)
    group.add_argument('--remove-meteo', action='store_true', help=argparse.SUPPRESS)
    group.add_argument('--wind-correction', action='store_true')
    group.add_argument('--no-wind-correction', action='store_false', dest='wind_correction', help=argparse.SUPPRESS)
    group.add_argument('--include-all-psmsl', action='store_false', dest='frederikse_only', help=argparse.SUPPRESS)
    group.add_argument('--frederikse-only', action='store_true', help=argparse.SUPPRESS)
    group.add_argument('-c','--coords', type=float, nargs=2, action='append', default=[], help=argparse.SUPPRESS) # stores a list of coordinates
    group.add_argument('--grid', help=argparse.SUPPRESS) # define the grid
    group.add_argument('--grid-bbox', type=float, nargs=4, help=argparse.SUPPRESS) # define the grid
    group.add_argument('--include-gps-stations', choices=["hammond2021-neighboring", "hammond2021-all"], nargs="*", help=argparse.SUPPRESS) #, non-PSMSL GPS stations") # help="include
    group.add_argument('--classical-formula-for-tides', action='store_true', help=argparse.SUPPRESS)
    group.add_argument('--no-classical-formula-for-tides', action='store_false', dest='classical_formula_for_tides', help=argparse.SUPPRESS)

    group = parser.add_argument_group("Local obs mask")
    group.add_argument('--leave-out-tidegauge-fraction', type=float, help='if provided leave out a random fraction of tide-gauges')
    group.add_argument('--leave-out-tidegauge', nargs='+', type=int, help='specify TG ideas to leave out (by PSMSL IDs)')
    group.add_argument('--leave-out-tidegauge-ordinal', nargs='+', type=int, help='specify TG ideas to leave out by ordinal index (0, 1, 2, ..., N)')
    group.add_argument('--leave-out-tidegauge-basin', nargs='+', help='specify ocean basins from which TG constraints are left out', choices=list(BASINS_MAP.values()))
    group.add_argument('--leave-out-tidegauge-basin-id', nargs='+', type=int, help='specify ocean basin IDs from which TG constraints are left out', choices=list(BASINS_MAP))
    group.add_argument('--leave-out-obs-type', nargs="*", choices=["tidegauge", "satellite", "gps"],
                       default=["tidegauge", "satellite", "gps"],
                       help='indicate which obs type to leave out (by default tidegauge only)')

    group = parser.add_argument_group("Tide gauge / sat constraints")
    group.add_argument('--model-mean-rate', action="store_true", dest="model_mean_rate_instead_of_lintrend", help=argparse.SUPPRESS)
    group.add_argument('--split-tidegauges', action="store_true", help=argparse.SUPPRESS)
    group.add_argument('--no-split-tidegauges', action="store_false", dest='split_tidegauges', help=argparse.SUPPRESS)
    group.add_argument('--mask-pre-1990', action="store_true", help=argparse.SUPPRESS)
    group.add_argument('--no-mask-pre-1990', action="store_false", dest='mask_pre_1990', help=argparse.SUPPRESS)
    group.add_argument('--mask-post-1990', action="store_true", help=argparse.SUPPRESS)
    group.add_argument('--no-mask-post-1990', action="store_false", dest='mask_post_1990', help=argparse.SUPPRESS)
    group.add_argument('--covariance-source', choices=["cmip6", "satellite", "satellite_eof_ar1"], default="cmip6", help=argparse.SUPPRESS)
    group.add_argument('--no-rescale-cmip6', action="store_false", dest="rescale_cmip6", help=argparse.SUPPRESS)
    group.add_argument('--rescale-cmip6', action="store_true", help=argparse.SUPPRESS)

    group.add_argument('--cmip6-interp-method', help=argparse.SUPPRESS) # FOR DEV ONLY
    group.add_argument('--method', default="trend", help=argparse.SUPPRESS) # FOR DEV ONLY

    group.add_argument('--covariance-method')
    group.add_argument('--no-mixed-covariance', action="store_false", dest='mixed_covariance')
    group.add_argument('--mixed-covariance', action="store_true")
    group.add_argument('--independent-constraints', nargs='+', default=[], help='name of constraints to consider independent')

    # group.add_argument('--constraints-file', help='csv file with full constraints definitions: lon, lat, ID, tidegauge, satellite and gps columns')
    group.add_argument('--tidegauge-mask', default=['psmsl'], nargs='*')
    group.add_argument('--satellite-mask', default=['psmsl'], nargs='*')
    group.add_argument('--gps-mask', default=['psmsl'], nargs='*')

    group.add_argument('--tidegauge-measurement-error', type=float, default=0.1, help='added on top of ocean dyn, to the diagonal')
    group.add_argument('--no-estimate-tidegauge-measurement-error', action="store_false",
                        dest="estimate_tidegauge_measurement_error", help='estimate tidegauge measurement error from data (fix a in #46)')
    group.add_argument('--estimate-tidegauge-measurement-error', action="store_true")
    group.add_argument('--scale-tidegauge-oceandyn', action="store_true", help='if True, scale oceandyn samples for T.G. with T.G. to SAT variance (fix b in #46)')
    group.add_argument('--no-scale-tidegauge-oceandyn', action="store_false", dest='scale_tidegauge_oceandyn')
    group.add_argument('--satellite-measurement-error', type=float, default=0.1, help='added on top of ocean dyn, to the diagonal')
    group.add_argument('--satellite-measurement-error-method', default='prandi2021', choices=['constant', 'prandi2021'])
    # group.add_argument('--gia-eof', action='store_true')

    group.add_argument('--gps-distance-tol', default=.1, type=float, help="max distance for GPS stations to be used directly")
    group.add_argument('--gps-formal-error', default="roughness-no-filtering")
    group.add_argument('--gps-quality', nargs="*", choices=["good", "medium", "bad"], help="quality of GPS stations (all by default)")
    group.add_argument('--gps-dist', default="normal")
    group.add_argument('--gps-scale-error', default=1, type=float)

    group = parser.add_argument_group("Local model / prior")
    group.add_argument('--gia-mvnormal', dest='gia_eof', action='store_false', help='gia_eof True by default')
    group.add_argument('--gia-eof', action="store_true")
    group.add_argument('--gia-eof-num', type=int, default=30)
    group.add_argument('--vlm-res-mode', choices=["constant", "split", "decadal"], default="constant")
    group.add_argument('--vlm-res-split-year', default=2000, type=int)
    group.add_argument('--vlm-res-autocorrel', type=float)
    group.add_argument('--vlm-res-decadal-autocorrel', type=float, help=argparse.SUPPRESS) # DEPRECATED
    group.add_argument('--vlm-res-sd', type=float, default=2, help="prior parameter for vlm-res rate (1 mm/yr per default)")
    group.add_argument('--vlm-res-cauchy', action="store_false", dest="vlm_res_normal", help='Cauchy instead of normally distributed VLM res prior')
    group.add_argument('--vlm-res-normal', action="store_true")
    group.add_argument('--vlm-res-domain', default="psmsl")
    group.add_argument('--vlm-res-spatial-scale', type=float, default=100)
    group.add_argument('--no-oceandyn', action="store_false", dest="oceandyn")
    group.add_argument('--oceandyn', action="store_true")
    group.add_argument('--mean-oceandyn', action="store_true", help="If true, use multi-model mean for oceandyn")
    group.add_argument('--oceandyn-models', nargs='+', help="Specify models for the fingperints (zos patterns). Not this does not affect the COV matrix.")
    # hidden arguments only used to generate model for posterior predictive sampling
    group.add_argument('--no-steric-coef-cov', action='store_false', dest="steric_coef_cov", help=argparse.SUPPRESS)
    group.add_argument('--steric-coef-cov', action="store_true")
    group.add_argument('--steric-coef-mvnormal', action='store_false',
                       dest='steric_coef_eof', help='make steric coef EOF true by default')
    group.add_argument('--steric-coef-scale', type=float,
                       help='introduce spatial decay in the steric coef covariance structure (in km)')

    group.add_argument('--historical-mass-fingerprints', nargs="*", choices=["landwater", "ais", "gis","glacier"], help="if specified, time-dependent historical fingerprint is used instead of fixed one")
    group.add_argument('--historical-window', type=int, default=21, help="time-window of historical mass fingerprint, must be odd-integer (default: %(default)s)")

    group = parser.add_argument_group("ISIMIP")
    group.add_argument('--isimip-mode', action='store_true', help=argparse.SUPPRESS) #, help='in this mode perform the tuning according to one ISIMIP model')
    group.add_argument('--isimip-model', help=argparse.SUPPRESS) #)
    group.add_argument('--isimip-all-scaling-patterns', action='store_true', help=argparse.SUPPRESS) #)
    group.add_argument('--isimip-steric-sigma', default=10, type=float, help=argparse.SUPPRESS) #, help='steric SLR standard deviation for 2100 target (in mm)')
    group.add_argument('--isimip-tas-noise', action="store_true", help=argparse.SUPPRESS) #, help='dont smooth ISIMIP tas')
    group.add_argument('--isimip-tas-no-obs', action="store_true", help=argparse.SUPPRESS) #, help='dont use historical obs for tas forcing')

    return parser


def parse_args(cmd=None, parser=None, options={}):
    if parser is None:
        parser = get_parser()

    # first check for presets
    import argparse
    preset_parser = argparse.ArgumentParser(parents=[preparser], add_help=False)
    opres, _ = preset_parser.parse_known_args(cmd)

    opres.preset = options.pop("preset", opres.preset) # options has priority (manual debugging)
    opres.param_file = options.pop("param_file", opres.param_file) # options has priority (manual debugging)

    if opres.preset and opres.param_file:
        preset_parser.error("The --preset option is incompatible with --param-file. Please use one or the other.")
        preset_parser.exit(1)

    if opres.preset: # modify the defaults with the preset args
        set_presets(opres, opres.preset,
                    check_fields=True,
                    all_option_names=set(a.dest for a in parser._actions))
        parser.set_defaults(**vars(opres))

    elif opres.param_file:
        params = json.load(open(opres.param_file, 'r'))
        parser.set_defaults(**params)

    # now do the full parsing
    o = parser.parse_args(cmd)

    # options provided as a dictionary
    if options:
        # check that all options are valid
        for k in options.keys():
            if not hasattr(o, k):
                parser.print_help()
                raise ValueError(f"Unknown option {k} provided.")
        # update the options
        vars(o).update(options) # user-provided

    return o

def main(cmd=None, **options):
    preparser = argparse.ArgumentParser(description="Get run ID for the SLR model.")
    preparser.add_argument('--no-static', action='store_false', dest="static", help="actually import the modules instead of the static code analysis to get some defaults (now the main code uses `static` too for consistency)")  # static mode
    preparser.add_argument('--query', required=False, help="name of the parameter to get the run ID for (if not provided, all parameters are used)")
    preparser.add_argument('--print-config', action='store_true', help="print the full config file")
    # preparser.add_argument('--static', action='store_true', help="static code analysis for faster crunching of the run ID (otherwise import the modules)")  # static mode
    preargs, remainder = preparser.parse_known_args(cmd)
    preargs.otherargs = remainder  # store the rest of the arguments for the main parser

    parser = get_parser(static=preargs.static)
    # parser = get_parser()
    args = parse_args(preargs.otherargs, parser, options)

    if preargs.print_config:
        import json
        print(json.dumps(vars(args), indent=2))
        return 0

    if preargs.query is not None:
        if not hasattr(args, preargs.query):
            parser.print_help()
            raise ValueError(f"Parameter '{preargs.query}' not found in the arguments.")
        # if a parameter name is provided, get the run ID for that parameter
        value = getattr(args, preargs.query)
        if type(value) is list or type(value) is tuple:
            # if the value is a list, join it with commas
            value = ','.join(map(str, value))
        print(value)
        return 0

    print(get_runid(parser, **vars(args)))
    return 0

if __name__ == "__main__":
    main()