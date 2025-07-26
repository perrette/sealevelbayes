from scipy.stats import norm
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import xarray as xa
from sealevelbayes.datasets.manager import get_datapath, require_dataset
from sealevelbayes.logs import logger
from sealevelbayes.cache import cached
from sealevelbayes.datasets.ar6.tables import ar6_table_9_5
from sealevelbayes.datasets.climate import load_temperature
from sealevelbayes.datasets.cmip5 import read_cmip5_tglobal_df

from sealevelbayes.datasets.constants import kg_to_mm_sle, gt_to_mm_sle, km3_to_mm_sle


DATAPATH = get_datapath("")
MM21_FORCING = ['CRU TS 4.03', 'ERA20C', '20CRV3', 'CFSR', 'JRA55', 'ERA5', 'MERRA2', 'ERA_Interim', '(Mean input)', '(Median input)']

def parse_ar6_region(region):
    name, rest = region.split("(")
    number = rest.split(")")[0]
    return name.strip(), number.strip().replace("13–15", "13-15")

def get_ar6_table9sm2(drop_hma=False):
    """Load AR6 Table 9.SM.2"""
    ar6_table9sm2 = pd.read_csv(get_datapath("ar6_wg1/chap9/glaciers_ar6_table_9_sm_2.csv"))
    ar6_table9sm2.columns = [c.strip() for c in ar6_table9sm2.columns]
    ar6_table9sm2.index = [parse_ar6_region(r)[1] for r in ar6_table9sm2["Region"][:-1]]
    if drop_hma:
        ar6_table9sm2 = ar6_table9sm2[ar6_table9sm2.index != "13-15"]
        ar6_table9sm2.index = ar6_table9sm2.index.astype(int)
    return ar6_table9sm2


def load_zemp(region_number):
    fname = list(require_dataset("zenodo-3557199-zemp2019").glob(f"Zemp_etal_results_region_{region_number}_*.csv"))[0]
    return pd.read_csv(fname, skipinitialspace=True, comment='#')


@cached("glacier_datasets")
def load_glacier_datasets():


    ar6_table9sm2 = pd.read_csv(get_datapath("ar6_wg1/chap9/glaciers_ar6_table_9_sm_2.csv"))
    ar6_table9sm2.columns = [c.strip() for c in ar6_table9sm2.columns]
    ar6_region_map = {parse_ar6_region(r)[1]: r for r in ar6_table9sm2["Region"][:-1]}

    hock_path = require_dataset("zenodo-7492152-hock2023")
    f19 = pd.read_csv(hock_path / "fmaussion-global_glacier_volume_paper-882ae46/data/f19.csv")  # Farinotti et al 2019
    m22 = pd.read_csv(hock_path / "fmaussion-global_glacier_volume_paper-882ae46/data/m22_corr.csv")  # Millan et al 2022 (corrected for RGI 6.0 mask)

    hugonnetpath = require_dataset("hugonnet2021/41586_2021_3436_MOESM2_ESM.xlsx")
    h21 = pd.read_excel(hugonnetpath)
    h21.columns = [c.replace("\n"," ").strip() for c in h21.columns]
    h21 = h21.ffill()

    zemp_present = pd.read_csv(get_datapath("zemp2019/glaciers_zemp2019_table1.csv"))

    gmip_past_path = require_dataset("marzeionmalles2021/suppl_reconstruction_data_region.nc")
    gmip_past = xa.open_dataset(gmip_past_path)

    gmip_future_path = require_dataset("pangaea-MarzeionB-etal_2020/suppl_GlacierMIP_results.nc")
    gmip_future = xa.open_dataset(gmip_future_path)

    gmip_past = gmip_past.assign_coords(
        Forcing=list(gmip_past["Forcing"].attrs.values()),
    )

    gmip_future = gmip_future.assign_coords(
        Scenario=list(gmip_future["Scenario"].attrs.values()),
        Climate_Model=list(gmip_future["Climate_Model"].attrs.values()),
        Glacier_Model=list(gmip_future["Glacier_Model"].attrs.values()),
    )

    return ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future



# _ar6_glacier_lo, _ar6_glacier_mid, _ar6_glacier_hi = ar6_table_9_5["glacier"]["Δ (mm)"]
# from IPCC Table 9.5, summing lower and upper bounds as fully correlated
_ar6_glacier = ar6_table_9_5["glacier"]["Δ (mm)"]
_ar6_glacier_1901_1970 = (
    np.array(_ar6_glacier['1901-2018']) - np.array(_ar6_glacier['1971-2018']))


def get_gmip_glacier_regions(as_dict=False):
    GMIPDIR = get_datapath("zenodo-6419954-garner_kopp_2022/modules-data/ipccar6/gmipemuglaciers")
    df = pd.read_csv(GMIPDIR / "fingerprint_region_map.csv")
    if as_dict:
        return df.set_index("IceID")["IceName"].to_dict()

    return df


def get_ar6_sel(region_number):
    ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()
    return ar6_table9sm2[ar6_table9sm2["Region"] == ar6_region_map[str(region_number)]]


### Rate 2000
def _get_rate2000_ar6(ar6_sel):
    y_spec = -ar6_sel["Glacier mass change rate 2000-2019 (kg m-2 yr -1)"].item() * kg_to_mm_sle
    area = ar6_sel["Glacier-covered area in 2000 (km2)"].item() * 1e6
    y = y_spec * area

    # Now add uncertainty from area and from specific melt rate (assume they are indepedent since we dont now better)
    y_spec_sd = ar6_sel["rate_sd"].item() * kg_to_mm_sle
    area_sd = ar6_sel["area_sd"].item() * 1e6
    y_sd = ((area*y_spec_sd)**2 + (area_sd*y_spec)**2)**.5

    return y, y_sd


def get_rate2000_ar6(region_number):
    ar6_sel = get_ar6_sel(region_number)
    return _get_rate2000_ar6(ar6_sel)


def get_rate2000_h21(region_number, field="Mass balance (mm SLE yr-1)", period="2000-2019"):
    ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()
    xx = h21[(h21["RGI Region (#)"] == region_number) & (h21["Reference"] == "This Study") & (h21["Period"] == period)][field]
    x = float(xx.item().split("±")[0])
    x_sd = float(xx.item().split("±")[1])
    return x, x_sd


def get_rate2000_z19(region_number, field="Mass balance (mm SLE yr-1)", period="2000-2019"):
    ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()
    sel = zemp_present.iloc[region_number - 1] if region_number <= 19 else zemp_present.iloc[-1]
    xx = sel["Mass change (Gt yr−1)"]
#     xx = h21[(h21["RGI Region (#)"] == region_number) & (h21["Reference"] == "This Study") & (h21["Period"] == period)][field]
    x = -float(xx.split("±")[0]) * gt_to_mm_sle
    x_sd = float(xx.split("±")[1]) * gt_to_mm_sle
    return x, x_sd


### Past volume

def get_sle_ar6(region_number):
    ar6_sel = get_ar6_sel(region_number)
    return ar6_sel["Glacier mass in 2000 (mm SLE)"].item(), ar6_sel["mass_sd"].item()

def get_area_ar6(region_number):
    ar6_sel = get_ar6_sel(region_number)
    return ar6_sel['Glacier-covered area in 2000 (km2)'].item(), ar6_sel["area_sd"].item()

def get_sle_f19(region_number):
    ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()
    sel = f19.iloc[region_number-1]
    return sel["SLE"], sel["SLE_err"]

def get_area_f19(region_number):
    ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()
    sel = f19.iloc[region_number-1]
    return sel["A"], sel["A"]*0.1  # error not provided, but in AR6 it amount to 5-10%, and closer to 10% for HMA

def get_sle_mm21(region_number, year=None, include_uncertainty=True, slr=False):
    ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()
    mass = gmip_past["Mass"].sel(Region=region_number)*gt_to_mm_sle
    if slr:
        mass = mass.sel(Time=slice(1995-1900, 2014-1900)).mean("Time") - mass

    if year is not None:
        mass = mass.sel(Time=year-1900)

    x = mass.mean("Forcing")
    x_sd = mass.std("Forcing")

    if include_uncertainty:
        mass_u = gmip_past["Mass uncertainty"].sel(Region=region_number, Time=mass.Time)*gt_to_mm_sle
        x_sd = (x_sd**2 + (mass_u**2).mean("Forcing"))**.5

    if year is not None:
        x = x.item()
        x_sd = x_sd.item()

    return x, x_sd


### Past rate of change

def get_sle_change_mm21(region_number, period=(2000, 2018), rate=False):
    ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()
    mass = gmip_past["Mass"].sel(Region=region_number)*gt_to_mm_sle
    y1, y2 = period
    rate2000 = mass.sel(Time=y1-1900) - mass.sel(Time=y2-1900)
    if rate:
        rate2000 = rate2000/(y2-y1)
    x = rate2000.mean("Forcing").item()
    x_sd = rate2000.std("Forcing").item()
    return x, x_sd


### Future melt

def get_sle_future_m20(region_number, scenario):
    ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()
    # indices, scenarios = zip(*gmip_future.Scenario.attrs.items())
    # i_scenario = indices[scenarios.index(scenario)]
    if region_number is None:
        return gmip_future["Mass"].sel(Scenario=scenario) * gt_to_mm_sle
    else:
        return gmip_future["Mass"].sel(Region=region_number, Scenario=scenario) * gt_to_mm_sle


def get_sle_future_relative_m20(region_number, scenario, refyear=2005, normalized=True, targetyear=None, power=None):
    sle = get_sle_future_m20(region_number, scenario)
    if power is not None:
        sle = sle ** power # e.g. 1 - 0.76 ==> linearize the constraint for better convergence
    sle0 = sle.sel(Time=refyear)
    if targetyear:
        sle = sle.sel(Time=targetyear)
    sle = sle0 - sle
    if normalized:
        sle = sle / sle0
    return sle


def get_slr21_m20(region_number, scenario, refyear=2005, relative=False, **kw):
    proj = get_sle_future_relative_m20(region_number, scenario, refyear=refyear, targetyear=2099, normalized=relative, **kw)
    x = proj.mean(["Climate_Model", "Glacier_Model"]).item()
    x_sd = proj.std(["Climate_Model", "Glacier_Model"]).item()
    return x, x_sd

### Sum-up of current / past volume ranges

## Present-day volume

def get_volume_ranges(region_number):

    ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()

    ranges = {}
    for k, source in enumerate(["ar6", "f19", "m22"]):
        if source == "ar6":
            if region_number in [13, 14, 15]: continue # high-mountain glaciers are aggregated in AR6 table
            if region_number < 13:
                j = region_number - 1
            elif region_number < 20:
                j = region_number - 3
            else:
                j = 17
            sle = ar6_table9sm2.iloc[j]["Glacier mass in 2000 (mm SLE)"]
            sle_sd = ar6_table9sm2.iloc[j]["mass_sd"]

        elif source == "z19":
            j = -1 if region_number == 20 else region_number - 1
            sle = zemp_present.iloc[j]["Total volume (km3)"] * km3_to_mm_sle
            sle_sd = 0

        elif source == 'm22':
            if region_number in [1, 2, 13, 14, 15]: continue # aggregated in Millan et al
            if region_number < 13:
                j = region_number - 2
            elif region_number <= 19:
                j = region_number - 4
            else:
                j = 17
            sle = m22.iloc[j]["SLE"]
            sle_sd = m22.iloc[j]["SLE_err"]

        elif source == 'f19':
            sle = f19.iloc[region_number - 1]["SLE"]
            sle_sd = f19.iloc[region_number - 1]["SLE_err"]
        else:
            continue
#             raise NotImplementedError(source)

        ranges[source] = sle, sle_sd

    return ranges


### Sum-up of all useful data above

RCP_SCENARIOS = ["rcp26", "rcp45", "rcp60", "rcp85"]
SSP_SCENARIOS = ["ssp126", "ssp245", "ssp360", "ssp585"]
M20_SCENARIOS = ["RCP2.6", "RCP4.5", "RCP6.0", "RCP8.5"]
# Correspondence from short-form RCP scenarios and from SSP scenarios
# M20_SCENARIOS_MAP = {"rcp26": "RCP2.6", "rcp45": "RCP4.5", "rcp60": "RCP6.5", "rcp85": "RCP8.5"}
# M20_SCENARIOS_MAP.update({"ssp126": "RCP2.6", "ssp245": "RCP4.5", "ssp585": "RCP8.5"})

def _load_model_data(region_number,
            include_mass_uncertainty=True,
            v2000_source="mm21",
            rate2000_source="mm21",
            slr20_source="mm21",
            gt=False,
                ):

    # ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()

    r = {}

    if region_number in [19]: ## Antarctica
#             r["V1901"], r["V1901_sd"] = get_sle_f19(region_number)
        r["V1901"], r["V1901_sd"] = get_sle_ar6(region_number)
    else:
        r["V1901"], r["V1901_sd"] = get_sle_mm21(region_number, 1901, include_uncertainty=include_mass_uncertainty)

    logger.debug(f"Glacier {region_number} : volume source: {v2000_source}")

    if v2000_source == "ar6":

        # High Mountain Asia from Farinotti et al 2019
        if region_number in [13, 14, 15]:
            r["V2000"], r["V2000_sd"] = get_sle_f19(region_number)
            r["A2000"], r["A2000_sd"] = get_area_f19(region_number)

        else:
            r["V2000"], r["V2000_sd"] = get_sle_ar6(region_number)
            r["A2000"], r["A2000_sd"] = get_area_ar6(region_number)

    elif v2000_source == "mm21":

        if region_number in [19]: ## Antarctica
            r["V2000"], r["V2000_sd"] = get_sle_f19(region_number)

        else:
            r["V2000"], r["V2000_sd"] = get_sle_mm21(region_number, 2000, include_uncertainty=include_mass_uncertainty)

        r["A2000"], r["A2000_sd"] = get_area_f19(region_number)


    else:
        raise NotImplementedError(v2000_source)

    if slr20_source == "mm21":
        r["slr20_y1"], r["slr20_y2"] = y1, y2 = 1901, 2000

        if region_number != 19:
            r["slr20"], r["slr20_sd"] = get_sle_change_mm21(region_number, period=(y1, y2))
        else:
            r["slr20"], r["slr20_sd"] = 0, 20

    elif slr20_source == "z19":
        df = load_zemp(region_number).set_index("Year")
        x_ts, x_sd_ts = -df["INT_Gt"]*gt_to_mm_sle, df["sig_Total_Gt"]*gt_to_mm_sle
        mean_slr = x_ts.loc[1960:2000].sum()
        sd_slr = (x_sd_ts.loc[1960:2000]**2).sum()**.5
        r["slr20"], r["slr20_sd"] = mean_slr, sd_slr
        r["slr20_y1"], r["slr20_y2"] = 1960, 2000
    else:
        raise NotImplementedError(slr20_source)

    ## Present-day melt rate

    # AR6 and best obs alternatives (2000 - 2019)
    if rate2000_source == "ar6":

        # High Mountain Asia from Zemp (not avail in Hugonnet)
        if region_number in [13, 14, 15]:
            r["rate2000"], r["rate2000_sd"] = get_rate2000_z19(region_number)
            r["rate2000_y1"], r["rate2000_y2"] = 2006, 2016

        else:
            r["rate2000"], r["rate2000_sd"] = get_rate2000_ar6(region_number)
            r["rate2000_y1"], r["rate2000_y2"] = 2000, 2019


    # Malles and Marzeion 2021 reconstruction (1993-2019) (and obs alternative for AA, 2000-2019)
    elif rate2000_source == "mm21":

        if region_number in [19]:
            r["rate2000"], r["rate2000_sd"] = get_rate2000_ar6(region_number)
            r["rate2000_y1"], r["rate2000_y2"] = 2000, 2019

        else:
            r["rate2000"], r["rate2000_sd"] = get_sle_change_mm21(region_number, period=(1993, 2018), rate=True)
            r["rate2000_y1"], r["rate2000_y2"] = 1993, 2018

    else:
        raise NotImplementedError(rate2000_source)

    for x, x2 in zip(RCP_SCENARIOS, M20_SCENARIOS):
        r[f"slr21_rel_{x}"], r[f"slr21_rel_{x}_sd"] = get_slr21_m20(region_number, scenario=x2, relative=True)
        r[f"slr21_{x}"], r[f"slr21_{x}_sd"] = get_slr21_m20(region_number, scenario=x2, relative=False)
        for power in [1-0.76]:
            powertag = "pow"+str(power).replace(".", "")
            r[f"slr21_{powertag}_{x}"], r[f"slr21_{powertag}_{x}_sd"] = get_slr21_m20(region_number, scenario=x2, relative=False, power=power)
            r[f"slr21_{powertag}_rel_{x}"], r[f"slr21_{powertag}_rel_{x}_sd"] = get_slr21_m20(region_number, scenario=x2, relative=True, power=power)

    if gt:
        for k in r:
            if not k.endswith(("_y1", "_y2")):
                r[k] /= gt_to_mm_sle

    return r

def load_model_data(glacier_regions=None, normalize_future=True, fill_nans=True,
                    constraints_kwargs={}, experiment=None,
                    **kw):
    """load data used to constrain the glacier model
    """
    if glacier_regions is None:
        glacier_regions = np.arange(1, 20)

    records = [_load_model_data(region_number, **constraints_kwargs) for region_number in glacier_regions]

    datasets = xa.Dataset(coords={"region": glacier_regions, "experiment": RCP_SCENARIOS})

    for k in ["V1901", "V1901_sd",
              "rate2000", "rate2000_sd", "rate2000_y1", "rate2000_y2",
              "V2000", "V2000_sd",
              "slr20", "slr20_sd", "slr20_y1", "slr20_y2"
              ]:

        datasets[k] = ("region",), np.array([r[k] for r in records])

    for k in ["slr21_rel", "slr21", "slr21_pow024_rel", "slr21_pow024"]:
        datasets[f"{k}"] = ("region", "experiment"), np.array([[r[f"{k}_{x}"] for x in RCP_SCENARIOS] for r in records])
        datasets[f"{k}_sd"] = ("region", "experiment"), np.array([[r[f"{k}_{x}_sd"] for x in RCP_SCENARIOS] for r in records])

    # add time-series
    timeseries = load_merged_timeseries(normalize_future=normalize_future, fill_nans=fill_nans,
                                        stack_sample_dims=False, return_gsat=True, **kw)

    timeseries.Scenario.values[:] = [dict(zip(M20_SCENARIOS, RCP_SCENARIOS))[value] for value in timeseries.Scenario.values]

    datasets.update(
        timeseries.rename({"Time":"year", "Region":"region", "Scenario":"experiment", "Climate_Model":"model"})
        )

    if experiment is not None:
        datasets = datasets.sel(experiment=[experiment] if isinstance(experiment, str) else experiment)

    return datasets


def load_historical_timeseries(glacier_regions=None, rate=False, source="mm21"):
    """
    source: "mm21" or "zemp2019" or "mixed"
        "mixed" means "mm21" for all and "zemp2019" for region Antarctica
    """

    ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()

    if glacier_regions is None:
        glacier_regions = range(1, 20)

    if source in ("mm21", "mixed"):
        years = np.arange(1901, 2018+1)
    else:
        years = np.arange(1950, 2016+1)

    ts = []
    ts_unc = []

    for region_number in glacier_regions:

        if source in ("mm21", "mixed") and region_number < 19:
            variable = "Mass change" if rate else "Mass"
            slr, slr_unc = [sign * gmip_past[v].sel(Region=region_number).mean("Forcing").values*gt_to_mm_sle
                            for sign, v in [(-1, variable), (1, f"{variable} uncertainty")]]
            if (slr_unc < 0).any():
                raise ValueError(f"Negative uncertainty for source {source}, region {region_number}, {variable}")
            assert slr.shape[0] == len(years)

        elif source == "zemp2019" or (source == "mixed" and region_number == 19):
            zemp = load_zemp(region_number).set_index("Year").reindex(years)
            slr = -zemp["INT_Gt"] * gt_to_mm_sle
            slr_unc = zemp["sig_Total_Gt"] * gt_to_mm_sle
            if not rate:
                slr = slr.cumsum()
                slr_unc = ((slr_unc**2).cumsum())**.5

        else:
            slr = np.full(len(years), np.nan)
            slr_unc = np.full(len(years), np.nan)

        ts.append(slr)
        ts_unc.append(slr_unc)

    slr, slr_unc = (
        pd.DataFrame(np.array(ts).T, index=years, columns=glacier_regions),
        pd.DataFrame(np.array(ts_unc).T, index=years, columns=glacier_regions),
        )

    # throw away the years where all are NaNs
    slr = slr.dropna(how="all", axis=0)
    slr_unc = slr_unc.dropna(how="all", axis=0)
    assert (slr.index == slr_unc.index).all()
    assert ((np.isnan(slr_unc)) | (slr_unc > 0)).values.all(), slr_unc

    return slr, slr_unc


def sample_timeseries(mu, sigma, rng=None, sample_size=100, autocorrel=0, deterministic=False, random_seed=None):
    """Draw auto-correlated samples from a mean + standard deviation time-series

    Parameters
    ----------
    mu: 1-d array (mean)
    sigma: 1-d array (standard deviation)
    rng: numpy random generator
    sample_size: number of ensemble members
    autocorrel: autocorrelation coefficient
    deterministic: if True, use a deterministic quantile-based resampling

    Returns
    -------
    numpy array of shape (len(mu), sample_size)
    """

    nt = np.size(mu)

    # Now sample auto-correlated samples
    if autocorrel is not None and autocorrel != 0:
        if autocorrel == 1:
            autocorrel = 1 - 1e-9 # so that it is cholesky-decomposable
        correlation_matrix = autocorrel**np.abs(np.arange(nt)[:, None] - np.arange(nt)[None, :])
        chol = np.linalg.cholesky(correlation_matrix)
    elif autocorrel > 1:
        raise ValueError("Autocorrelation should be between -1 and 1")
    else:
        chol = np.eye(nt)

    if rng is None:
        rng = np.random.default_rng(random_seed if random_seed is not None else None)

    # deterministic sampling
    if deterministic:
        step = 1/sample_size
        quantile_levels = np.linspace(step/2, 1-step/2, sample_size)[None] + np.zeros((nt, sample_size))
        for i in range(nt):
            rng.shuffle(quantile_levels[i])
        iid_samples = norm().ppf(quantile_levels)

    else:
        iid_samples = rng.normal(size=(nt, sample_size))

    correlated_samples = chol @ iid_samples

    return mu[:, None] + sigma[:, None] * correlated_samples


def _load_zemp_with_forcing_dimension(region, sample_dim='Forcing', sample_dim_values=None, sample_size=10,
                                      resample=False, deterministic=True, **kw):
    """Draw auto-correlated samples from Zemp et al, to be used by load_zemp_with_forcing_dimension

    Parameters
    ----------
    region: region number
    sample_dim: name of the dimension for the ensemble
    sample_size: number of ensemble members
    resample: if True, draw samples from an auto-correlated distribution
    **kw : keyword arguments to be passed to sample_timeseries

    Returns
    -------
    xa.DataArray with dimensions {sample_dim=Forcing} x Time
    """
    dims = [sample_dim, "Time"]
    zemp = load_zemp(region).set_index("Year")
    rate = zemp["INT_Gt"] * gt_to_mm_sle
    rate_sd = zemp["sig_Total_Gt"] * gt_to_mm_sle
    nt = rate.index.size

    if sample_dim_values is None:
        sample_dim_values = np.arange(sample_size)

    # no sampling
    if not resample:
        return xa.DataArray(rate.values, coords={"Time": rate.index.values}).expand_dims(**{sample_dim:sample_dim_values}).transpose(*dims)

    else:
        # logger.info(f"Resampling Zemp et al. 2019 for {region} (deterministic={deterministic})")
        samples = sample_timeseries(rate.values, rate_sd.values, sample_size=sample_size, deterministic=deterministic, **kw)
        return xa.DataArray(samples, coords={"Time": rate.index.values, sample_dim:sample_dim_values}).transpose(*dims)


def load_zemp_with_forcing_dimension(regions=None, random_seed=0, rng=None, reindex_kwargs={}, **kw):
    """load Zemp time-series with the same format as the GIMP dataset, which means with an ensemble dimension of size 10 by default

    Parameters
    ----------
    regions: list of region numbers
    random_seed: random seed for the RNG
    rng: numpy random generator
    **kw: keyword arguments to be passed to _load_zemp_with_forcing_dimension

    Returns
    -------
    xa.DataArray with dimensions Forcing x Region x Time
    """

    if rng is None:
        rng = np.random.default_rng(random_seed if random_seed is not None else None)

    if regions is None:
        regions = np.arange(1, 20)

    years = np.arange(1950, 2016+1)

    return xa.concat([_load_zemp_with_forcing_dimension(region, rng=rng, **kw).reindex(Time=years, **reindex_kwargs) for region in regions], dim='Region').assign_coords(Region=regions)

def _fix_gmip_past(gmip_past, mm21_forcing=None, mm21_drop_20CRV3_17=False):

    if mm21_forcing is not None:
        gmip_past = gmip_past.sel(Forcing=mm21_forcing)

    # drop 20CRV3 Forcing for Region 17
    if mm21_drop_20CRV3_17 and "20CRV3" in gmip_past.Forcing.values and 17 in gmip_past.Region.values:
        gmip_past = gmip_past.copy(deep=True)
        for k in gmip_past:
            gmip_past[k].loc[dict(Forcing="20CRV3", Region=17)] = np.nan

    return gmip_past


def load_gmip_past_merged_with_zemp_antarctica(fill_value=0, last_year=2016, zemp_kwargs={}, **fix_mm21):
    # ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()

    years = np.arange(1901, last_year+1) # 2016 is the last year of Zemp 2019

    # Note gmip_past first year is NaN for the rate
    with open_mm21() as gmip_past:
        gmip_past = _fix_gmip_past(gmip_past[["Mass", "Mass change"]], **fix_mm21).reindex(Time=years)
        mass_1_18 = gmip_past["Mass"]*gt_to_mm_sle
        rate_1_18 = gmip_past["Mass change"]*gt_to_mm_sle

    rate_1_18.values[:, 0, :] = rate_1_18.values[:, 1, :] # back-filling of NaNs for first year (1901)

    # extend gmip_past with Antarctica
    rate_19 = _load_zemp_with_forcing_dimension(19, sample_dim='Forcing',
                                                sample_dim_values=gmip_past.Forcing.values, sample_size=gmip_past.Forcing.size, **zemp_kwargs).expand_dims(Region=[19]).transpose("Forcing", "Time", "Region")

    rate = xa.concat([rate_1_18, rate_19.reindex(Time=years, fill_value=fill_value)], dim="Region")

    mass_19 = rate_19.cumsum("Time")
    mass_19 = mass_19 - mass_19.sel(Time=2016) + get_sle_f19(19)[0] # set the volume using Farinotti et al 2019

    mass = xa.concat([mass_1_18, mass_19], dim="Region")

    return mass, rate


def _fill_ensemble_nans(data, dim):
    # fill-in the gaps in past rate (note this reduces the variance)
    data = data.copy()
    mean_rate = data.mean(dim).expand_dims(**{dim:data[dim]}).transpose(*data.dims)
    fill_in = np.isnan(data.values)
    data.values[fill_in] = mean_rate.values[fill_in]
    return data


def load_matching_gsat(coords, smooth_gsat=False, smooth_window=21, smooth_method='savgol_filter', fill_nans=False,
                       time_dim="Time", model_dim="Climate_Model", experiment_dim="Scenario"):
    # Also load global-mean temperature of the glacier models
    gsat = read_cmip5_tglobal_df()
    gsat -= gsat.loc[1995:2014].mean()

    # average out "ensemble_member"
    gsat = gsat.T.groupby(level=(0, 1)).mean().T

    # replace past with historical data, since the past glacier simulations are also not forced by CMIP
    gsat_hist = load_temperature().iloc[:, 0].reindex(gsat.index)
    gsat_hist -= gsat_hist.loc[1995:2014].mean()
    gsat.loc[:2014].values[:] = gsat_hist.loc[:2014].values[:, None]

    if smooth_gsat:
        if smooth_method == 'rolling_mean':
            gsat = gsat.rolling(2*(smooth_window//2)+1, center=True).mean().dropna(axis=0)
        else:
            gsat = pd.DataFrame(np.array([savgol_filter(gsat[x].values, window_length=2*(smooth_window//2)+1, polyorder=1) for x in gsat]).T,
                                index=gsat.index, columns=gsat.columns)

    gsat.index.name = time_dim
    gsat.columns.set_names([model_dim, experiment_dim], inplace=True)
    gsat.columns.name = "model_experiment"
    gsat = xa.DataArray(gsat).unstack("model_experiment")
    # rename rcp85 into RCP8.5 to match GMIP naming
    gsat.Scenario.values[:] = [dict(zip(RCP_SCENARIOS, M20_SCENARIOS))[value] for value in gsat[experiment_dim].values]
    gsat = gsat.reindex(**coords)

    if fill_nans:
        gsat = _fill_ensemble_nans(gsat, model_dim)

    return gsat


def load_merged_timeseries(normalize_future=True, fill_nans=True,
                           past_source="mm21", future_source="m20",
                           zemp_kwargs={}, present_day_volume_source="mm21",
                           past_antarctica_rate=0, past_antarctica_rate_array=None,
                           mm21_forcing=None, mm21_drop_20CRV3_17=False,
                           return_gsat=False, stack_sample_dims=False, **gsat_kw):
    """Load merged past to future time-series

    Return
    ------
    xa.Dataset with dimensions Scenario x Sample x Time x Region
    """

    ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()

    gmip_mm21_kwargs = dict(
        mm21_forcing=mm21_forcing,
        mm21_drop_20CRV3_17=mm21_drop_20CRV3_17,
    )

    if past_source == "mm21":
        years_past = np.arange(1901, 2018+1)
        years_future = np.arange(2019, 2100+1)
        # years = np.concatenate([years_past, years_future])

        past_mass, past_rate = load_gmip_past_merged_with_zemp_antarctica(fill_value=np.nan, last_year=2018,
                                                                          zemp_kwargs=zemp_kwargs, **gmip_mm21_kwargs)

        present_day_volumes = past_mass.sel(Time=2008)
        del past_mass

    elif past_source == "zemp2019":
        years_past = np.arange(1950, 2016+1)
        years_future = np.arange(2017, 2100+1)
        # years = np.concatenate([years_past, years_future])

        past_rate = load_zemp_with_forcing_dimension(**zemp_kwargs)

        if present_day_volume_source == "mm21":
            # load the reconstruction from Marzeion and Malles 2021 only to get the 2008 volume
            mm21_mass, _ = load_gmip_past_merged_with_zemp_antarctica(fill_value=np.nan, last_year=2018, **gmip_mm21_kwargs)
            present_day_volumes = mm21_mass.sel(Time=2008).mean("Forcing")
            del mm21_mass
        elif present_day_volume_source == "f19":
            regions = np.arange(1, 20)
            present_day_volumes = xa.DataArray([get_sle_f19(region) for region in regions], coords={"Region": regions}, dims="Region")
        else:
            raise NotImplementedError(present_day_volume_source)

    else:
        raise NotImplementedError(past_source)

    # assert past_rate.Forcing.size == 10, past_rate.Forcing

    dims = "Scenario", "Sample", "Time", "Region"
    sample_dims = "Forcing", "Glacier_Model", "Climate_Model"
    unstacked_dims = ("Scenario",) + sample_dims + ("Time", "Region")

    # Past simulations can finish anywhere between 2008 and 2018
    if fill_nans:
        past_rate = _fill_ensemble_nans(past_rate, 'Forcing')

    # assert past_rate.Forcing.size == 10, past_rate.Forcing

    def _return_result(rate):

        mass = rate.cumsum("Time")
        mass = mass - mass.sel(Time=2008) + present_day_volumes
        mass.values[np.isnan(rate.values)] = np.nan

        result = xa.Dataset(dict(mass=mass, rate=rate))

        if return_gsat:
            result['gsat'] = load_matching_gsat(dict(Time=rate.Time.values, Scenario=rate.Scenario.values, Climate_Model=rate.Climate_Model.values),
                                                time_dim="Time", model_dim="Climate_Model", experiment_dim="Scenario",
                                                fill_nans=fill_nans, **gsat_kw)
        elif len(gsat_kw) > 0:
            logger.warning("some key-word parameters were not used: %s", gsat_kw)

        if stack_sample_dims:
            result = result.stack(Sample=sample_dims).transpose(*dims)

        return result

    if future_source is None:
        dims = "Sample", "Time", "Region"
        sample_dims = "Forcing",
        return _return_result(past_rate)

    if future_source != "m20":
        raise NotImplementedError(future_source)

    future_mass = gmip_future["Mass"].expand_dims(Forcing=past_rate["Forcing"].values).transpose(*unstacked_dims) * gt_to_mm_sle

    future_rate = future_mass.diff("Time")

    past_rate = past_rate.expand_dims(
        Scenario=future_mass["Scenario"],
        Glacier_Model=future_mass["Glacier_Model"],
        Climate_Model=future_mass["Climate_Model"],).transpose(*unstacked_dims)

    # in 2008 all the past (and future, when avail) simulations are valid
    if normalize_future:
        ref_volume = future_mass.sel(Time=2008)
        scaling = present_day_volumes / ref_volume
        future_rate = future_rate * scaling

    del future_mass

    # Future simulations can start anywhere between 2000 and 2005
    if fill_nans:
        future_rate = _fill_ensemble_nans(future_rate, 'Glacier_Model')
        future_rate = _fill_ensemble_nans(future_rate, 'Climate_Model')  # RCP6.0 scenario is not simulated by all models: fill-in with multi-model mean
        future_rate = future_rate.ffill("Time") # also fill the 2100 value is needed

        # (note NaNs are still present when no models exist for certain Scenario and Climate_Model combinations)

    # Fill past rate with future projs for Antarctica (Zemp et al)
    if fill_nans and years_past[-1] > 2016:
        assert years_past[-1] == 2018, 'check nan filling of AA for past dataset other than mm21'
        past_rate = past_rate.copy()
        past_rate.sel(Region=19, Time=slice(2017, 2018)).values[:] = future_rate.sel(Region=19, Time=slice(2017, 2018)).values
        # past_rate.bfill("Time", inplace=True)

    if past_antarctica_rate_array is not None:
        logger.info(f"fill past_antarctica_rate with an array: {past_antarctica_rate_array}")
        assert np.size(past_antarctica_rate) == past_rate.Forcing.size
        if not fill_nans: past_rate = past_rate.copy()
        for i, rate in enumerate(past_antarctica_rate):
            vals = past_rate.sel(Region=19, Time=slice(None, 1980)).isel(Forcing=i).values
            vals[np.isnan(vals)] = rate

    elif past_antarctica_rate is not None:
        logger.info(f"fill past_antarctica_rate: {past_antarctica_rate}")
        if not fill_nans: past_rate = past_rate.copy()
        vals = past_rate.sel(Region=19, Time=slice(None, 1980)).values
        vals[np.isnan(vals)] = past_antarctica_rate

    merged_rate = xa.concat([past_rate, future_rate.sel(Time=years_future)], dim='Time')
    return _return_result(merged_rate)


# def _get_valid_mm21_1901_datasets(rate):
#     """Same result for all regions, with valid forcing in 1901:
#     datasets: CRU TS 4.03' | 'ERA20C' | '20CRV3' | '(Mean input)' | '(Median input)'
#     indices: array([0, 1, 2, 8, 9])
#     """
#     for i in range(1, 19):
#         m = np.isfinite(rate.sel(Region=i, Time=1901)).values
#         print(rate.sel(Region=i).Region.values, ":", " | ".join(map(repr,rate.Forcing.values[m])))
#     return np.where(m)[0]

import numpy as np
import pandas as pd
import statsmodels.api as sm

def extrapolate_time_series(long_series, short_series, num_samples=5000, rng=None, random_effects=True,
                            short_series_error=None, out_of_sample_predictor=None, extrapolate_error_pool=False):
    """
    Extrapolate a short time-series based on a linear regression against a longer time-series.

    Parameters:
    - long_series: pandas Series with 'year' as the index.
    - short_series: pandas Series with 'year' as the index.
    - num_samples: Number of random effect samples to draw.
    - rng: numpy random generator.
    - random_effects: If True, sample random effects.
    - short_series_error: pandas Series with 'year' as the index.
    - out_of_sample_predictor, optional: pandas Series with 'year' as the index
        if provided, will be used instead of "long_series" to extrapolate the short series.
        Useful to extend the long_series for a few years
    - asymmetric_random_effect: if True, split the calibration residuals into two parts and sample from each part separately.
        Otherwise (default), will sample random effects from the whole series.
    - extrapolate_error_pool: if True, pool sigma errors to extrapolate, otherwise just use either extremity

    Returns:
    - extrapolated_series: pandas Series of the extrapolated short time-series plus random effects.
    """
    # Find the overlapping period
    overlap_start = max(long_series.index.min(), short_series.index.min())
    overlap_end = min(long_series.index.max(), short_series.index.max())
    overlap_index = pd.Index(range(overlap_start, overlap_end + 1))

    long_overlap = long_series.loc[overlap_index]
    short_overlap = short_series.loc[overlap_index]

    if rng is None:
        rng = np.random.default_rng()

    # Fit a linear regression model
    X = sm.add_constant(long_overlap)
    model = sm.OLS(short_overlap, X).fit()

    # Extrapolate the short series
    if out_of_sample_predictor is not None:
        long_series_extended = out_of_sample_predictor
    else:
        long_series_extended = long_series

    extrapolated_series = model.predict(sm.add_constant(long_series_extended))
    extrapolated_series = np.asarray(extrapolated_series)[:, None]

    # Sample random effects
    if random_effects:
        residuals = model.resid
        extrapolated_series = extrapolated_series + rng.choice(residuals, size=(len(extrapolated_series), num_samples), replace=True)

        random_effects = rng.choice(residuals, size=(len(extrapolated_series), num_samples), replace=True)
        extrapolated_series = extrapolated_series + random_effects

    if short_series_error is not None:

        # Pool the error ?
        if extrapolate_error_pool:
            short_series_error_resampled = rng.choice(np.asarray(short_series_error), size=(len(extrapolated_series), num_samples), replace=True)

        # More common case
        else:
            short_series_error_resampled = short_series_error.reindex(long_series_extended.index).values
            short_series_error_resampled[long_series_extended.index > short_series.index.max()] = np.asarray(short_series_error)[-1]
            short_series_error_resampled[long_series_extended.index < short_series.index.min()] = np.asarray(short_series_error)[0]
            short_series_error_resampled = short_series_error_resampled[:, None]

        extrapolated_series = extrapolated_series + rng.normal(0, 1, size=(len(extrapolated_series), num_samples)) * short_series_error_resampled

    assert not np.isnan(extrapolated_series).any()

    return extrapolated_series.T


def _predict_past_glacier_samples(rng, years, samples=5000, predictor_label="(Mean input)", mm21_forcing=None, mm21_drop_20CRV3_17=False, **predict_kwargs):
    """
    Return an array of samples for the past glacier mass change based on the GMIP dataset.
    """

    # ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()

    # first draw uncorrelated samples for both datasets

    assert years[0] >= 1900

    zemp_rate = load_zemp_with_forcing_dimension(rng=rng, resample=True, sample_size=samples).transpose("Forcing", "Time", "Region").reindex(Time=years).values

    # mass_1_18 = gmip_past["Mass"].assign_coords({"Time": np.arange(1901, 2018+1)}).reindex(Time=years)*gt_to_mm_sle
    with open_mm21() as gmip_past:
        gmip_past = _fix_gmip_past(gmip_past[["Mass"]], mm21_forcing=mm21_forcing, mm21_drop_20CRV3_17=mm21_drop_20CRV3_17)

    rate_1_18 = gmip_past["Mass change"].reindex(Time=years)*gt_to_mm_sle
    assert np.isfinite(rate_1_18.values).any()


    # a linear regression model to predict the NaN values based on the mean forcing
    predicted = np.full((rate_1_18.Forcing.size + 1, samples, len(years), 19), np.nan) # +1 is Zemp
    predictor_label = "(Mean input)"
    predictor_1_18 = rate_1_18.sel(Forcing=predictor_label)

    labels = rate_1_18.Forcing.values.tolist() + ["zemp2019"]

    for i, region in enumerate(range(1, 19+1)):

        # all but Antarctica
        if region < 19:
            predictor = predictor_1_18.sel(Region=region).to_pandas()
        else:
            predictor = predictor_1_18.sel(Region=17).to_pandas() # Southern Andes is most correlated to Antarctica

        for j, forcing in enumerate(labels):

            # Predictor: just keep it as it is (but not Antarctica)
            if forcing == predictor_label:
                if region == 19:
                    continue

                predicted[j, :, 2:, i] = predictor.dropna().values[None, :].repeat(samples, axis=0)
                predicted[j, :, 0:1, i] = predicted[j, :, 2:2+1, i] # same rate for 1900 and 1901 for simplicity
                predicted[j, :, 1:2, i] = predicted[j, :, 2:2+1, i] # same rate for 1900 and 1901 for simplicity
                continue

            # ZEMP et al : all regions
            elif forcing == "zemp2019":

                target = pd.Series(zemp_rate[..., i].mean(axis=0), index=years).dropna() # mean of the Zemp dataset
                target_error = pd.Series(zemp_rate[..., i].std(axis=0), index=years).dropna()

            # OTHER FORCINGS: all but Antarctica
            else:
                if region == 19:
                    continue
                else:
                    target = rate_1_18.sel(Region=region, Forcing=forcing).to_pandas().dropna()
                target_error = None

            # 1900 and 1901 are missing: just use 1902 as predictor...
            is_valid_predictor = np.isfinite(predictor.values)
            predictor_out_of_sample = pd.Series(
                np.where(is_valid_predictor, predictor.values, predictor.values[is_valid_predictor][0]),
                index=predictor.index)

            predicted[j, :, :, i] = extrapolate_time_series(predictor.dropna(), target, num_samples=samples, rng=rng, short_series_error=target_error,
                                                                             out_of_sample_predictor=predictor_out_of_sample, **predict_kwargs)

    # assert not np.isnan(predicted).any()
    return xa.DataArray(predicted, coords={"Forcing": labels, "Sample": np.arange(samples), "Time": years, "Region": np.arange(1, 19+1)}, dims=["Forcing", "Sample", "Time", "Region"])


def predict_past_glacier_samples(samples=5000, years=None, rng=None, random_seed=None, include_zemp=True, **kwargs):

    if years is None:
        years = np.arange(1900, 2018+1)

    if rng is None:
        rng = np.random.default_rng(random_seed if random_seed is not None else None)

    predicted_xa =  _predict_past_glacier_samples(rng, years, samples=samples, **kwargs)
    # Now merge the various "forcings" giving equal weight to "Zemp" and to all other forcings

    # ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()
    # predicted = predict_past_glacier_samples(samples=samples, rng=rng, years=years)
    labels = predicted_xa.Forcing.values.tolist()

    if include_zemp:
        zemp_index = rng.integers(0, 2, size=samples) == 1 # 1/2 of the samples are from Zemp
    else:
        zemp_index = np.zeros(samples, dtype=bool)
    forcing_index = rng.choice([i for i, f in enumerate(labels) if f != "zemp2019"], size=samples, replace=True)
    forcing_index[zemp_index] = labels.index("zemp2019")

    predicted_values = predicted_xa.values[forcing_index, np.arange(samples), :, :]
    # now Antarctica is only Zemp
    predicted_values[:, ..., 18] = predicted_xa.values[labels.index("zemp2019"), :, :, 18]

    assert not np.isnan(predicted_values).any()
    return xa.DataArray(predicted_values, coords={"Sample": np.arange(samples), "Time": years, "Region": np.arange(1, 19+1)})


def handle_drop_mm21_20CRV3_17(func):
    """ This is a hack, because the draw-past-glacier-samples doesn't currently support dropping a forcing for a single region.
    """
    def newfunc(*args, **kwargs):
        # HACK
        mm21_drop_20CRV3_17 = kwargs.pop("mm21_drop_20CRV3_17", False)

        rng = kwargs.pop("rng", None)
        if rng is None:
            rng = np.random.default_rng(seed=kwargs.get("random_seed", None))
        kwargs["rng"] = rng

        mm21_forcing = kwargs.pop("mm21_forcing", None)
        if mm21_forcing is None:
            mm21_forcing = MM21_FORCING

        res = func(*args, **kwargs)

        if not mm21_drop_20CRV3_17:
            return res

        if "20CRV3" not in mm21_forcing:
            return res

        if 17 not in res.Region.values:
            return res

        kwargs["mm21_forcing"] = [f for f in mm21_forcing if f != "20CRV3"]
        res_without = func(*args, **kwargs)

        logger.warning("Dropping 20CRV3 for region 17")
        res.loc[dict(Region=17)] = res_without.loc[dict(Region=17)]

        return res

    return newfunc


@handle_drop_mm21_20CRV3_17
def draw_past_glacier_samples(samples=5000, random_seed=None, rng=None, years=None,
                              fill_missing=np.nan, regress_on_mm21=False, regress_kwargs={}, include_zemp=True, include_mm21_error=True, **fix_mm21):
    """
    regress_on_mm21_mean: if True, use the mean forcing of the MM21 dataset as a predictor to fill missing values in the past dataset (outside Antarctica)
    """

    # ar6_table9sm2, f19, m22, h21, zemp_present, ar6_region_map, gmip_past, gmip_future = load_glacier_datasets()

    # zemp_years = np.arange(1950, 2016+1)
    # mm21_years = np.arange(1901, 2018+1)

    if years is None:
        years = np.arange(1900, 2018+1)

    if rng is None:
        rng = np.random.default_rng(random_seed if random_seed is not None else None)

    # first draw uncorrelated samples for both datasets

    assert years[0] >= 1900

    zemp_rate = load_zemp_with_forcing_dimension(rng=rng, resample=True, sample_size=samples).transpose("Forcing", "Time", "Region").reindex(Time=years).values

    if include_zemp:
        zemp_index = rng.integers(0, 2, size=samples) == 1 # 1/2 of the samples are from Zemp
    else:
        zemp_index = np.zeros(samples, dtype=bool)

    # mass_1_18 = gmip_past["Mass"].assign_coords({"Time": np.arange(1901, 2018+1)}).reindex(Time=years)*gt_to_mm_sle
    gmip_past = open_mm21()
    with gmip_past:
        gmip_past = _fix_gmip_past(gmip_past, **fix_mm21)
        rate_1_18 = gmip_past["Mass change"].reindex(Time=years)*gt_to_mm_sle
        rate_1_18_error = gmip_past["Mass change uncertainty"].reindex(Time=years)*gt_to_mm_sle


    assert np.isfinite(rate_1_18.values).any()

    if regress_on_mm21:
        # Fill auto-correlated MM21 dataset with predicted values
        predicted_xa = predict_past_glacier_samples(samples=samples, rng=rng, years=years, include_zemp=include_zemp, **regress_kwargs)
        predicted = predicted_xa.values

        # use MM21 when available and not zemp_index
        forcing_index = rng.integers(0, rate_1_18.Forcing.size, size=samples)
        forcing_error_scale = rng.normal(0, 1, size=(samples, len(years), rate_1_18.Region.size))
        mm21_correlated = rate_1_18.values[forcing_index]
        if include_mm21_error:
            mm21_correlated += forcing_error_scale * rate_1_18_error.values[forcing_index]
        fill_mm21 = ~np.isnan(mm21_correlated) & ~zemp_index[:, None, None]
        predicted[:, :, :18][fill_mm21] = mm21_correlated[fill_mm21]

        # use Zemp when available zemp index
        fill_zemp = np.isfinite(zemp_rate) & zemp_index[:, None, None]
        predicted[fill_zemp] = zemp_rate[fill_zemp]
        assert not np.isnan(predicted).any()

        return xa.DataArray(predicted, coords=predicted_xa.coords)


    mm21_uncorrelated = np.full((samples, len(years), rate_1_18.Region.size), np.nan)
    mm21_forcing_error = np.full((samples, len(years), rate_1_18.Region.size), np.nan)
    forcing_error_scale = rng.normal(0, 1, size=(samples, len(years), rate_1_18.Region.size))

    for i, year in enumerate(years):
        if year < 1902:
            sample_from_year = 1902
        else:
            sample_from_year = year

        # The forcing datasets are the same across regions for each given year
        rate_1_18_that_year = rate_1_18.sel(Time=sample_from_year).values
        rate_1_18_error_that_year = rate_1_18_error.sel(Time=sample_from_year).values
        valid_forcing_dataset_per_region = np.isfinite(rate_1_18_that_year)
        assert valid_forcing_dataset_per_region.any(), (year, sample_from_year)
        valid_forcing_dataset = valid_forcing_dataset_per_region.all(axis=-1)
        assert (valid_forcing_dataset[..., None] == valid_forcing_dataset_per_region).all(axis=-1).all(), f"some regions have different valid forcing datasets {sample_from_year}"
        valid_forcing_dataset_idx = np.where(valid_forcing_dataset)[0]
        if not valid_forcing_dataset.any():
            raise ValueError(f"No valid forcing dataset for year {sample_from_year}")

        resampled_that_year = rng.integers(0, valid_forcing_dataset_idx.size, size=samples)

        mm21_uncorrelated[:, i, :] = rate_1_18_that_year[valid_forcing_dataset_idx[resampled_that_year]]
        mm21_forcing_error[:, i, :] = rate_1_18_error_that_year[valid_forcing_dataset_idx[resampled_that_year]] * forcing_error_scale[:, i, :]

    # add the intra-forcing error to inter-forcing error
    if include_mm21_error:
        mm21_uncorrelated += mm21_forcing_error

    # Fill auto-correlated MM21 dataset with uncorrelated dataset
    forcing_index = rng.integers(0, rate_1_18.Forcing.size, size=samples)
    mm21_correlated = rate_1_18.values[forcing_index]
    if include_mm21_error:
        mm21_correlated += mm21_forcing_error
    fill_nans_with_uncorrelated_mm21 = np.isnan(mm21_correlated)
    mm21_correlated[fill_nans_with_uncorrelated_mm21] = mm21_uncorrelated[fill_nans_with_uncorrelated_mm21]

    # Fill years before 1950 and after 2016 with MM21 data

    # outside Antarctica
    missing = np.isnan(zemp_rate)
    zemp_rate[..., :18][missing[..., :18]] = mm21_uncorrelated[missing[..., :18]]

    # Antarctica
    years_valid = np.isfinite(zemp_rate[0, :, 18])

    first_valid_year = np.where(years_valid)[0][0]
    before_1950 = years < years[first_valid_year]
    idx = rng.integers(0, samples, size=before_1950.sum()*samples) # draw from the first year samples (1950)
    zemp_rate[:, before_1950, 18] = zemp_rate[idx, first_valid_year, 18].reshape(samples, -1)

    last_valid_year = np.where(years_valid)[0][-1]
    after_1950 = years > years[last_valid_year]
    idx = rng.integers(0, samples, size=after_1950.sum()*samples) # draw from the first year samples (1950)
    zemp_rate[:, after_1950, 18] = zemp_rate[idx, last_valid_year, 18].reshape(samples, -1)

    assert np.isfinite(zemp_rate).all(), "Some missing values in Zemp rate after filling"

    # Sample across MM21 and Zemp for each year and each region including Antarctica
    resampled = np.full((samples, len(years), 19), fill_value=fill_missing)

    # All regions but Antarctica: half of the samples are from MM21
    resampled[~zemp_index, :, :18] = mm21_correlated[~zemp_index, :, :] # initialized with the correlated MM21 data

    # Antarctica: use Zemp et al samples
    resampled[~zemp_index, :, 18] = zemp_rate[~zemp_index, :, 18] # Antarctica

    # Other half from Zemp
    resampled[zemp_index, :, :] = zemp_rate[zemp_index, :, :] # ZEMP

    return xa.DataArray(resampled, coords={"Sample": np.arange(samples), "Time": years, "Region": np.arange(1, 20)}, dims=["Sample", "Time", "Region"])


def open_mm21():
    require_dataset("marzeionmalles2021/suppl_reconstruction_data_region.nc")
    return xa.open_dataset(get_datapath("marzeionmalles2021/suppl_reconstruction_data_region.nc")).assign_coords({"Time": np.arange(1901, 2018+1), "Region": np.arange(1, 18+1), "Forcing": MM21_FORCING})


def load_mm21_constraints(rate=True, slr_eq=True, **kwargs):
    with open_mm21() as ds:
        if rate:
            names = ["Mass change", "Mass change uncertainty"]
        else:
            names = ["Mass", "Mass uncertainty"]

        ds = ds[names]
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if kwargs:
            ds = ds.sel(**kwargs)
        ds = ds.rename(dict(zip(names, ["obs", "obs_sd"])))
        if slr_eq:
            ds["obs"] = ds["obs"] * gt_to_mm_sle * (-1)
            ds["obs_sd"] = ds["obs_sd"] * gt_to_mm_sle

        return ds
    # df = ds.isel(forcing=i).to_pandas().dropna(axis=0)

    # for i, forcing in enumerate(ds.Forcing):
    #     df = ds.isel(forcing=i).to_pandas().dropna(axis=0)
    #     obs = df[names[0]].values * gt_to_mm_sle * (-1)
    #     obs_sd = df[names[1]].values * gt_to_mm_sle
    #     obs_years = df.index.values
    #     yield (obs, obs_sd, obs_years)
