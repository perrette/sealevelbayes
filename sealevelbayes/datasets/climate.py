"""A module to build temperature scenarios
"""
import numpy as np
import pandas as pd

from sealevelbayes.datasets.manager import get_datapath, require_dataset


SSP_EXPERIMENTS = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]
IMP_EXPERIMENTS = ['Ren', 'SP', 'LD', 'Ren-2.0', 'GS', 'Neg-2.0', 'Neg', 'ModAct', 'CurPol']
CX_EXPERIMENTS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

def get_imp_experiment_data(experiments=IMP_EXPERIMENTS, df=None):
    if df is None: df = load_ar6_wg3_scenarios()
    model = np.array([s.split('|')[2] for s in df['variable']])
    df = df[(df["IMP_marker"] != "non-IMP") & (model == "MAGICCv7.5.3")]
    i2000 = df.columns.tolist().index('2000-01-01 00:00:00')

    AR6GSAT = require_dataset("ar6_wg1/spm_08/v20210809/panel_a")
    year0, low0, mean0, high0 = np.loadtxt(AR6GSAT/"tas_global_Historical.csv", skiprows=1, delimiter=",").T
    baseline = mean0[1995-int(year0[0]):2014+1-int(year0[0])].mean()
    baseline2 = mean0[2000-int(year0[0])]

    def get_dict(df):
        # df = df.set_index('pct')
        # pct = np.array([float(s.split('|')[3][:-len("th Percentile")]) for s in df['variable']])
        lo, med, hi = (df.iloc[:, i2000:].values - df.iloc[:, i2000].values[:, None])
        s = df.iloc[0]

        return {
            "median": med + baseline2 - baseline,
            "p95": hi + baseline2 - baseline,
            "p05": lo + baseline2 - baseline,
            "years": np.arange(2000, 2100+1),
                # "mu": np.array(med),
                # "sigma": (hi - lo)/2/norm.ppf(0.95),
                # "5th": lo,
                # "95th": hi,
            "category": s['Category'],
            "climate model": s['variable'].split('|')[2],
            "model": s['model'],
            "scenario": s['scenario'],
            "IMP_marker": s['IMP_marker'],
            }

    return {x: get_dict(df[df['IMP_marker'] == x]) for x in experiments}


def get_ssp_experiment_data(experiments=SSP_EXPERIMENTS):
    """return a dict of experiments with fields

    "years": must be from 2014 to 2099 (will be re-indexed as such)
    "median": median time-series of temperature above to 1995-2014 baseline
    "p05": 5th percentile
    "p95": 95th percentile
    """
    AR6GSAT = require_dataset("ar6_wg1/spm_08/v20210809/panel_a")
    year0, low0, mean0, high0 = np.loadtxt(AR6GSAT/"tas_global_Historical.csv", skiprows=1, delimiter=",").T
    baseline = mean0[1995-int(year0[0]):2014+1-int(year0[0])].mean()

    experiments_data = {}
    for i, x in enumerate(experiments):
        ssp = f"{x[:4]}_{x[4]}_{x[5]}".upper() # ssp585 => SSP5_8_5
        year, low, mean, high = np.loadtxt(AR6GSAT/ f"tas_global_{ssp}.csv", skiprows=1, delimiter=",").T
        assert 2014 == year[0]-1  # mean0 starts in 1950, so we use tas_df
        experiments_data[x] = {
            "median": mean - baseline,
            "p95": high - baseline,
            "p05": low - baseline,
            "years": np.arange(2015, 2099+1),
        }

    return experiments_data

def load_ar6_wg3_scenarios():
    return pd.read_csv(require_dataset("zenodo-6496232-AR6-WG3-plots/spm-box1-fig1-warming-data.csv"))


def load_temperature():
    """Load median temperature scenarios from AR6 (ssp19,ssp26,ssp45,ssp70,ssp85) merged with historical scenarios.

    See data folder: AR6_Projections/SurfaceTemperatureGlobal for sources.
    (Data obtained from AR6 supplementary material and online repos.)

    Past data is obtained from SPM1_1850-2020_obs.csv (kind of smooth) and tas_global_Historical.csv (normally smooth but starting only in 1950)
    The merged time-series start in 1855.

    Recipe: notebooks/topics/temperature-hist-rcp-ssp-ar5-ar6.ipynb
    """
    filepath = get_datapath("savedwork/tas_ar6_merged.csv")
    if not filepath.exists():
        df = create_merged_temperature_df()
        df = df.to_csv(filepath)
    return pd.read_csv(filepath).set_index("year")


def create_merged_temperature_df():
    """Create a merged DataFrame with historical and SSP scenarios."""

    spm_08 = require_dataset("ar6_wg1/spm_08/v20210809/panel_a")
    spm_01 = require_dataset("ar6_wg1/spm_01/v20221116/panel_a")

    tas_ar6_85 = np.loadtxt(spm_08 / "tas_global_SSP5_8_5.csv", skiprows=1, delimiter=",")[:, [0, 2]]
    tas_ar6_70 = np.loadtxt(spm_08 / "tas_global_SSP3_7_0.csv", skiprows=1, delimiter=",")[:, [0, 2]]
    tas_ar6_45 = np.loadtxt(spm_08 / "tas_global_SSP2_4_5.csv", skiprows=1, delimiter=",")[:, [0, 2]]
    tas_ar6_26 = np.loadtxt(spm_08 / "tas_global_SSP1_2_6.csv", skiprows=1, delimiter=",")[:, [0, 2]]
    tas_ar6_19 = np.loadtxt(spm_08 / "tas_global_SSP1_1_9.csv", skiprows=1, delimiter=",")[:, [0, 2]]

    tas_ar6_hist = np.loadtxt(spm_08 / "tas_global_Historical.csv", skiprows=1, delimiter=",")[:, [0, 2]]
    tas_ar6_hist2 = np.loadtxt(spm_01 / "SPM1_1850-2020_obs-clean.csv", skiprows=1, delimiter=",")[::-1]

    sat_ar6_hist = pd.Series(tas_ar6_hist[:,1], index=tas_ar6_hist[:,0])
    sat_ar6_hist2 = pd.Series(tas_ar6_hist2[:,1], index=tas_ar6_hist2[:,0])

    years = np.arange(1855, 2100)

    def merged_with_past(ssp):
        merged = np.empty(years.size)
        merged[years < 1950] = sat_ar6_hist2.loc[:1949].values
        merged[(years >= 1950) & (years < 2015)] = sat_ar6_hist.loc[1950:2014].values
        merged[years >= 2015] = ssp
        return merged


    df = pd.DataFrame({
        "year":years,
        "ssp119":merged_with_past(tas_ar6_19[:,1]),
        "ssp126":merged_with_past(tas_ar6_26[:,1]),
        "ssp245":merged_with_past(tas_ar6_45[:,1]),
        "ssp370":merged_with_past(tas_ar6_70[:,1]),
        "ssp585":merged_with_past(tas_ar6_85[:,1]),
        }).set_index("year")

    return df