"""A module to build temperature scenarios
"""
from itertools import groupby
import numpy as np
import xarray as xa
from scipy.stats import norm
import pandas as pd

from sealevelbayes.config import logger
from sealevelbayes.datasets.manager import get_datapath
from sealevelbayes.datasets.catalogue import require_ar6_wg3

AR6GSAT = get_datapath("AR6_Projections/SurfaceTemperatureGlobal")


SSP_EXPERIMENTS = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]
IMP_EXPERIMENTS = ['Ren', 'SP', 'LD', 'Ren-2.0', 'GS', 'Neg-2.0', 'Neg', 'ModAct', 'CurPol']
CX_EXPERIMENTS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

def get_imp_experiment_data(experiments=IMP_EXPERIMENTS, df=None):
    if df is None: df = load_ar6_wg3_scenarios()
    model = np.array([s.split('|')[2] for s in df['variable']])
    df = df[(df["IMP_marker"] != "non-IMP") & (model == "MAGICCv7.5.3")]
    i2000 = df.columns.tolist().index('2000-01-01 00:00:00')

    year0, low0, mean0, high0 = np.loadtxt(f"{AR6GSAT}/tas_global_Historical.csv", skiprows=1, delimiter=",").T
    baseline = mean0[1995-int(year0[0]):2014+1-int(year0[0])].mean()
    baseline2 = mean0[2000-int(year0[0])]

    def get_dict(df):
        # df = df.set_index('pct')
        pct = np.array([float(s.split('|')[3][:-len("th Percentile")]) for s in df['variable']])
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

    year0, low0, mean0, high0 = np.loadtxt(f"{AR6GSAT}/tas_global_Historical.csv", skiprows=1, delimiter=",").T
    baseline = mean0[1995-int(year0[0]):2014+1-int(year0[0])].mean()

    experiments_data = {}
    for i, x in enumerate(experiments):
        ssp = f"{x[:4]}_{x[4]}_{x[5]}".upper() # ssp585 => SSP5_8_5
        year, low, mean, high = np.loadtxt(f"{AR6GSAT}/tas_global_{ssp}.csv", skiprows=1, delimiter=",").T
        assert 2014 == year[0]-1  # mean0 starts in 1950, so we use tas_df
        experiments_data[x] = {
            "median": mean - baseline,
            "p95": high - baseline,
            "p05": low - baseline,
            "years": np.arange(2015, 2099+1),
        }

    return experiments_data


def sample_ar6_categories_scenarios(size, experiments=None, random_seed=None, uncertainty=True):

    import ar6.misc

    tas_df = ar6.misc.load_temperature()
    tas_pi = tas_df['ssp585'].loc[1995:2014].mean() - tas_df['ssp585'].loc[1855:1900].mean()
    tas_df = tas_df.loc[1900:2099] # only defined up to 2099
    tas_df -= tas_df.loc[1995:2014].mean()

    tas_hist = tas_df['ssp585'].loc[:2014].values

    rng = np.random.default_rng(seed=random_seed + 98097 if random_seed else None)

    s = rng.normal(size=size)

    cats = make_ar6_categories_scenarios()

    if experiments is None:
        experiments = list(cats.keys())

    shape = size if type(size) is tuple else (size,)
    tas = np.empty( shape + (len(experiments), 2099-1900+1))

    for i, x in enumerate(experiments):
        n = len(cats[x])
        sample_x = rng.integers(n, size=size)

        mean = np.array([np.concatenate([tas_hist, np.array(r['mu'][15:100]) - tas_pi]) for r in cats[x]])
        if uncertainty:
            p95 = np.array([np.concatenate([tas_hist, np.array(r['95th'][15:100]) - r['95th'][14] + r['mu'][14] - tas_pi]) for r in cats[x]])
            p05 = np.array([np.concatenate([tas_hist, np.array(r['5th'][15:100]) - r['5th'][14] + r['mu'][14] - tas_pi]) for r in cats[x]])
            sigma_up = (p95-mean)/norm.ppf(0.95)  # 1.64 is the factor to pass from 90% to 1-sigma range
            sigma_lo = (mean-p05)/norm.ppf(0.95)  # 1.64 is the factor to pass from 90% to 1-sigma range
            samples = mean[sample_x] + np.where(s[..., None] > 0, s[..., None]*sigma_up[sample_x], s[..., None]*sigma_lo[sample_x])
        else:
            samples = mean[sample_x]
        tas[..., i, :] = samples


    coords={"experiment": experiments, "year": np.arange(1900, 2099+1)}

    if np.ndim(size) == 0:
        dims = ("sample", "experiment", "year")
        # coords['sample'] = np.arange(size)
    else:
        dims = ("chain", "draw", "experiment", "year")
        # coords['chain'] = np.arange(size[0])
        # coords['draw'] = np.arange(size[1])

    return xa.DataArray(tas, dims=dims, coords=coords)


def load_ar6_wg3_scenarios():
    return pd.read_csv(require_ar6_wg3())


if __name__ == "__main__":
    load_ar6_wg3_scenarios()
