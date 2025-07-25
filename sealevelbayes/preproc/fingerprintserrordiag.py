import xarray as xa
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from sealevelbayes.preproc.fingerprintserror import filter_data, load_data, make_fingerprint_error_diag, get_gps_sampling_period
from sealevelbayes.postproc.figures import argsort_stations_by_id


def load_all_data(variables=["tws", "AIS", "GrIS", "glac", "all"],
                  smoothings=[None, 15],
                  fields=["rad", "rsl"],
                  fingerprint_period=(2000, 2018),
                  data=None):
    if data is None:
        data = {}
    for k in variables:
        for s in smoothings:
            for f in fields:
                key = (f, k, s)
                if key not in data:
                    if k == "all":
                        data[key] = sum(data[(f, kk, s)] for kk in ["tws", "AIS", "GrIS", "glac"])
                    else:
                        data[key] = load_data(k, fingerpring_period=fingerprint_period, smoothing=s, field=f)

    return data

def make_diag_all_data(data, high_passes=[None, 15], diags=None):
    if diags is None:
        diags = {}
    for k in data:
        for high_pass in high_passes:
            key = k + (high_pass,)
            if key not in diags:
                diags[key] = make_fingerprint_error_diag(data[k], gps_period=(2000, 2018), high_pass=high_pass)
    return diags

def filtered(x, high_pass=None):
    values = filter_data(x.values, high_pass=high_pass, inplace=False)
    return xa.DataArray(values, coords=x.coords, dims=x.dims)

def format_field(field):
    return "RSL" if field == "rsl" else "VLM"

def format_key(key, id=None):
    title = f"{format_field(key[0])} | {key[1]}"
    if len(key) > 2 and key[2] is not None:
        title += f" | Smoothing {key[2]}y"
    if len(key) > 3 and key[3] is not None:
        title += f" | High pass {key[3]}y"
    if id is not None:
        title += f" | station {id}"
    return title

def check_model_obs(data, ids, fields=["rad"], smoothings=[15], high_passes=[None], variables=["tws", "AIS", "GrIS", "glac"], ni=None, nj=2, axes=None, **kwargs):
    items = [((f, k, s), id, h) for f in fields for k in variables for s in smoothings for h in high_passes for id in ids]
    if axes is None:
        n = len(items)
        ni = ni or int(np.ceil(n / nj))
        f, axes = plt.subplots(ni, nj, figsize=(10, 4*ni))
    else:
        f = axes.flat[0].get_figure()
    for (key, id, high_pass), ax in zip(items, axes.flat):
        filtered(data[key]['model'].sel(station=id), high_pass).plot(label='model', ax=ax, **kwargs)
        filtered(data[key]['obs'].sel(station=id), high_pass).plot(label='obs', ax=ax, **kwargs)
        title = format_key(key+(high_pass,), id)
        ax.set_title(title)
        ax.legend()
    f.tight_layout()
    return f, axes


def _check_model_obs_trends_samples(data, diag, id, label="", period=None, ni=None, nj=2, axes=None):
    f, axes = plt.subplots(2, 2, figsize=(10, 8))
    if period is None:
        period = get_gps_sampling_period([id])[0]
    ax = axes.flat[0]
    data['model'].sel(station=id).plot(label='model', ax=ax)
    data['obs'].sel(station=id).plot(label='obs', ax=ax)
    ax.set_title(f"{label}\nGPS period: {period[0]:.0f}-{period[1]:.0f}")
    ax.legend()

    for k, ax in zip(["model_trends_samples", "obs_trends_samples", "trend_errors_samples"], axes.flat[1:]):
        # values = diags[src][k].sel(station=id).values
        if k == "trend_errors_samples":
            check = diag["model_trends"].sel(station=id).values - diag["obs_trends"].sel(station=id).values
            check2 = (diag[k].sel(station=id)**2).mean()**.5
        elif k == "model_trends_samples":
            check = diag["model_trends"].sel(station=id).values
            check2 = diag[k].sel(station=id).mean()
        elif k == "obs_trends_samples":
            check = diag["obs_trends"].sel(station=id).values
            check2 = diag[k].sel(station=id).mean()
        ax.set_title(f"{k} (gps period: {check:.2f}; ALL: {check2:.2f})")
        ax.hist(diag[k].sel(station=id).values, bins=20, edgecolor='k')

    f.tight_layout()
    return f, axes

def check_model_obs_trends_samples(data, diags, id, field="rad", variable="tws", smoothing=15, high_pass=None):
    key = (field, variable, smoothing)
    key2 = key + (high_pass,)
    return _check_model_obs_trends_samples(data[key], diag=diags[key2], id=id, label=format_key(key2, id=id));


def check_trend_error_hist_resampled(diags, ids=None, fields=["rad"], smoothings=[15], high_passes=[None], variables=["tws", "AIS", "GrIS", "glac"], ni=None, nj=2, axes=None, **kwargs):
    items = [(f, k, s, h) for f in fields for k in variables for s in smoothings for h in high_passes]
    if axes is None:
        n = len(items)
        ni = ni or int(np.ceil(n / nj))
        f, axes = plt.subplots(ni, nj, figsize=(10, 4*ni))
    else:
        f = axes.flat[0].get_figure()
    kwargs.setdefault('bins', 20)
    kwargs.setdefault('edgecolor', 'k')
    kwargs.setdefault('density', True)
    kwargs.setdefault('alpha', 0.5)
    for key, ax in zip(items, axes.flat):
        trend_errors = diags[key]['model_trends'] - diags[key]['obs_trends']
        hist0, bins, _ = ax.hist(trend_errors, label='actual errors', **kwargs)
        rms = diags[key]['rms'].values
        cov = diags[key]['cov_rms'].values
        cov = cov + np.eye(cov.shape[0])*1e-6
        valid = np.isfinite(rms)
        samples = np.random.multivariate_normal(mean=np.zeros(rms[valid].size), cov=cov[valid][:, valid], size=100)
        hists = []
        for i in range(samples.shape[0]):
        #     ax.hist(samples[i], label=f'sampled error #{i+1}', facecolor="none", density=True, lw=.5, bins=kwargs.get("bins"))
            res, _ = np.histogram(samples, bins=bins, density=True)
            hists.append(res)
        # ax.hist(np.mean(hists, axis=0), label='mean sampled error', **kwargs)
        binc = (bins[1:] + bins[:-1]) / 2
        height = np.mean(hists, axis=0)
        ax.bar(binc, height, width=(binc[1]-binc[0]), label='mean sampled error', edgecolor='k', alpha=0.5, color="tab:orange")
        title = format_key(key)
        ax.set_title(title)
        ax.legend()
    f.tight_layout()
    return f, axes

def check_trend_error_hist(diags, ids=None, fields=["rad"], smoothings=[15], high_passes=[None], variables=["tws", "AIS", "GrIS", "glac"], ni=None, nj=2, axes=None, **kwargs):
    items = [(f, k, s, h) for f in fields for k in variables for s in smoothings for h in high_passes]
    if axes is None:
        n = len(items)
        ni = ni or int(np.ceil(n / nj))
        f, axes = plt.subplots(ni, nj, figsize=(10, 4*ni))
    else:
        f = axes.flat[0].get_figure()
    kwargs.setdefault('bins', 20)
    kwargs.setdefault('edgecolor', 'k')
    kwargs.setdefault('density', True)
    kwargs.setdefault('alpha', 0.5)
    for key, ax in zip(items, axes.flat):
        rms = diags[key]['rms'].values
        ax.hist(rms, label='derived fingerprint- \nand GPS-sampling related RMS error', **kwargs)
        title = format_key(key)
        ax.set_title(title)
        ax.legend()
    f.tight_layout()
    return f, axes

def check_trend_error_scatter(diags, ids=None, fields=["rad"], smoothings=[None, 15], high_passes=[None, 15], variables=["tws"], ni=None, nj=2, axes=None, **kwargs):
    items = [(f, k, s, h) for f in fields for k in variables for s in smoothings for h in high_passes]
    if axes is None:
        n = len(items)
        ni = ni or int(np.ceil(n / nj))
        f, axes = plt.subplots(ni, nj, figsize=(10, 4*ni))
    else:
        f = axes.flat[0].get_figure()
    # kwargs.setdefault('bins', 20)
    # kwargs.setdefault('edgecolor', 'k')
    # kwargs.setdefault('density', True)
    # kwargs.setdefault('alpha', 0.5)
    for key, ax in zip(items, axes.flat):
        trend_errors = diags[key]['model_trends'] - diags[key]['obs_trends']
        ax.scatter(diags[key]['obs_trends'], diags[key]['model_trends'], c=np.abs(trend_errors), **kwargs)
        title = format_key(key)
        ax.set_title(title)
        ax.set_aspect('equal')

    xmin = min(ax.get_xlim()[0] for ax in axes.flat)
    xmax = max(ax.get_xlim()[1] for ax in axes.flat)

    for ax in axes.flat:
        # ax.set_ylim(ax.get_ylim())
        ax.plot([xmin, xmax], [xmin, xmax], "k--")

    f.tight_layout()
    return f, axes


def _check_covariance(diags, ni=None, nj=2, axes=None):

    psmsl_ids = next(iter(diags.values())).station.values
    i = argsort_stations_by_id(psmsl_ids)

    items = list(diags)
    # f, axes = plt.subplots(3, 2, figsize=(10, 12))
    if axes is None:
        n = len(items)
        ni = ni or int(np.ceil(n / nj))
        f, axes = plt.subplots(ni, nj, figsize=(10, 4*ni))
    else:
        f = axes.flat[0].get_figure()

    for (ax, key) in zip(axes.flat,
                            list(diags)):
        ds = diags[key]
        cov = ds["cov_rms"].values
        # cov = ds["cov"].values
        # h = ax.imshow(cov[i][:, i], cmap="RdBu_r")
        # h = ax.imshow(cov[i][:, i], cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        h = ax.imshow(cov[i][:, i], cmap="RdBu_r", vmin=-0.1, vmax=0.1)
        # h = ax.imshow(cov[i][:, i], cmap="RdBu_r", vmin=-0.05, vmax=0.05)
        # h.cmap.set_over("darkred")
        h.cmap.set_under("purple")
        plt.colorbar(h, ax=ax, label="(mm/yr)^2", extend="both")
        # diags[key].plot(ax=ax)
        ax.set_title(format_key(key))
    axes.flat[-1].axis("off")
    f.tight_layout()

def check_covariance(data, fields=["rad"], smoothings=[15], high_passes=[None], variables=["tws", "AIS", "GrIS", "glac"], ni=None, nj=2, axes=None, **kwargs):
    items = [(f, k, s, h) for f in fields for k in variables for s in smoothings for h in high_passes]
    return _check_covariance({k: data[k] for k in items}, ni=ni, nj=nj, axes=axes)