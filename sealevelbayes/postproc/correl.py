import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib as mpl
import matplotlib.pyplot as plt
from sealevelbayes.postproc.gmslfigure import TABLE_FIG_NAMES
from sealevelbayes.postproc.colors import xcolors, xlabels, sourcecolors, sourcelabels, sourcelabelslocal, basincolors, diaglabels
from sealevelbayes.datasets.ar6.tables import ar6_table_9_5, ar6_table_9_8_medium_confidence, ar6_table_9_5_quantiles

def add_kde_contours(axes, dataset, color, kde=True, levels=5, bins=10, grid=100, cmap=None):

    qsets = np.empty_like(axes, dtype=object)

    for j, col in enumerate(list(dataset.columns)):
        for i, row in enumerate(list(dataset.columns)):
            if i == j:
                continue
            if i > j:
                continue
            ax = axes[i, j]
            # ax2 = axes[j, i]
            x = dataset[col].values
            y = dataset[row].values

            # Create a grid of points
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()

            if kde:
                # Fit the KDE
                kde_fn = gaussian_kde([x, y])

                xgrid, ygrid = np.meshgrid(np.linspace(xmin, xmax, grid), np.linspace(ymin, ymax, grid))
                positions = np.vstack([xgrid.ravel(), ygrid.ravel()])

                # Evaluate the KDE on the grid
                z = kde_fn(positions).reshape(xgrid.shape)

            else:
                # This makes edgy contours
                H, xe, ye = np.histogram2d(x, y, bins=bins, density=True)
                xgrid, ygrid = (xe[1:]+xe[:-1])/2, (ye[1:]+ye[:-1])/2
                z = H.T
                # # https://stackoverflow.com/a/12311139/2192272
                # import scipy.ndimage
                # from scipy.ndimage.filters import gaussian_filter
                # z = gaussian_filter(z, [ygrid.ptp(), xgrid.ptp()])
                # z = scipy.ndimage.zoom(z, 3)
                # xgrid = scipy.ndimage.zoom(xgrid, 3)
                # ygrid = scipy.ndimage.zoom(ygrid, 3)

            levels_ = levels[i, j] if isinstance(levels, np.ndarray) else levels
            if hasattr(levels_, "levels"): # QuadContourSet
                levels_ = levels_.levels
            if isinstance(levels_, int):
                import matplotlib as mpl
                levels_ = mpl.ticker.MaxNLocator(nbins=levels_, prune="lower").tick_values(z.min(), z.max())

            # Plot the contours
            qsets[i, j] = qset_ = ax.contour(xgrid, ygrid, z, levels=levels_, colors=[color], linewidths=0.5)
            levels_ = qset_.levels
            qsets[j, i] = axes[j, i].contour(ygrid.T, xgrid.T, z.T, levels=levels_, colors=[color], linewidths=0.5)

            if cmap is not None:
                qset_ = ax.contourf(xgrid, ygrid, z, levels=levels_, cmap=cmap, zorder=-1, alpha=0.5)
                levels_ = qset_.levels
                axes[j, i].contourf(ygrid.T, xgrid.T, z.T, levels=levels_, cmap=cmap, zorder=-1, alpha=0.5)

            # ax2.contour(ygrid, xgrid, z.T, levels=5, colors=color, linewidths=0.5)

    return qsets

def add_kde_diag(ax, values, **kwargs):
    ax.hist(values, **kwargs)

def bold(s):
    return f"$\\bf{{{s}}}$"

def _make_dataset(trace, var_names, experiment=None, rename=None, glacier_region=None, year=None):
    if rename is None:
        rename = {c:c.split("_")[2] for c in var_names}

    selkw = {}
    if glacier_region is not None:
        selkw["glacier_region"] = glacier_region
    if experiment is not None:
        selkw["experiment"] = experiment
    if year is not None:
        selkw["year"] = year

    def _make_dataframe(trace, experiment, rn=None):
        if hasattr(trace, "posterior"):
            trace = trace.posterior
        ds = trace[var_names].sel(**selkw)
        if year is not None:
            ds = ds.sel(year=year).mean("year")
        return ds.to_dataframe()[var_names].rename((rn or rename) or {}, axis=1).rename(sourcelabels, axis=1)

    if type(experiment) is list:
        # renames = [{c:(bold if 'past20c' in c else lambda s:s)(" ".join(
        renames = [{c:" ".join(
            ([xlabels.get(x, x)] if "proj" in c else [])
            + [diaglabels.get(c.split("_")[0], c.split("_")[0])]
            + [(bold if 'past20c' in c else lambda s:s)(sourcelabels.get(c.split("_")[2]))]
          ) for c in var_names} for x in experiment]
        dataset = pd.concat([_make_dataframe(trace, x, rn).iloc[:, [i]] for i, (x, rn) in enumerate(zip(experiment, renames))], axis=1)

    else:
        dataset = _make_dataframe(trace, experiment)

    return dataset

def get_constraint(var_name, experiment="ssp585", glacier_region=None, constant_data=None):
    s = 1.64 # 90% confidence instead of 66%

    if var_name == "glacier_sea_level_rate":
        if constant_data is None:
            return
        assert glacier_region is not None, "Need to specify glacier region"
        mean = constant_data["glacier_data_rate2000"].sel(glacier_data_region=glacier_region).values.item()
        sd = constant_data["glacier_data_rate2000_sd"].sel(glacier_data_region=glacier_region).values.item()
        return mean - s*sd, mean, mean + s*sd

    try:
        diag, field, source = var_name.split("_")
    except:
        return None
    if source in ar6_table_9_5:
        tab = ar6_table_9_5[source]
    else:
        return
    if diag == "past20c":
        return ar6_table_9_5.get(source, {}).get("Δ (mm)", {}).get('1901-1990')
    elif diag == "rate2000":
        return ar6_table_9_5.get(source, {}).get("mm/yr", {}).get('1993-2018')
    elif diag == "proj2100":
        rng = ar6_table_9_8_medium_confidence[experiment].get(source)
        if rng is None: return
        lo, mi, hi = np.array(rng)*1000
        return mi-s*(mi-lo), mi, mi+s*(hi-mi)


def add_constraints_to_scatter(axes, ranges):
    for i, rng in enumerate(ranges):
        for j, rng2 in enumerate(ranges):
            if rng is not None:
                lo, mi, hi = rng
            if rng2 is not None:
                lo2, mi2, hi2 = rng2
            if i == j:
                if rng is None: continue
                axes[i, j].axvline(mi, color="k", linestyle="--", linewidth=1)
                axes[i, j].axvline(lo, color="k", linestyle=":", linewidth=1)
                axes[i, j].axvline(hi, color="k", linestyle=":", linewidth=1)

            elif rng is None and rng2 is None:
                continue

            elif rng is None:
                axes[i, j].axvline(mi2, color="k", linestyle="--", linewidth=1)
                axes[i, j].axvline(lo2, color="k", linestyle=":", linewidth=1)
                axes[i, j].axvline(hi2, color="k", linestyle=":", linewidth=1)

            elif rng2 is None:
                axes[i, j].axhline(mi, color="k", linestyle="--", linewidth=1)
                axes[i, j].axhline(lo, color="k", linestyle=":", linewidth=1)
                axes[i, j].axhline(hi, color="k", linestyle=":", linewidth=1)

            else:
                ellipse = mpl.patches.Ellipse((mi2, mi), width=hi2-lo2, height=hi-lo, edgecolor="k", facecolor="none", linestyle=":")
                axes[i, j].add_patch(ellipse)
                axes[i, j].scatter(mi2, mi, color="k", marker="x")


def scatter_matrix2(traces, var_names, suptitle, experiment="ssp585_mu", rename=None,
                    labels=None, colors=None, cmaps=None, glacier_region=None, year=None, add_contours=True, **kw):

    datasets = [_make_dataset(trace, var_names, experiment=experiment, rename=rename, glacier_region=glacier_region, year=year) if trace is not None else None for trace in traces]
    colors = colors or [f"C{j}" for j in range(len(datasets))]
    cmaps = cmaps or [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens, plt.cm.Purples, plt.cm.Reds]
    labels = labels or [f"" for j in range(len(datasets))]

    axes = None
    for j, (dataset, label, color) in enumerate(zip(datasets, labels, colors)):
        if dataset is None:
            continue
        axes = pd.plotting.scatter_matrix(dataset, alpha=0.5, figsize=(10, 10), color=color, diagonal='kde', s=2,
                                density_kwds={'label': label, 'color': color}, ax=axes)

    fig = axes[0,0].get_figure()
    plt.suptitle(suptitle, y=.91)

    if add_contours:
        for j, (dataset, label, color, cmap) in enumerate(zip(datasets, labels, colors, cmaps)):
            if dataset is None:
                continue
            add_kde_contours(axes, dataset.dropna(), color=color, cmap=cmap,
                            #  levels=qsets,
                            **kw)

    axes[0, 0].legend(fontsize='x-small')

    return fig, axes


def correl_matrix2(trace, var_names, suptitle, experiment="ssp585_mu", rename=None, ax=None, cbar=True,
                    trace_alt=None,
                    figsize=(5, 5),
                    cov=False,
                    labelsize=None,
                    **kw):

    dataset = _make_dataset(trace, var_names, experiment=experiment, rename=rename)

    if trace_alt is not None:
        dataset2 = _make_dataset(trace_alt, var_names, experiment=experiment, rename=rename)

    else:
        dataset2 = dataset
        dataset = None

    if ax is not None:
        f = ax.get_figure()
    else:
        f, ax = plt.subplots(1, 1, figsize=figsize)

    i, j = np.meshgrid(range(dataset2.shape[1]), range(dataset2.shape[1]))

    # cmap = plt.cm.coolwarm
    cmap = plt.cm.RdBu_r
    cmap.set_bad(color='black')
    cmap.set_over(color='black')


    if cov:
        corr = dataset2.cov().values
        vmin, vmax = corr.min(), corr.max()
        if dataset is not None:
            corr[i>j] = dataset.cov().values[i>j]
            vmin, vmax = corr.min(), corr.max()
            corr[i==j] = vmax + 1

    else:
        corr = dataset2.corr().values
        if dataset is not None:
            corr[i>j] = dataset.corr().values[i>j]
            corr[i==j] = 1.1
        vmin, vmax = -1, 1

    h = ax.pcolor(corr, cmap=cmap, vmin=vmin, vmax=vmax)
    # h.cmap.set_bad("black")
    ax.set_xticks(np.arange(dataset2.shape[1])+0.5)
    ax.set_xticklabels(dataset2.columns, rotation=45, horizontalalignment="right", fontsize=labelsize)
    ax.set_yticks(np.arange(dataset2.shape[1])+0.5)
    ax.set_yticklabels(dataset2.columns, fontsize=labelsize)
    ax.set_title(suptitle)

    if cbar:
        cb = f.colorbar(h, ax=ax, label=f"{'Covariance' if cov else 'Correlation'}{' (upper: no local constraint, lower: posterior)' if dataset is not None else ''}")

    return f, ax
