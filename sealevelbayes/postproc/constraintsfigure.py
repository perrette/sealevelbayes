import arviz
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

from sealevelbayes.postproc.figures import sourcecolors

DEFAULT_COLORS = {
            'gmsl': 'steelblue',
            'rsl': 'darkcyan',
            'gsl': 'darkseagreen',
            'rad': 'brown',
      }


def _transpose(dataset, dims, append=False):
    "partial reordering of a dataset (not all dimensions must be provided here) -- the reordered dimensions are put in front"
    dims = [d for d in dims if d in dataset.dims]
    head, tail = ([], dims) if append else (dims, [])
    return dataset.transpose(*head + [d for d in dataset.dims if d not in dims] + tail)


def as_posterior(trace_k):
    post = arviz.extract(trace_k.posterior)
    # post = post.rename_dims({"year_output": "year"})
#     post = post.rename_dims(year_output="year")
    if "year" in trace_k.posterior.coords:
        years = trace_k.posterior.coords['year']
        post = _transpose(post, dims=["year", "station"], append=True) # make sure we keep the year x station shape
    else:
        years = np.arange(post.year_output[0], post.year_output[-1]+1)
        post = post.interp({"year_output": years})
        post = _transpose(post, dims=["year_output", "station"], append=True) # make sure we keep the year x station shape
        for k in post:
            if "year_output" in post[k].dims:
                post[k] = post[k].swap_dims(year_output="year")
    return post


class ConstraintPlot:
    def __init__(self, trace_k, post=None, gps_data=None):
        self.trace_k = trace_k
        self.gps_data = gps_data
        self.post = as_posterior(trace_k) if post is None else post
        self.samples = np.random.default_rng(seed=847).integers(self.post.sample.size, size=500)
        self.scale = 1/10

    def get_tidegauge(self):
        tg_values = np.ma.array(self.trace_k.constant_data["tidegauge_mm"].values, mask=self.trace_k.constant_data["tidegauge_mask"].values)
        tg_years = self.trace_k.constant_data["tidegauge_year"].values
        return tg_years[~tg_values.mask], tg_values.data[~tg_values.mask] * self.scale

    def get_satellite(self):
        tg_values = self.trace_k.constant_data["satellite_mm"].values
        tg_years = self.trace_k.constant_data["satellite_year"].values
        return tg_years, tg_values * self.scale

    def get_gps(self):
        if self.gps_data is not None:
            return self.gps_data.index.values, (self.gps_data - self.gps_data.mean()).values
        gps_years = np.arange(2000, 2020+1)
        return gps_years, gps_years + np.nan

    def get_model_data(self, name):
        a = self.post[f"change_{name}_total"]
        slr = a.isel(sample=self.samples).squeeze()
        slr -= slr.sel(year=slice(1995, 2014)).mean("year")
        return slr * self.scale

    def get_model_data_dict(self):
        return {
            "gmsl": self.get_model_data("global"),
            "rsl": self.get_model_data("rsl"),
            "gsl": self.get_model_data("gsl"),
            "rad": self.get_model_data("rad"),
            }

    def obs_timeseries_dict(self):
        return {
            "tidegauge": self.get_tidegauge(),
            "satellite": self.get_satellite(),
            "gps": self.get_gps(),
        }

    def get_obs_trend(self, obs_name):
        return (self.trace_k.constant_data[f"{obs_name}_mu"].values*self.scale,
         self.trace_k.constant_data[f"{obs_name}_sd"].values*self.scale)

    def plot(self, fields=None, show_trend=True, **kwargs):
        obs_ts = self.obs_timeseries_dict()
        return _plot_constraints(
            self.get_model_data_dict(), obs_ts, {k:self.get_obs_trend(k) for k in (obs_ts if show_trend else ['gps'])}, show_trend=show_trend, fields=fields, **kwargs)


    def plot_global(self, **kwargs):
        w, h = plt.rcParams["figure.figsize"]
        kwargs.setdefault("figsize", (w, h*.8))
        kwargs.setdefault("title", "Constraints from global AR6 numbers")
        return self.plot(fields=['gmsl'], **kwargs)


    def plot_local(self, **kwargs):
        kwargs.setdefault("title", "Constraints from local observations")
        return self.plot(fields=['rsl', 'gsl', 'rad'], local_offset=0, **kwargs)


# def _get_offset(a):
#     return a.sel(year=slice(1995, 2014)).median('sample').values.mean()

# https://github.com/matplotlib/matplotlib/issues/697#issuecomment-3859591
from matplotlib import transforms

def rainbow_text(ax, x,y,ls,lc,**kw):
    """
    Take a list of strings ``ls`` and colors ``lc`` and place them next to each
    other, with text ls[i] being shown in color lc[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.
    """
    t = ax.transData
    # fig = plt.gcf()
    # fig = ax.get_figure()
    # plt.show()

    #horizontal version
    for i, (s,c) in enumerate(zip(ls,lc)):
        text = ax.text(x,y-15*i+10," "+s+" ",color=c, transform=t, **kw)


def _plot_constraints(model_data, obs_timeseries, obs_trends={},
    fields = None,
    offsets = None,
    title = None,
    local_offset=100, local_spacing=75,
    figsize=None,
    show_trend=True,
    colors=DEFAULT_COLORS,
      labels={
            # "gmsl": r'($\bf{a}$) global thermal, glaciers, GIS, AIS',
            # "gmsl": r'($\bf{a}$) global thermal, glaciers, Greenland, Antarctica',
            # "gmsl": '($\\bf{a}$) global mean contributions:\n    Steric, Glaciers, Greenland, Antarctica',
            "gmsl": '($\\bf{A}$) Global mean contributions\n    Steric, Glaciers, Greenland, Antarctica, Land Water'.split('\n'),
        #     "gmsl": 'global x 5 (AR6 numbers)',
            # "rsl": '($\\bf{b}$) tide gauges\n   <color="black">local relative sea level</color>',
            "rsl": '($\\bf{B}$) Tide gauges\n   Local relative sea level'.split('\n'),
            "gsl": '($\\bf{C}$) Satellite altimetry\n    Local geocentric sea level'.split('\n'),
            "rad": '($\\bf{D}$) GPS\n     Local land subsidence'.split('\n'),
  }, ax=None):

    offsets = offsets or {
        "gmsl": 0,
        "rsl": -local_offset,
        "gsl": -local_offset - local_spacing,
        "rad": -local_offset - 2*local_spacing,
    }

    if fields is None:
        fields = list(offsets.keys())
    else:
        offsets = {k: offsets.get(k,0) for k in fields}

    data = { k: model_data[k]*(-1 if k == "rad" else 1) + offsets[k] for k in offsets}

    obs_mapping = {'satellite': 'gsl', 'tidegauge': 'rsl', 'gps': 'rad'}
    field_to_obs = {field: [obs_name] for obs_name, field in obs_mapping.items()}

    # offsets = { k: _get_offset(v) for k, v in data.items()}


    def plot_one(name):
        values = data[name].values
        years = data[name].year

        color = (0, 0, 0, 0.8)
        # label = labels[name]
        if values.ndim > 2:
             values = values.reshape(np.prod(values.shape[:-1]), values.shape[-1])
        l = ax.plot(years, values.T,
             alpha=1e-1/10, color=color, linewidth=0.5);

        # ax.plot(a.year, a.median("sample").values, color=color, linewidth=.5)


    def plot_trend(name, obs_name, y1, y2, linewidth=.5, linestyle='-', show_range=False, color=None, **kw):
        field = obs_mapping[obs_name]

        if obs_name not in obs_trends:
            return
        m, sd = obs_trends[obs_name]

        lo = m + sd*norm.ppf(.05)
        hi = m + sd*norm.ppf(.95)

        if obs_name == 'gps':
            m, lo, hi = -m, -lo, -hi
        years = np.arange(y1, y2+1)
        x = years - obs_years.mean()

        color = color or colors[field]

#         offset = offsets[field]
        offset = data[name].median('sample').reindex(year=obs_years).mean().values - (x*m).mean()
#         offset = data[name].median("sample").reindex(year=obs_timeseries[obs_name][0]).mean().values

        l, = plt.plot(years, x*m + offset, color=color, linewidth=linewidth, linestyle=linestyle, **kw)
        if show_range:
            plt.plot(years, x*hi + offset, color=color, linewidth=linewidth, linestyle=':', **kw)
            plt.plot(years, x*lo + offset, color=color, linewidth=linewidth, linestyle=':', **kw)
        l.set_label(labels[field])


    if ax is None:
        w, h = plt.rcParams["figure.figsize"]
        fig = plt.figure(figsize=figsize or (w, h*1.5*len(fields)/4))
        ax = plt.gca()
    else:
        fig = ax.get_figure()
        plt.sca(ax)

    # Relative SLR
    for field in fields:

        plot_one(field)

        for obs_name in field_to_obs.get(field, []):

            obs_years, obs_values = obs_timeseries[obs_name]
            if obs_name == "gps":
                obs_values = -obs_values # shown in SLR convention
            offset = obs_values.mean() - data[field].median('sample').reindex(year=obs_years).mean().values
            # ax.plot(obs_years, obs_values - offset, color=colors[field], linewidth=1 if show_trend else 2, zorder=999)
            ax.plot(obs_years, obs_values - offset, color=colors[field], linewidth=1, zorder=999)

            if show_trend or obs_name == "gps":
                y1, y2 = obs_years[[0, -1]]
                # plot_trend(field, obs_name, y1, y2, linewidth=(2 if obs_name == "gps" else .5), zorder=1001)
                plot_trend(field, obs_name, y1, y2, linewidth=.5, zorder=1001)
                # plot_trend(field, obs_name, y1, y2, linewidth=2, zorder=998)

    # add offsets
    yticks = []
    yticklabels = []
    for k, v in offsets.items():
        ax.axhline(v, color='k', linestyle='--', linewidth=0.5)

        # yticks
        for dy in [100, 50, 25, 0]:
            y = v + dy
            if not yticks or y < np.min(yticks)-1:
                yticks.append(y)
                yticklabels.append(dy)

        if type(labels[k]) is list:
            rainbow_text(ax, 1905, v+(20 if k == 'gmsl' else 10), labels[k], [colors[k], "black"], bbox=dict(edgecolor='none', facecolor='white', alpha=0.7))
        else:
            ax.text(1905, v+(20 if k == 'gmsl' else 10), labels[k], color=colors[k], bbox=dict(edgecolor='none', facecolor='white', alpha=0.7))
    #     ax.text(1905, v+25, labels[k], bbox=dict(edgecolor=colors[k], facecolor='white', alpha=0.7))
    #     ax.text(1905, v+25, labels[k], bbox=dict(edgecolor='none', facecolor=colors[k], alpha=0.3))

    ax.set_yticks(yticks, yticklabels, fontsize='small')
    plt.xticks(fontsize='small')

    # lo, hi = -50, 200
    plt.ylabel("Sea level (cm)")
    # plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
    plt.xlim(1900, 2100)
    plt.ylim(ymax=125)
    # plt.ylim(lo, hi)
    # plt.tight_layout()
    ax.yaxis.grid()
    plt.xlabel('Year')
    plt.title(title or 'Constraints from global AR6 numbers and local observations')

    if "gmsl" in fields:
        # show Fred total
        import xarray as xa
        from sealevelbayes.datasets.frederikse2020 import root
        with xa.open_dataset(root.joinpath("GMSL_ensembles.nc")) as ds:
            # total = ds["sum"].median('likelihood').load()
            total = (ds["Steric"] + ds["Barystatic"])
            total = (total - total.sel(time=slice(1995, 2014)).mean("time")).median('likelihood').load()
        l, = plt.plot(total["time"], total.values/10, color=colors.get('gmsl'), linewidth=1, zorder=999)

        # plt.fill_between([1900, 1990], [-15, -15], [0, 0], edgecolor=colors['gmsl'], facecolor='none')
        # plt.fill_between([1993, 2018], [-5, -5], [10, 10], edgecolor=colors['gmsl'], facecolor='none')
        # plt.fill_between([2020, 2100], [100, 100], [0, 0], edgecolor=colors['gmsl'], facecolor='none',zorder=10, linestyle='--')
        # plt.fill_between([2020, 2100], [100, 100], [0, 0], edgecolor=colors['gmsl'], facecolor='none',zorder=10, linestyle='--')
        ax.axvline(2019, color='black', linestyle='--', linewidth=.5)
        ax.axhline(0, color=colors['gmsl'], linestyle='--', linewidth=1)
        # Plot an arrow from the 0 line to 100 in 2100
        ax.annotate('', xy=(2100, 100), xytext=(2100, 0),
                arrowprops=dict(facecolor=colors['gmsl'], edgecolor='none', width=2, headwidth=6, headlength=10))

        hline = 4
        bbox = dict(facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round")
        kw = dict(fontsize="x-small", horizontalalignment="center", color=colors.get("gmsl"), verticalalignment="top", bbox=bbox)
        # plt.text(1945, -15-hline, "Δ 1901-1990 (mm)\n(AR6 Table 9.5, MM21)", **kw)
        # plt.text(2005, -5-hline, "1993-2018\n(mm/yr)\n(AR6 Table 9.5, MM21)", **kw)
        # plt.text(1950, -8-hline, "1901-2018 (mm/yr)\nFrederikse et al 2020\nMalles and Marzeion 2021", ha="left", **kw)
        # plt.text(2005, -5-hline, "1993-2018\n(mm/yr)\n(AR6 Table 9.5, MM21)", **kw)
        # plt.text(2015, 95, "Frederikse et al 2020\nMalles and Marzeion 2021", ha="right", **kw)
        # plt.text(2025, 95, "SSP1-2.6 & SSP5-8.5 Δ 2100 (mm)\n(AR6 Table 9.8, M20)", ha="left", **kw)
        plt.text(2105, 105, "SSP1-2.6 & SSP5-8.5 Δ 2100\n(AR6 Table 9.8 (ref 1); ref. 52)", ha="left", rotation=90, **kw)

        from matplotlib.transforms import blended_transform_factory
        texttransform = blended_transform_factory(ax.transData, ax.transAxes)
        plt.text(2015, .98, "Past", fontweight="bold", ha="right", transform=texttransform, va="top")
        plt.text(2025, .98, "Future", fontweight="bold", ha="left", transform=texttransform, va="top")

        linewidth = 15
        textbottom = -15-hline-linewidth
        if plt.ylim()[0] > textbottom:
            plt.ylim(ymin=textbottom)

    return fig, ax


    # plt.text(1945, -15-hline, "Δ 1900 (mm)\n(AR6 Table 9.5)", **kw)
    # plt.text(2005, -5-hline, "1993-2018\n(mm/yr)\n(AR6 Table 9.5)", **kw)
    # plt.text(2060, 0-hline, "Δ 2100 (mm)\n(AR6 Table 9.8)", **kw)

    # plt.fill_between([1993, 2020], [-10+offsets['gsl'], -10+offsets['gsl']], [15+offsets['gsl'], 15+offsets['gsl']], edgecolor='black', facecolor='none')
    # plt.fill_between([2000, 2020], [-10+offsets['rad'], -10+offsets['rad']], [15+offsets['rad'], 15+offsets['rad']], edgecolor='black', facecolor='none')
