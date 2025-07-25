from sealevelbayes.datasets.figures import xcolors, xlabels

def plot_global(global_slr, experiments=['C1', 'C6'], ax=None, label=None, scale=1, diff=True):

    if ax is None:
        plt.figure()
        ax.gca()

    for experiment in experiments:
        x = global_slr.sel(experiment=experiment)*scale
        c = xcolors.get(experiment)
        ax.fill_between(global_slr.year, *x.quantile([.05, .95], dim='sample'), alpha=0.3, color=c)
        ax.plot(global_slr.year, x.median(dim='sample'), color=c, label=xlabels.get(experiment, experiment))
        
    if diff:
        add_global_diff(ax, global_slr, experiments[1], experiments[0], scale=scale)

    ax.grid()
    ax.legend(loc='upper left')
    ax.set_xlabel('Year')
    ax.set_ylabel(label)
    
    
def add_global_diff(ax, global_slr, x1, x2, scale=1):
    diff = (global_slr.sel(experiment=x1) - global_slr.sel(experiment=x2))*scale
    c = 'black'
    ax.plot(global_slr.year, diff.quantile([.05, .95], dim='sample').T, color=c, ls=':')
    ax.plot(global_slr.year, diff.median(dim='sample'), color=c, ls='--', label='difference')

    

def make_bottom_proj_panel(ax3, stations_js, experiments=None, extra_experiments=[], diff=True):
    if experiments is None:
        experiments = stations_js[0]['experiments']
    barplot = BarPlot(stations_js)
    barplot.plot_proj(experiments, ax=ax3, uncertainty=True)

    for experiment in extra_experiments:
        recs = [r for js in barplot.stations_js for r in js['records'] if r['experiment'] == experiment and r['field'] == 'rsl' and r['source'] == 'total' and r['diag'] == "proj2100"]
        ax3.plot(barplot.x, [r['median']/10 for r in recs], '.', color=xcolors.get(experiment), markersize=1, label=xlabels.get(experiment, experiment))
        
    if diff:
        experiment = "difference"
        xcolors[experiment] = "black"
#         recs = [r for js in barplot.stations_js for r in js['records'] if r['experiment'] == experiment and r['field'] == 'rsl' and r['source'] == 'total' and r['diag'] == "proj2100"]
#         ax3.plot(barplot.x, [r['median']/10 for r in recs], '.', color=xcolors.get(experiment), markersize=1, label=xlabels.get(experiment, experiment))                
        barplot.plot_proj(['difference'], ax=ax3, uncertainty=True)
    

    ax3.get_legend().remove()
    barplot.set_axis_layout(ax3)
    ax3.set_ylim(-100, 200)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')
    # ax3.set_xticks([], [])
    # ax3.set_xticks(barplot.counts, [f"{lab} ({c})" for lab, c in zip(barplot.basins, barplot.bcounts)], horizontalalignment='right')
    # ax3.set_xticks(np.array(barplot.counts)-0.5, barplot.counts, fontsize='small')
    # ax3.set_xticks(np.array(barplot.counts)-0.5, [])
    barplot.set_basins_labels(ax3)
    ax3.set_title('Local, relative sea-level rise projections')
    
    
def make_proj
