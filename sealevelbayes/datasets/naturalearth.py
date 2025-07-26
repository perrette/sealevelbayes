import itertools

from sealevelbayes.datasets.manager import get_datapath, require_dataset

NE_DATA = get_datapath("naturalearth")

coastline_data = {}

def _init_coast(res="110m"):

    require_dataset(f"naturalearth/ne_{res}_coastline")

    import fiona
    import shapely.geometry as shg

    coll = list(fiona.open(NE_DATA/f"ne_{res}_coastline/ne_{res}_coastline.shp"))
    return shg.GeometryCollection([shg.shape(s["geometry"]) for s in coll])


def get_coast_data(res="110m"):
    if res not in coastline_data:
        coastline_data[res] = _init_coast(res)
    return coastline_data[res]

land_data = {}

def _init_land(res='110m'):

    import fiona
    import shapely.geometry as shg

    require_dataset(f"naturalearth/ne_{res}_land")

    coll = list(fiona.open(NE_DATA/f"ne_{res}_land/ne_{res}_land.shp"))
    return shg.GeometryCollection([shg.shape(s["geometry"]) for s in coll])

def get_land_data(res="50m"):
    if res not in land_data:
        land_data[res] = _init_land(res)
    return land_data[res]


def add_coast(ax=None, color='k', linewidth='.5', lon0=None, shift_lon=0, res='50m', bbox=None, **kw):
    import numpy as np
    import matplotlib.pyplot as plt
    import shapely.affinity
    import shapely.geometry as shg

    gcoll = get_coast_data(res=res)

    if ax is None:
        ax = plt.gca()

    def iterator(gcoll):
        for s in gcoll.geoms:
            if shift_lon:
                s = shapely.affinity.translate(s, shift_lon, 0)
            if bbox is not None:
                l, r, b, t = bbox
                bbox_geom = shg.box(l, b, r, t)
                if not s.intersects(bbox_geom):
                    continue
                s = s.intersection(bbox_geom)
                if hasattr(s, "geoms"):
                    for s_ in s.geoms:
                        yield s_
                else:
                    yield s
            else:
                yield s

    for s in iterator(gcoll):

        if lon0 is None:
            x, y = np.asarray(s.coords).T
            ax.plot(x, y, color=color, linewidth=linewidth)
        else:
            for negative, coords in itertools.groupby(s.coords, key=lambda c: c[0] < lon0):
                x, y = np.array(list(coords)).T
                if negative:
                    ax.plot(x+360, y, color=color, linewidth=linewidth, **kw)
                else:
                    ax.plot(x, y, color=color, linewidth=linewidth, **kw)


def add_land(ax=None, lon0=None, shift_lon=0, res='50m', domain=None, bbox=None, **kwargs):
    import matplotlib.pyplot as plt
    # from descartes import PolygonPatch
    from shapely.plotting import plot_polygon
    import shapely.affinity
    import shapely.geometry as shg

    kwargs.setdefault("color", "wheat");

    gcoll_land = get_land_data(res)

    if ax is None:
        ax = plt.gca()

    if bbox is not None:
        l, r, b, t = bbox
        domain = shg.Polygon([(l, b), (r, b), (r, t), (l, t)])

    for poly in gcoll_land.geoms:
        if shift_lon:
            poly = shapely.affinity.translate(poly, shift_lon, 0)

        if domain:
            if not domain.intersects(poly):
                continue
            else:
                poly = poly.intersection(domain)

        # ax.add_patch(PolygonPatch(poly, **kwargs))
        plot_polygon(poly, ax=ax, add_points=False, **kwargs)