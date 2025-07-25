"""Code to make coastline figures and stats
"""
import numpy as np
import fiona
import shapely
import shapely.geometry as shg
import shapely.ops
import matplotlib.pyplot as plt

from sealevelbayes.datasets.naturalearth import NE_DATA


def load_coastlines(res="110m"):

    coll = list(fiona.open(NE_DATA/f"ne_{res}_coastline/ne_{res}_coastline.shp"))
    geomcoll = shg.GeometryCollection([shg.shape(s["geometry"]) for s in coll])

    return geomcoll


def project_coords_on_coastline(coords, coastline):
    """Project the input `coords` onto the coastline

    Parameters
    ----------
    coords: n x 2 array of lon lat pairs: locations to be projected
    coastline: N x 2 array of lon lat pair: continuous coastline

    Returns
    -------
    matching_index: int, projection index on the coastline
    matching_dist: distance to the projected point
    """

    n = len(coords)
    matching_index = np.zeros(n, dtype=int)
    matching_dist = np.empty(n)
    # matching_coastline2 = []

    for i, p in enumerate(coords):
        lon, lat = p
        if lon > 180: lon -= 180
        clons, clats = coastline.T
        dists2 = ((lon-clons)*np.cos(np.deg2rad(lat)))**2 + (clats - lat)**2
        idist = np.argmin(dists2)
        matching_index[i] = idist
        # matching_coord[i] = full_coast[idist]
        matching_dist[i] = dists2[idist]**.5*111

    return matching_index, matching_dist


# # def filter_coastlines(geoms, minlength=50):
# matching_coastline[matching_dist < 50]
#     full_coast = np.concatenate([geom.coords[:] for geom in geomcoll.geoms])
#     geom_index = np.concatenate([[i]*len(geom.coords) for i, geom in enumerate(geomcoll.geoms)])

#     matching_index, matching_dist = project_coords_on_coastline(coords, full_coast)
#     matching_coord = full_coast[matching_index]
#     matching_coastline = geom_index[matching_index]

def plot_coastline_map_overview_per_length(geoms, ax=None):
    """Plot the requested geometries, indicating the start of the line with a circle
    """

    lengths = np.array([geom.length for geom in geoms])

    if ax is None:
        f, ax = plt.subplots(1,1)

    count = -1
    for i, (geom, length, index) in enumerate(sorted(zip(geoms, lengths, np.arange(lengths.size)), key=lambda t: t[1], reverse=True)):
    # for i, (geom, length) in enumerate(zip(geomcoll.geoms, lengths)):
        # if index not in matching_coastline[matching_dist < 50]:
        #     continue
        # if i > 19:
            # continue

        count += 1

        lon, lat = np.asarray(geom.coords[:]).T
        dist = 111*(((np.diff(lon)*np.cos(np.deg2rad((lat[1:]+lat[:-1])/2)))**2 + np.diff(lat)**2)**.5).sum()
        l, = ax.plot(lon, lat, label=f"({i}) {dist:.0f} km", color=plt.cm.tab20_r(count))
        ax.plot(lon[0], lat[0], 'o', markerfacecolor='none', color=l.get_color())

    ax.legend(fontsize='xx-small')
    ax.set_aspect(1)


def define_northamericas(geomcoll):
    northamerica = sorted(geomcoll.geoms, key=lambda geom: geom.length, reverse=True)[3]
    test1 = northamerica.interpolate(180)
    test2 = northamerica.interpolate(410)
    # print("Cut points", test1.coords[:], test2.coords[:])
    northamerica_pacific = shapely.ops.split(northamerica, test1.buffer(.1)).geoms[0]
    northamerica_atlantic = shapely.ops.split(northamerica, test2.buffer(.1)).geoms[-1]
    return northamerica_atlantic, northamerica_pacific


def define_europe_coastlines(geomcoll):
    europe = sorted(geomcoll.geoms, key=lambda geom: geom.length, reverse=True)[0]
    test1 = europe.interpolate(165)
    test2 = europe.interpolate(222)
    test3 = europe.interpolate(390)
    europe = shapely.ops.split(europe, test1.buffer(.1)).geoms[2]
    northmediterranean = shapely.ops.split(europe, test2.buffer(.1)).geoms[0]
    westerneurope = shapely.ops.split(europe, test2.buffer(.1)).geoms[2]
    westerneurope = shapely.ops.split(westerneurope, test3.buffer(.1)).geoms[0]
    return northmediterranean, westerneurope


def define_southeastasia(geomcoll):
    asia = sorted(geomcoll.geoms, key=lambda geom: geom.length, reverse=True)[1]
    test1 = asia.interpolate(115)
    test2 = asia.interpolate(305)
    southeastasia = shapely.ops.split(asia, test1.buffer(.1)).geoms[2]
    southeastasia = shapely.ops.split(southeastasia, test2.buffer(.1)).geoms[0]
    return southeastasia


def define_southamericas(geomcoll):
    southamerica_pacific = sorted(geomcoll.geoms, key=lambda geom: geom.length, reverse=True)[9]
    southamerica_pacific = shg.LineString(southamerica_pacific.coords[::-1]) # south to north to avoid a duplicate
    southamerica_atlantic = sorted(geomcoll.geoms, key=lambda geom: geom.length, reverse=True)[6]
    return southamerica_atlantic, southamerica_pacific


def define_greatbritain(geomcoll):
    return sorted(geomcoll.geoms, key=lambda geom: geom.length, reverse=True)[14]

def define_australia(geomcoll):
    return sorted(geomcoll.geoms, key=lambda geom: geom.length, reverse=True)[5]


def define_coastlines(geomcoll, resample=False, dx=50):
    """Define world coastlines without resampling

    Parameters
    ----------
    geomcoll: Geometry collection from Natural Earth
    resample: bool, False by default
        if True, resample with spatial distance `dx`
    dx: float, 50 (km) by default
        resampling step in km

    Returns
    -------
    coastlines: shapely's GeometryCollection of world's coastlines
    names: list of coastline names matching coastlines's `geoms` attribute
    """

    northamerica_atlantic, northamerica_pacific = define_northamericas(geomcoll)
    northmediterranean, westerneurope = define_europe_coastlines(geomcoll)
    southeastasia = define_southeastasia(geomcoll)
    southamerica_atlantic, southamerica_pacific = define_southamericas(geomcoll)
    greatbritain = define_greatbritain(geomcoll)
    australia = define_australia(geomcoll)

    all_geoms = [northamerica_pacific, northamerica_atlantic, northmediterranean, westerneurope, southeastasia, southamerica_atlantic, southamerica_pacific, greatbritain, australia]
    names = ["North America Pacific", "North America Atlantic", "Mediterranean North", "Western Europe", "Southeast Asia", "South America (Atlantic)", "South America (Pacific)", "Great Britain", "Australia"]

    # resample with a continuous spatial step
    if resample:
        all_geoms = [resample_geom(geom, dx) for geom in all_geoms]

    return shg.GeometryCollection(all_geoms), names


def running_distance(coords, degrees=False):
    lon, lat = np.asarray(coords).T
    if degrees:
        dist = ((np.diff(lon)**2 + np.diff(lat)**2)**.5).cumsum()
    else:
        dist = 111*(((np.diff(lon)*np.cos(np.deg2rad((lat[1:]+lat[:-1])/2)))**2 + np.diff(lat)**2)**.5).cumsum()

    return np.concatenate([[0], dist])


def resample_geom(geom, dx=50):
    lon, lat = np.asarray(geom.coords[:]).T
    true_distance = running_distance(geom.coords[:])
    shg_distance = running_distance(geom.coords[:], degrees=True)
    resampled = np.arange(0, true_distance[-1], dx)
    resampled_shg = np.interp(resampled, true_distance, shg_distance)
    return shg.LineString([geom.interpolate(x) for x in resampled_shg])