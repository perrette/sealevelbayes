import numpy as np
from scipy.io import loadmat
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator
import shapely
from shapely.geometry import Polygon, Point

from sealevelbayes.datasets.manager import get_datapath

basinlabels_ml = {
        'Indian Ocean - South Pacific': 'Indian Ocean\nSouth Pacific',
        'Northwest Pacific': 'Northwest\nPacific',
        'Subtropical North Atlantic': 'Subtropical\nNorth Atlantic',
        'Subpolar North Atlantic': 'Subpolar\nNorth Atlantic',
        'Subpolar North Atl. West': 'Subpolar\nNorth Atl. West',
        'Subpolar North Atl. East': 'Subpolar\nNorth Atl. East',
        }


# Define custom regions to correct for coarse-resolution Thompson mask
def _in_bbox(lon, lat, bbox):
    l, r, b, t = bbox
    lon = np.mod(lon, 360)
    return lon >= l and lon <= r and lat >= b and lat <= t

def _is_hawaii(lon, lat):
    t, r = 23.173339, -154.407504+360
    b, l = 17.551605, -161.036605+360
    return _in_bbox(lon, lat, [l, r, b, t])


def _define_panama_north():
    latlongs = [
        [10.574903, -84.473254],
        [8.745277, -82.557025],
        [8.354640, -81.089780],
        [9.294047, -79.470601],
        [8.781417, -78.083103],
        [6.856132, -76.710253],
        [12.393343, -71.366405],
        ]

    coords = [(np.mod(lo, 360), la) for la,lo in latlongs]
    return Polygon(coords)


def _define_mediterranean():
    latlongs = [
        [36.026961, -5.559775],
        [41.215258, -1.796843],
        [47.576240, 5.152215],
        [47.042844, 14.067988],
        [41.151908, 29.074603],
        [38.561130, 39.138485],
        [30.393092, 34.912699],
        [29.822949, 19.327690],
        [33.992542, -4.664769],
        [35.904547, -5.464394],
    ]
    coords = [(lo, la) for la,lo in latlongs]
    poly_across_the_line = Polygon(coords)
    med_left = poly_across_the_line.intersection(_LEFT_HEMISPHERE)
    med_right = poly_across_the_line.intersection(_RIGHT_HEMISPHERE)
    return med_right.union(shapely.transform(med_left, lambda x: x + np.array([360, 0])))

_LEFT_HEMISPHERE = Polygon([(-180, -90), (0, -90), (0, 90), (-180, 90)])
_RIGHT_HEMISPHERE = Polygon([(180, -90), (0, -90), (0, 90), (180, 90)])


_PANAMA_NORTH = _define_panama_north()
_MEDITERRANEAN = _define_mediterranean()

def _is_panama_north(lon, lat):
    return _PANAMA_NORTH.contains(Point(lon, lat))

def _is_mediterranean(lon, lat):
    return _MEDITERRANEAN.contains(Point(lon, lat))

class ThompsonBasins:
    def __init__(self, lon, lat, reg, map):
        lon = np.mod(lon, 360)
        if lon.ndim == 1:
            self.lon = lon
            self.lat = lat
            self.Lon, self.Lat = np.meshgrid(lon, lat)
        else:
            Lon, Lat = lon, lat
            self.lon = Lon[0]
            self.lat = Lat[:, 0]
            self.Lon, self.Lat = Lon, Lat
        self.reg = reg
        self.map = map
        self.map_inv = {v:k for k,v in map.items()}
        mask = ~np.isnan(reg)
        coords0 = np.array([self.Lon[mask], self.Lat[mask]]).T
        self.interp_nearest = NearestNDInterpolator(coords0, reg[mask])


    def plot(self, coords=None, shift_lon=None, alpha=0.2, **kw):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, to_rgba
        from sealevelbayes.postproc.colors import basincolors

        labels = [b for b in basincolors if b in self.map.values()]
        self = self._reorder_colormap(labels)
        colors = [basincolors[lab] for lab in labels]
        a = self.reg.copy()
        Lon = self.Lon.copy()
        if shift_lon is not None:
            Lon[Lon < shift_lon] += 360
            Lon[Lon > 360 + shift_lon] -= 360
            j_sort = np.argsort(Lon[0])
            a = a[:, j_sort]
            Lon = Lon[:, j_sort]

        h = plt.pcolormesh(Lon, self.Lat, a, cmap=ListedColormap(colors), alpha=alpha, **kw)
        h.set_clim(-0.5, -0.5+7)
        cb = plt.colorbar(h)
        cb.ax.tick_params(size=0)
        vmin, vmax = h.get_clim()
        n = len(colors)
        cb.set_ticks(np.arange(len(colors)))
        cb.set_ticklabels([basinlabels_ml.get(lab, lab) for lab in labels])
        return h, cb

    def get_region(self, lon, lat):
        lon = np.mod(lon, 360)
        if _is_hawaii(lon, lat):
            return self.map_inv["Northwest Pacific"]
        if "Mediterranean" in self.map_inv and _is_mediterranean(lon, lat):
            return self.map_inv["Mediterranean"]
        if "Northwest Atlantic" in self.map_inv and _is_panama_north(lon, lat):
            return self.map_inv["Northwest Atlantic"]
        return int(self.interp_nearest((lon, lat)))

    def get_region_label(self, lon, lat):
        lab = self.get_region(lon, lat)
        return self.map[lab]

    @classmethod
    def load(cls, fname=None):
        if fname is None:
            fname = get_datapath("regionMap_Thompson_inv.mat")
        d = loadmat(fname)
        map = {
            1: 'South Atlantic',
            2: 'Indian Ocean - South Pacific',
            3: 'East Pacific',
            4: 'Subpolar North Atlantic',
            5: 'Subtropical North Atlantic',
            6: 'Northwest Pacific'
            }
        return cls(d["Lon"].T[0], d["Lat"].T[:, 0], d["reg"].T, map)


    # modify basins

    def split_atlantic(self, subtropical_north=45, include_mediterranean=True):
        reg = self.reg.copy()
        map = self.map.copy()

        # # Cut-off northern latitudes north of 45N
        region_subpolar = self.map_inv['Subpolar North Atlantic']
        region_subtropical = self.map_inv['Subtropical North Atlantic']
        # reg[(self.reg == region_subtropical) & (self.Lat > subtropical_north)] = region_subpolar

        # # Split into East and West
        # m = reg == region_subpolar
        # reg[m] = np.where((self.Lon[m] < 100 ) | (self.Lon[m] > 320 ), 7, 8)
        # map[7] = "Subpolar North Atl. East"
        # map[8] = "Subpolar North Atl. West"
        # del map[region_subpolar]

        # Split Atlantic into East and West
        m = (reg == region_subpolar) | (reg == region_subtropical)
        region_northwest = region_subtropical
        region_northeast = region_subpolar
        map[region_northwest] = "Northwest Atlantic"
        map[region_northeast] = "Northeast Atlantic"
        reg[m] = np.where((self.Lon[m] < 100 ) | (self.Lon[m] > 320 ), region_northeast, region_northwest)

        # Separate Mediterranean
        region_med = 8
        map[region_med] = "Mediterranean"
        med_bottom = self.Lat > 30
        med1 = (self.Lon >= -5.57 + 360) & (self.Lat < 41)
        med2 = (self.Lon < 36) & (self.Lat < 46)
        reg[(reg == region_northeast) & (med1 | med2) & med_bottom] = region_med

        return ThompsonBasins(self.Lon, self.Lat, reg, map)


    def _reorder_colormap(self, colormap):
        map_inv = {v:k for k,v in self.map.items()}
        regions = [map_inv[lab] for lab in colormap if lab in map_inv]
        # regions = np.unique(self.reg[np.isfinite(self.reg)]).tolist()
        array = np.empty(self.reg.shape, dtype=float)
        array.fill(np.nan)
        map = {}
        for i, r in enumerate(regions):
            array[self.reg == r] = i
            map[i] = self.map[r]
        return ThompsonBasins(self.Lon, self.Lat, array, map)


    def interp(self, lon, lat):
        interp = RegularGridInterpolator((self.lon, self.lat), np.isnan(self.reg.T)+0., bounds_error=False, fill_value=1)
        Lon, Lat = np.meshgrid(lon, lat)
        # coords = np.array([Lon.flatten(), Lat.flatten()]).T
        coords = np.mod(Lon, 360), Lat
        newmask = interp(coords) > .5
        newreg = self.interp_nearest(coords).astype(float)
        newreg[newmask] = np.nan
        return ThompsonBasins(lon, lat, newreg, self.map.copy())