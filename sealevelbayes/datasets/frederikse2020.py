"""Load data from Frederikse2020"""
from pathlib import Path
import numpy as np
from scipy.stats import norm
import netCDF4 as nc
import pandas as pd
import xarray as xa

from sealevelbayes.datasets.manager import get_datapath
from sealevelbayes.datasets.catalogue import require_frederikse, require_bamber
from sealevelbayes.datasets.constants import gt_to_mm_sle

sources = ["AIS", "GrIS", "glac", "steric", "tws", "total"]
extended_sources = sources + ["gia", "vlm"]

root = get_datapath("zenodo-3862995-frederikse2020")
PERSONALCOM = get_datapath("frederikse2020-personal-comm")

def load_region_info():
    require_frederikse()
    sheets = ["Subpolar North Atlantic", "Indian Ocean - South Pacific", "Subtropical North Atlantic",
              "East Pacific", "South Atlantic", "Northwest Pacific"]

    dfs = []
    for i, basin in enumerate(sheets):
        sheet = pd.read_excel(root / "region_info.xlsx", sheet_name=i).set_index("Unnamed: 0")
        sheet.index = sheet.index + 1000*i
        sheet["Basin"] = basin
        if basin == "Subpolar North Atlantic":
            lon = sheet["Longitude"].values
            coastline = np.where((lon < 50) | (lon > 320), "Subpolar North Atl. East", "Subpolar North Atl. West")
        else:
            coastline = basin
        sheet["coastline"] = coastline
        dfs.append(sheet)
    region_info = pd.concat(dfs)
    duplicate_indices = [119, 120]
    region_info = region_info.loc[[ix for i, ix in enumerate(region_info.index) if ix not in duplicate_indices]]
    return region_info

region_info = load_region_info()

def open_gia():
    # fp_gia = root / "../GIA/dsea.1grid_O512.nc" # ICe6G-C
    # return xa.open_dataset(fp_gia)["Dsea_250"].rename({"Lat":"lat", "Lon": "lon"})
    fp_gia =  PERSONALCOM / "GIA_Caron_stats_05.nc" # Lambert
    return xa.open_dataset(fp_gia)


def _nearest_valid(point, lon, lat, valid_mask):
    """point: [lon, lat]
    lon, lat, valid_mask: mask of valid points to look for
    """
    ii, jj = np.where(valid_mask)
    values_valid = valid_mask[ii, jj]  # 1-D array of valid values
    lon_valid = lon[jj]
    lat_valid = lat[ii]
    distances = (lon_valid - point[0])**2 * np.cos(np.deg2rad(point[1]))**2 + (lat_valid - point[1])**2
    k = distances.argmin()
    return {"i": ii[k], "j": jj[k], "lat": lat_valid[k], "lon": lon_valid[k], "distance": distances[k]**0.5*6371e3*np.pi/180}

# & (bathy.values < -2000)

def load_grid(lon, lat, include_gia=False, include_err=False, max_distance=150e3, steric_depth=None, raise_error=True, verbose=True):
    """load all time-series for a specific location on gridded data (does not include VLM)"""

    # search for nearest valid data
    # not all data points are valid
    # "total" component provides the loosest mask
    # "steric" component has less valid data
    # in some cases, it might be preferable to serach for a nearby, deep point to have a more representative "steric quantity" (e.g. 2000 meters)
    # use the "steric-depth" parameter

    # Map tide-gauge locations to i,j indices for the steric component (the other components are OK)
        # main mask
    total = xa.open_dataset(root/f"total.nc").sel(time=2000)["total_rsl_mean"].load()
    valid_total = np.isfinite(total.values)

    loc = _nearest_valid([lon, lat], total.lon.values, total.lat.values, valid_total)

    if loc["distance"] > max_distance:
        if verbose:
            print(f"!! nearest lon/lat location with valid points at {loc['distance']*1e-3} km:",(loc["lon"], loc["lat"]),"instead of", (lon, lat))
        if raise_error:
            raise ValueError(f"max distance too large: {loc['distance']:.0f} > {max_distance}")
            # assert loc["distance"] < max_distance

    # mask for steric data
    steric = xa.open_dataset(root/f"steric.nc").sel(time=2000)["steric_mean"].load()
    valid_steric = np.isfinite(steric.values)

    if steric_depth:
        bathy = xa.open_dataset(get_datapath("Bathymetry/GEBCO_2020_05deg2.nc"))["elevation"]
        valid_steric = valid_steric & (bathy.values < -steric_depth)

    loc_steric = _nearest_valid([lon, lat], total.lon.values, total.lat.values, valid_steric)

    if verbose and loc_steric["distance"] > 50e3:
        print(f"!! nearest lon/lat location with valid (and/or deep) steric data at {loc_steric['distance']*1e-3} km:",(loc_steric["lon"], loc_steric["lat"]),"instead of", (lon, lat))

    data = {}
    for source in sources:
        ds = xa.open_dataset(root/f"{source}.nc")
        if source == "steric":
            variable = f"{source}_mean"
            i, j = loc_steric["i"], loc_steric["j"]
        else:
            variable = f"{source}_rsl_mean"
            i, j = loc["i"], loc["j"]

        timeseries = ds[variable][:, i, j].to_pandas()  # sel method is buggy and returns NaNs
        assert timeseries.dropna().size > 0
        data[source] = timeseries

        if include_err:
            variable = f"{source}_sterr" if source == "steric" else f"{source}_rsl_sterr"
            timeseries_std = ds[variable][:, i, j].to_pandas()
            data[source+"_std"] = timeseries_std

    # also add GIA ? (GIA is defined everywhere, and should be chosen as close as possible to the actual site requested)
    if include_gia:
        gia = open_gia().sel(lon=lon, lat=lat, method="nearest")
        trend = gia["rsl_mean"].to_pandas()
        data["gia"] = pd.Series((timeseries.index-2000)*trend, index=timeseries.index)
        if include_err:
            trend_err = gia["rsl_sterr"].to_pandas()
            data["gia_err"] = pd.Series((timeseries.index-2000)*trend_err, index=timeseries.index)

    return pd.DataFrame(data)


def search_station_ID(name):
    results = []
    for i, station in enumerate(region_info["Station names"]):
        if name.lower() in station.lower():
            results.append( region_info.iloc[i] )
            # print("Found station: ", region_info.iloc[i])
            # print("Found station: ", region_info.iloc[i])
    return pd.DataFrame(results)


def load_location(ID, include_vlm=True, include_err=False, **kwargs):
    """load all time-series for a specific tide-gauge location(see region_info.xlsx): includes GIA and VLM"""

    station = region_info.loc[ID]
    lat = station["Latitude"]
    lon = station["Longitude"]

    data = load_grid(lon, lat, include_err=include_err, include_gia=include_vlm, **kwargs)

    # also include the VLM
    if include_vlm:
        vlm = station["Residual VLM [mean]"] #*1e-3 # mm/yr (?) to m/yr
        data["vlm-residual"] = -pd.Series((data.index-2000)*vlm, index=data.index)
        data["vlm"] = data["gia"] + data["vlm-residual"]

        if not np.isfinite(vlm):
            data["vlm"] = data["gia"]

        if include_err:
            vlm_lo = -station["Residual VLM [lower]"]
            vlm_hi = -station["Residual VLM [upper]"]
            vlm_err = (vlm_hi-vlm_lo)/2 / norm().ppf(0.95)  # they indicate 5th to 95th percentils (90% range = 2 * 1.64 sigma)
            data["vlm-residual_err"] = pd.Series(np.abs(data.index-2000)*vlm_err, index=data.index)
            data["vlm_err"] = data["vlm-residual_err"] ## it is tricky to come back to the actual error, so
            # here we assume that the VLM error is the same as VLM residual error
            # According to Thomas Frederikse (if I understood him correctly), this should be an overestimate, because
            # it was derived by adding VLM trend error and GIA error (quadratically).
            # VLM residual error was computed as sum of vlm error and gia error: we do the inverse operation here
            # data["vlm_err"] = (data["vlm-residual_err"]**2 - data["gia_err"]**2)**.5
            # data["vlm_err"] = ((data["gia_err"] if "gia_err" in data else 0)**2 + data["vlm-residual_err"]**2)**.5

            # here special case where vlm-residual could not be retrieved with sufficient precision
            # vlm residual assumed to be 0 +- 1 mm/yr
            if not np.isfinite(vlm_err):
                data["vlm_err"] = (1**2 + data["gia_err"]**2)**.5

    return data



def load_global():
    df0 = pd.read_excel(root / "global_basin_timeseries.xlsx").set_index("Unnamed: 0")
    df0.index.name = "Year"

# sources = ["AIS", "GrIS", "glac", "steric", "tws", "total"]
    data = {}
    data["steric"] = df0["Steric [mean]"]
    data["glac"] = df0["Glaciers [mean]"]
    data["AIS"] = df0["Antarctic Ice Sheet [mean]"]
    data["GrIS"] = df0["Greenland Ice Sheet [mean]"]
    data["tws"] = df0["Terrestrial Water Storage [mean]"]
    data["total"] = df0["Observed GMSL [mean]"]

    return pd.DataFrame(data)




def open_rgi_fingerprint(region_id):
    fname = PERSONALCOM / f"fingerprints/RGI_{region_id}.nc"
    return xa.open_dataset(fname)


def load_bamber():
    filepath = require_bamber()
    return pd.read_csv(filepath, sep="\t", skiprows=20, index_col=0)

def load_mouginot():
    filepath = get_datapath("10.1073/pnas.1904242116/mouginot_extracted.csv")
    return pd.read_csv(filepath, index_col=0, sep=",")

def load_kjeldsen():
    # sealeveldata/10.1038/nature16183/kjeldsen_extracted_mb_fig3b.csv
    filepath = get_datapath("10.1038/nature16183/kjeldsen_extracted_mb_fig3b.csv")
    return pd.read_csv(filepath, index_col=0, sep="\t")

def load_greenland_datasets(years=None, datasets=None):

    if years is None:
        years = np.arange(1900, 2018+1)

    if datasets is None:
        datasets = ["bamber2018", "mouginot2019", "kjeldsen2015"]

    mean = {}
    sigma = {}

    for name in datasets:
        if name == "bamber2018":
            bamber = load_bamber().reindex(years)
            mean[name] = bamber["MB [Gt/a] (Mass balance GrIS (dM GrIS): ...)"] * gt_to_mm_sle
            sigma[name] = bamber["Std dev [±] (Stdev GrIS: 1-sigma uncertain...)"] * gt_to_mm_sle

        elif name == "mouginot2019":
            mouginot = load_mouginot().reindex(years)
            mean[name] = mouginot["MB.GIS"] * gt_to_mm_sle
            sigma[name] = mouginot["MBERROR.GIS"] * gt_to_mm_sle

        elif name == "kjeldsen2015":
            kjeldsen = load_kjeldsen().reindex(years)
            mean[name] = kjeldsen["MB_Gt_yr"] * gt_to_mm_sle
            sigma[name] = kjeldsen["MB_1sigma"] * gt_to_mm_sle

        else:
            raise ValueError(f"Unknown dataset: {name}")

    return pd.DataFrame(mean), pd.DataFrame(sigma)


def sample_greenland_dataset(samples=5000, rng=None, random_seed=42, mean=None, sigma=None, years=None, datasets=None):

    if mean is None:
        mean, sigma = load_greenland_datasets(years, datasets)

    years = mean.index.values

    if rng is None:
        rng = np.random.default_rng(random_seed)

    all_mean = mean.values.T # datasets x time
    all_sigma = sigma.values.T # datasets x time

    all_samples = rng.normal(all_mean, all_sigma, size=(samples, all_mean.shape[0], all_mean.shape[1])) # samples x datasets x time

    sampled = np.empty((years.size, samples))
    for i in range(len(years)):
        choices = np.where(np.isfinite(all_mean[..., i]))[0]
        dataset_choice = rng.choice(choices, size=samples)
        sampled[i] = all_samples[np.arange(samples), dataset_choice, i]

    return -pd.DataFrame(sampled, index=years)