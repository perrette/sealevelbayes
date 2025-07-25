## Additional code to laod the data
import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xa

import sealevelbayes.datasets.frederikse2020 as frederikse2020
from sealevelbayes.preproc.linalg import _prepare_lstsq


def _calc_globalmean(lon, lat, zos):

    # compute global mean glacier contribution
    LON, LAT = np.meshgrid(lon, lat)
    w = np.cos(np.deg2rad(LAT))

    globalmean = np.empty(zos.shape[:-2])
    for i in range(globalmean.size):
        m = np.isfinite(zos[i])
        globalmean.flat[i] = np.sum(w[m]*zos[i][m])/np.sum(w[m])

    return globalmean

def calc_globalmean(zos):
    globalmean = _calc_globalmean(zos.lon, zos.lat, zos.values)
    return xa.DataArray(globalmean, coords={"time":zos.coords["time"]}, dims=["time"])


def _calc_fingerprint(global_slr, local_slr):
    a, b = _prepare_lstsq(local_slr, global_slr)
    x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)
    finger = x.reshape(x.shape[0], local_slr.shape[1], local_slr.shape[2])
    # residuals = residuals.reshape(data.shape[1], data.shape[2])
    return finger[0]


def calc_fingerprint(global_slr, local_slr):
    finger = _calc_fingerprint(np.asarray(global_slr), np.asarray(local_slr))

    return xa.DataArray(finger,
                    coords={"lon": local_slr.lon, "lat": local_slr.lat, "time": local_slr.time.values.mean().round().astype(int)},
                    dims=["lat", "lon"], attrs={"time": f"{local_slr.time[0].values}-{local_slr.time[-1].values}"})


def calc_fred_fingerprints(variable, year1=2000, year2=None):

    fingers = {}

    with xa.open_dataset(frederikse2020.root / f"{variable}.nc") as ds:

        rsl = ds[variable + "_rsl_mean"].loc[year1:year2].load()
        glb = calc_globalmean(rsl)

        # for suffix in ["rsl_mean", "rsl_sterr", "rad_mean", "rad_sterr"]:
        # The error terms are very small, calculated this way => skip them
        for suffix in ["rsl", "rad"]:
            v = variable + "_" + suffix
            data = ds[v+"_mean"].loc[year1:year2]

            fingers[v] = calc_fingerprint(glb, data)
            # The residuals here are zero, because we only recover an underlying fingerprint

    return xa.Dataset(fingers)


# def calc_fred_fingerprints_time_dependent(variables=["glac", "GrIS", "AIS", "tws"]):

#     fingers = {}

#     for variable in variables:

#         with xa.open_dataset(frederikse2020.root / f"{variable}.nc") as ds:

#             rsl = ds[variable + "_rsl_mean"]
#             glb = calc_globalmean(rsl)

#             # for suffix in ["rsl_mean", "rsl_sterr", "rad_mean", "rad_sterr"]:
#             # The error terms are very small, calculated this way => skip them
#             for suffix in ["rsl", "rad"]:
#                 v = variable + "_" + suffix
#                 data = ds[v+"_mean"].values

#                 coefs = np.diff(data, axis=0) / np.diff(glb, axis=0)[:, None, None]
#                 coefs = np.concatenate([coefs[[0]], coefs], axis=0)  # add the 1900 fingerprint equal to 1901

#                 fingers[v] = xa.DataArray(coefs,
#                     coords={"lon": ds.lon, "lat": ds.lat, "time": rsl.time.values},
#                     dims=["time", "lat", "lon"])

#     return xa.Dataset(fingers)

def calc_yearly_fred_fingerprints(variable):
    fingers = {}
    with xa.open_dataset(frederikse2020.root / f"{variable}.nc") as ds:
        rsl = ds[variable + "_rsl_mean"]
        glb = calc_globalmean(rsl)
        for suffix in ["rsl", "rad"]:
            v = variable + "_" + suffix
            data = ds[v+"_mean"]

            coefs = data.diff(dim="time") / glb.diff(dim="time")
            # coefs = np.concatenate([coefs[[0]], coefs], axis=0)  # add the 1900 fingerprint equal to 1901
            fingers[v] = xa.concat([coefs.isel(time=[0]), coefs], dim="time") # add the 1900 fingerprint equal to 1901

        return xa.Dataset(fingers)


def calc_fred_fingerprints_time_dependent(variable, w=5):
    """ Sliding w-year window over which the fingerprint is computed.
    """

    fingers = {}

    with xa.open_dataset(frederikse2020.root / f"{variable}.nc") as ds:

        rsl = ds[variable + "_rsl_mean"].load()
        glb = calc_globalmean(rsl)

        # for suffix in ["rsl_mean", "rsl_sterr", "rad_mean", "rad_sterr"]:
        # The error terms are very small, calculated this way => skip them
        for suffix in ["rsl", "rad"]:
            v = variable + "_" + suffix
            data = ds[v+"_mean"].load()

            finger_slices = []
            for i in range(rsl.time.size - w + 1):
                finger = calc_fingerprint(glb[i:i+w], data[i:i+w])
                finger_slices.append(finger)

            fingers[v] = xa.concat(finger_slices, dim='time')

    return xa.Dataset(fingers)



# def main():
#     ds = calc_fred_fingerprints_time_dependent()
#     ds.to_netcdf(FINGERPRINTSDIR/"fingerprints_ice_time_dependent.nc")
#     # ds = calc_fred_fingerprints()
#     # ds.to_netcdf(FINGERPRINTSDIR/"fingerprints_ice.nc")


# if __name__ == "__main__":
#     main()