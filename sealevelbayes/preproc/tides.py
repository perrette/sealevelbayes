import numpy as np

def nodal_tide_classical_formula(year, lat, solid_earth_factor=0.69):
    """see https://github.com/ISI-MIP/slr-tidegauges-future/issues/23

    Note the classical formula could be scaled up to 1.24 to account for additional weight exerted by the ocean onto the sea-floor after tidal redistrubion
    Graphically we can see that 1.24 is better than 1, but something in the middle would fit best.
    For example, 1.15 could be a good middle ground.
    """
    magnitude = solid_earth_factor*8.8*(3*np.sin(np.deg2rad(lat))**2-1)
    return -magnitude*np.cos(2*np.pi/18.61*(year - 1922.7))


def nodal_tide_scaled(year, lat, solid_earth_factor=0.69, loading=1.15):
    return loading*nodal_tide_classical_formula(year, lat, solid_earth_factor)