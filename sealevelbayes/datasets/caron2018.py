"""
The dataset for this module was provided by L. Caron (personal communication) in the form of a 5000-member ensemble.

REFERENCES
----------
Caron, L., E.R. Ivins, E. Larour, S. Adhikari, J. Nilsson and G. Blewitt (2018), GIA model statistics for GRACE hydrology, cryosphere and ocean science, Geophys. Res. Lett., 45, doi: 10.1002/2017GL076644
M. A. Wieczorek, M. Meschede, A. Broquet, T. Brugere, A. Corbin, EricAtORS, A. Hattori, A. Kalinin, J. Kohler, D. Kutra, K. Leinweber, P. Lobo, I. Oshchepkov, P.-L. Phan, O. Poplawski, M. Reinecke, E. Sales de Andrade, E. Schnetter, S. Schröder, J. Sierra, A. Vasishta, A. Walker, xoviat, B. Xu (2023). SHTOOLS: Version 4.10.4, Zenodo, doi:10.5281/zenodo.592762
Ditmar (2018), Journal of Geodesy, doi:10.1007/s00190-018-1128-0  (Equations (1-4))
"""
from sealevelbayes.datasets.manager import get_datapath

def get_caron2018_by_frederikse():
    return get_datapath("frederikse2020-personal-comm/GIA_Caron_stats_05.nc")

def get_caron2018_ensemble_data():
    return get_datapath("caron2018/gia_ensemble_caron2018.nc")