import numpy as np
from scipy.stats import norm, lognorm
from sealevelbayes.logs import logger

class ReverseDist:
    def __init__(self, dist):
        self.dist = dist
        self.args = ('reverse of',) + self.dist.args
        self.name = f"revserse {dist.dist.name}"

    def ppf(self, q):
        return -self.dist.ppf(q)[::-1]
    

def fit_dist_to_quantiles(values, quantiles, dist_name="auto", skewness_threshold=5, raise_reverse=True):
    """
    Parameters
    ----------
    values: length-3 array
    quantiles: length-3 array
    dist_name: "auto", "norm" or "lognorm"
    skewness_threshold: to determine whether to use lognorm in auto mode, 5% by default
        skewness defined as ((hi + lo)/2 - mid)/(hi - lo)*100
    raise_reverse: bool, True by default
        this function supports fitting reverse log-norm distribution, but here is no corresponding 
        scipy function, so a ReverseDist wrapper was written, with only a limited set of compatible methods for now
        By default an exception is raised, but setting this flag to True can enable this experiment feature.

    Return
    ------
    scipy-like distribution (scipy or ReverseDist) with a `ppf` method
    """
    if not len(values) == 3: raise NotImplementedError("Only 3 quantiles are supported to far")
    if not len(quantiles) == len(values): raise ValueError("The number of quantiles must match the number of values")
    if 0.5 not in quantiles: raise NotImplementedError('quantile levels must include 0.5')

    # make sure we have mid, lo, hi
    i_sort = np.argsort(quantiles)
    lo, mid, hi = np.asarray(values)[i_sort]
    lo_q, mid_q, hi_q = np.asarray(quantiles)[i_sort]
    values = [mid, lo, hi]
    quantiles = [mid_q, lo_q, hi_q]

    assert lo_q < mid_q < hi_q, "THIS SHOULD NOT HAPPEN: quantile levels are not sorted"
    assert lo < mid < hi, "THIS SHOULD NOT HAPPEN: quantiles are not sorted"

    if dist_name == "auto":
        skewness = ((hi + lo)/2 - mid)/(hi - lo)*100
        if skewness > skewness_threshold:  # AIS in IPCCAR6 medium conf yields 10%, resulting total SLR yields > 6%
            dist_name = "lognorm"
        else:
            dist_name = "norm"
        logger.info("fit_dist::skewness", skewness, '=> choice:',dist_name)        

    if dist_name == "norm":
        return norm(mid, ((hi-mid)+(mid-lo))/2 / norm.ppf(hi_q))

    if dist_name == "lognorm":

        reverse = hi - mid < mid - lo

        if reverse:
            if raise_reverse: raise ValueError("The data's upper tail is larger than the lower tail. Set raise_reverse=False to return a ReverseDist (experimental).")
            mid, lo, hi = -mid, -hi, -lo

        # this ensures symmetry in the log-transformed quantiles (I wrote it down and solved the equality)
        loc = (mid ** 2 - hi*lo) / (2*mid - lo - hi)

        assert lo - loc > 0
        # It's not too difficult to prove `lo - loc > 0` since we have hi - mid >= mid - lo, and as a result 2*mid - lo - hi <= 0
        # the equality lo - loc > 0 becomes lo * (2*mid - lo - hi) - mid **2 - hi*lo <= 0
        # and suffices to note that lo * (2*mid - lo - hi) - mid **2 - hi*lo = - (mid - lo)**2 which is always < 0

        normdist = fit_dist_to_quantiles([np.log(mid - loc), np.log(lo - loc), np.log(hi - loc)], quantiles, "norm")
        mu, sigma = normdist.args
        dist = lognorm(sigma, loc, np.exp(mu))

        if reverse:
            dist = ReverseDist(dist)

        return dist

    else:
        raise NotImplementedError(dist_name)
