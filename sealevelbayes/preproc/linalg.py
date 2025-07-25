import numpy as np
from numpy.linalg import pinv, inv, cholesky
from numpy import dot
from scipy.linalg import lstsq
from sealevelbayes.logs import logger

def detrend_timeseries(values, n=1, return_coefs=False, mask=None, keep_mean=False):
    """ Detrend time-series field by fitting a quadratic trend to each grid cell

    Input shape: nt x ... where nt is the length of the time-series and subsequent dimensions in n-d space
    Function written with "zos" in mind as input variable (cumulative sea level rise).
    n: polynomial order of trend (n=1 : linear trend, n=2 quadratic trend)
    mask: boolean mask along the time-axis (from masked array)
    """
    nt = values.shape[0]
    t = np.arange(values.shape[0]) - values.shape[0]//2
    A = np.array([t**2, t, np.ones(t.size)][-n-1:]).T

    logger.debug(f"detrend_timeseries A shape: {A.shape}")

    if mask is not None:
        A = A[~mask]
        values = values[~mask]

    x, _, _, _ = lstsq(A, values, cond=None)
    trend = A@x

    detrended = values - trend

    if keep_mean:
        mean = values.mean(axis=0)
        detmean = detrended.mean(axis=0)
        detrended += (mean - detmean)[None]

    if return_coefs:
        return detrended, x
    else:
        return detrended


def calc_lineartrend_fast(values, mask=None):
    """ values shape: nt x ...; mask must be 1D
    """
    _, coef = detrend_timeseries(values, n=1, return_coefs=True, mask=mask)
    return coef[0]


def lscov(A, B, V=None, chol=None):
    """Matlab's lscov as described in https://www.mathworks.com/help/matlab/ref/lscov.html

    Ax = B with covariance matrix V

    Based on matlab doc:

    x = inv(A'*inv(V)*A)*A'*inv(V)*B
    mse = B'*(inv(V) - inv(V)*A*inv(A'*inv(V)*A)*A'*inv(V))*B./(m-n)
    S = inv(A'*inv(V)*A)*mse
    stdx = sqrt(diag(S))

    I calculate that if V = L L'
    And by replacing A by inv(L)A and B by inv(L)B

    P = inv(A'A)
    x = PA'B
    Y = Ax
    mse = B'(B - Y) / (m-n)
    S = P*mse

    Note that is the covariance matrix is exact (and not arbitrarily scaled as is assumed here), you need to scale back S -> S/mse and stdx/sqrt(mse)
    """
    m, n = A.shape
    if V is not None:
        chol = cholesky(V)
    if chol is None:
        chol = np.eye(m)
    iL = pinv(chol)
    A = dot(iL, A)
    B = dot(iL, B)
    P = pinv(dot(A.T, A))
    x = dot(dot(P, A.T), B)
    Y = dot(A, x)
    mse = dot(B.T, B-Y)/(m-n)
    S = P*mse
    stdx = np.sqrt(np.diag(S))
    return x, stdx, mse, S


def nancorrcoef(res1, res2):
    res1 = res1 - np.nanmean(res1)
    res2 = res2 - np.nanmean(res2)
    rho = np.nanmean(res1*res2)/(np.nanmean(res1**2)*np.nanmean(res2**2))**.5
    return rho


def ar1cov(rho, sigma2, n):
    # ...build cov matrix for an ar1 process
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = sigma2*rho**np.abs(i-j)
    return cov


def cholrootar1(rho, sigma, n):
    """Direct computation of Cholesky root for an AR(1) matrix

    reference: https://blogs.sas.com/content/iml/2018/10/03/ar1-cholesky-root-simulation.html

    start AR1Root(rho, p);
       R = j(p,p,0);                /* allocate p x p matrix */
       R[1,] = rho##(0:p-1);        /* formula for 1st row */
       c = sqrt(1 - rho**2);        /* scaling factor: c^2 + rho^2 = 1 */
       R2 = c * R[1,];              /* formula for 2nd row */
       do j = 2 to p;               /* shift elements in 2nd row for remaining rows */
          R[j, j:p] = R2[,1:p-j+1];
       end;
       return R;
    finish;
    """
    R = np.zeros((n,n))
    rhop = 1
    for j in range(n):
        # R[0,j] = rhop
        R[j, 0] = rhop
        rhop *= rho
    c = (1-rho**2)**.5
    R2 = c*R[:, 0]
    for i in range(1, n):
        R[i:, i] = R2[:n-i]
    return R*sigma


def lsar1(A, B):
    """NOTE: This is slower than using statsmodel iterative_fit method...
    """
    # first get an estimate without covariance
    x, stdx, mse, S = lscov(A, B, np.eye(B.size))

    # estimate AR1 from residual
    res = B - dot(A, x)
    rho = np.corrcoef(res[1:], res[:-1])[0, 1]
    # cov = ar1cov(rho, np.var(res), B.size)
    L = cholrootar1(rho, np.std(res), B.size)

    # now get final estimate assuming AR1 covariance
    x, stdx, mse, S = lscov(A, B, chol=L)

    return x, stdx, mse, S


def lsar1statsmodels(A, B):
    import statsmodels.api as sm
    fit = sm.GLSAR(B, A).iterative_fit()
    cov = fit.cov_params()
    return fit.params, np.diag(cov)**.5, (fit.resid**2).mean()**.5, cov


def lscovstatsmodels(A, B, V=None):
    import statsmodels.api as sm
    if V is None:
        fit = sm.OLS(B, A).fit()
    else:
        fit = sm.GLS(B, A, sigma=V).fit()
    cov = fit.cov_params()
    return fit.params, np.diag(cov)**.5, (fit.resid**2).mean()**.5, cov


### Some more earlier code used for auto-correlated trend calculation (previously in tidegaugeobs)

def calc_lineartrend(values, years=None, cov=None, err=None):
    """Returns linear trend, its standard error and the residuals

    cov: optionally provide covariance matrix for the error residuals
    """
    import statsmodels.formula.api as smf

    if years is None:
        years = np.arange(values.size)

    dat = pd.DataFrame({
            "obs" : values,
            "year": years,
            })

    if err is not None:
        assert cov is None
        cov = np.diag(err**2)

    if cov is not None:
        fit = smf.gls('obs ~ year', data=dat, sigma=cov).fit()
    else:
        fit = smf.ols('obs ~ year', data=dat).fit()

    intercept_sd, trend_sd = np.diag(fit.cov_params())**.5
    return fit.params["year"], trend_sd, values - fit.params["year"]*years - fit.params['Intercept']
    # return fit.params["year"], trend_sd, fit.resid.values


def calc_lineartrend_with_ar1(values, years=None, err=None, ar1_regress=False, return_ar1_params=False):
    # Compute linear trend in observed tide-gauge data:
    import statsmodels.api as sm

    if years is None:
        years = np.arange(values.size)

    # Here we ignore error because otherwise the residual are not centered
    # and AR1 parameters detection might be affected
    trend, sd, res = calc_lineartrend(values, years=years)

    if ar1_regress:
        rho = sm.OLS(res[1:], sm.add_constant(res[:-1]), missing='drop').fit().params[1]
        # print(rho)
    else:
        rho = nancorrcoef(res[1:], res[:-1])
        # print(rho)

    sigma2 = np.nanvar(res)

    cov = ar1cov(
        rho=rho,
        sigma2=sigma2,
        n=len(res))

    if err is not None:
        cov += np.diag(err**2)

    # only use non-nan values
    ii = np.isfinite(values)

    trend, sd, res = calc_lineartrend(values[ii], cov=cov[ii][:, ii], years=years[ii])

    if return_ar1_params:
        return trend, sd, res, rho, sigma2/(1-rho**2)**.5
    else:
        return trend, sd, res


    # fit = smf.gls('obs ~ year', data=dat.iloc[ii], sigma=cov[ii][:, ii]).fit()
    # obs_trend_mean = fit2.params["year"]
    # _, obs_trend_std = np.diag(fit2.cov_params())**.5

    # return tg_years




def _prepare_lstsq(zos, vs=None):
    """prepare least square preditor for fingerprints calculation
    """
    if vs is None:
        vs = np.arange(zos.shape[0])
    a = np.array([vs, np.ones(vs.size)]).T
    if zos.ndim > 2:
        b = np.reshape(zos, (vs.size, zos.size // vs.size))
    return a, b
