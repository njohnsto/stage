
from __future__ import division, print_function
import numpy as np


def lnlike(params, cos2, y, yerr):
    """Log-likelihood of function given parameters

    Parameters
    ----------
    params : list
             function parameters
    cos2   : array (float)
             cosine square values
    y      : array (float)
             function values
    yerr   : array (float)
             function uncertanty values
    """
    a, b, f = params
    model = f * (b * cos2**2 + a * cos2 + 1)

    return -0.5 * np.sum((y - model)**2 / yerr**2 + np.log(yerr**2))


def lnprior(params):
    """Flat prior on parameters for MCMC

    Parameters
    ----------
    params : list
             Function parameters
    """
    a, b, f = params
    if -5.0 < b < 0. and 0. < a < 5 and 0. < f:
        return 0.0

    return -np.inf


def lnprob(params, cos2, y, yerr):
    """Total log-likelihood provided observations

    Parameters
    ----------
    params : list
             Function parameters
    cos2   : array (float)
             cosine square values
    y      : array (float)
             function values
    yerr   : array (float)
             function uncertanty values

    Returns
    -------
    llh    : float
             log-likelihood
    """

    # Get prior given parameters
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf

    # Include likelihood given data
    llh = lp + lnlike(params, cos2, y, yerr)

    return llh
