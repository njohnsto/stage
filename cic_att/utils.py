
from __future__ import division, print_function
import numpy as np
import math


def random_pl(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b

    Parameters
    ----------
    a    : float
          Lower bound of x range
    b    : float
          Upper bound of x range
    g    : float
          Power law index
    size : int
          Number of samples

    Returns
    -------
    rpl  : numpy array (float)
          Random power law

    Raises
    ------
    ValueError
        If ``a >= b``
    """
    if not a < b:
        raise ValueError('Lower limit not less than upper limit: (a, b) = {} >= {}'.format(a, b))

    r = np.random.random(size=size)
    ag, bg = a**g, b**g

    rpl = (ag + (bg - ag) * r)**(1. / g)

    return rpl


def get_signal_ref(e_ref, B, energy):
    """Function for computing reference signal

    Parameters
    ----------
    e_ref   : float
            Reference energy
    B       : float
            Relational scale to s38
    energy  : array (float)
            Energy values at which to get reference signal

    Returns
    -------
    sig_ref : array (float)
             Scaled reference signal values
    """
    sig_ref = (energy / e_ref)**(1 / B)

    return sig_ref


def get_s125(cos2, alpha, beta, ref_signal):
    """Function for computing s125

    Parameters
    ----------
    cos2       : array (float)
                Values of cos^2
    alpha      : float
                Linear coefficient of S125 parametrization
    beta       : float
                Quadratic coefficient of S125 parametrization
    ref_signal : array (float)
                Reference signal value, s38 (S at 38 deg zenith)

    Returns
    -------
    s_125      : array (float)
                Signal at 125 parametrized
    """

    # Center cosine **2 values on cosine(38)**2
    cos2_ref = np.cos(math.radians(38))**2
    x = cos2 - cos2_ref

    s_125 = ref_signal * (1 + alpha*x + beta*x**2)

    return s_125
