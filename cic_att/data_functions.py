
from __future__ import division, print_function
import numpy as np
import pandas as pd
import scipy as sp

import math

import emcee

from utils import random_pl, get_signal_ref, get_s125
from likelihood import lnlike, lnprior, lnprob


def generate_toy_data(events, minE, maxE, gamma, A, B, alpha, beta, maxTheta):
    """Function that generates simulated data given predefined input

    Parameters
    ----------
    events    : int
                Number of events to simulate
    minE      : float
                Minimum energy (eV)
    maxE      : float
                Maximum energy (eV)
    gamma     : float
                Spectral index
    A         : float
                Reference energy (eV)
    B         : float
                Relation to s38
    alpha     : float
                True attenuation coeff
    beta      : float
                True attenuation coeff
    maxTheta  : float
                Minimum zenith value (degrees)

    Returns
    -------
    data : pandas array
           Pandas array with simulated data
    """
    # Random energy samples from power law
    energy = random_pl(minE, maxE, gamma, events)
    s38 = get_signal_ref(A, B, energy)
    cos2 = np.random.uniform(np.cos(math.radians(maxTheta))**2, 1, events)
    s125 = get_s125(cos2, alpha, beta, s38)

    # Define & fill pandas data array object
    data = pd.DataFrame()
    data['energy'] = energy
    data['cos2'] = cos2
    data['s38'] = s38
    data['zenith'] = np.arccos(np.sqrt(data.cos2))
    data['zenith_er'] = np.random.uniform(math.radians(0.5), math.radians(1.5), events)
    data['s125'] = s125
    data['s125_error'] = np.abs(np.random.uniform(0.05, 0.1, events) * s125)
    data['I'] = 0

    return data


def set_intensity(data, n_bins):
    """Function that calculates the intensity for each group

    Parameters
    ----------
    data  : pandas array
           Pandas data array object
    n_bins : int
           number of bins to

    Returns
    -------
    data : pandas array
    """
    # Define cosine squared bins
    min_cos2 = data.cos2.min()
    cos2_bins = np.linspace(min_cos2, 1, n_bins, endpoint=True)

    # sort data by S125 values
    data = data.sort_values(['s125'])

    # Bin and group by cosine ^ 2
    ind = np.digitize(data['cos2'], cos2_bins)
    groups = data.groupby(ind)

    for name, group in groups:
        values = group['s125'].apply(lambda x: group[group['s125'] > x].count())
        data.loc[group.I.index.tolist(), 'I'] = values.I

    return data


def get_data_to_fit(data, intensity, n_bins):
    """Function that extracts requested data based
       on intensity, and bins in cosine**2 in
       accordance with requested input

    Parameters
    ----------
    data      : pandas array
                Pandas data array object
    intensity : float
                Requested intensity value
    n_bins    : int
                Number of zenith bins

    Returns
    -------
    s125_fit        : array (float)
                      s125 values to fit
    s125_fit_error  : array (float)
                      associated uncertainties
    bin_centers     : array (float)
                      centers of cosine**2 bins
    """
    # Define centers of bins
    min_cos2 = data.cos2.min()
    bins = np.linspace(min_cos2, 1., n_bins, endpoint=True)
    bin_centers = np.diff(bins)/2+bins[0:-1]
    # Get data values at intensity
    val = data.loc[data.I == intensity]
    s125_fit = np.asarray(val.s125.tolist())
    s125_fit_error = np.asarray(val.s125_error.tolist())
    # introduce checks for intensity
    return (s125_fit, bin_centers, s125_fit_error)


def get_attenuation_parameters(s125, cos2):
    """Function that performs a s125 parametrization fit with scipy

    Parameters
    ----------
    s125 : array (float)
           S125 values
    cos2 : array (float)
           associated cosine**2 values

    Returns
    -------
    parameters  : list (float)
                  best fit parameters
    cov2        : 2darray (float)
                  covariance matrix
    """
    parameters, cov2 = sp.optimize.curve_fit(get_s125, cos2, s125)
    return (parameters, cov2)


def get_attenuation_parameters2(init_params, s125_fit, s125_fit_error, bins):
    """Function that performs 

    Parameters
    ----------
    init_params     : list (float)
                      initial parameters
    s125_fit        : array (float)
                      s125 values to fit
    s125_fit_error  : array (float)
                      associated uncertainties
    bin_centers     : array (float)
                      centers of cosine**2 bins

    Returns
    -------
    good_samples    : 2darray (float)
                      MCMC samples having removed
                      values outside allowed region
    """

    # Use as guess
    a_true = init_params[0]
    b_true = init_params[1]
    f_true = init_params[2]
    y = get_s125(bins, a_true, b_true, f_true)

    # Recenter cosine**2 to 38 deg
    cos_ref = np.cos(math.radians(38))**2
    cos2 = bins - cos_ref
    # Minimize negative log-likelihood function
    nll = lambda *args: -lnlike(*args)
    result = sp.optimize.minimize(nll, [a_true, b_true, f_true], args=(cos2, y, s125_fit_error))

    # Prep for emcee
    ndim, nwalkers = 3, 100
    # Set starting positions wiggled normally around min nll estimate
    pos = [result['x'] * np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(cos2, y, s125_fit_error))

    # Run emcee, get samples
    sampler.run_mcmc(pos, 5000)
    all_samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
    good_samples = []
    # Remove burn-in and samples outside of allowed prior bounds
    for i in range(0, 5000-500):
        if lnprior(all_samples[i]) != -np.inf:
            good_samples.append(all_samples[i])
    good_samples = np.asarray(good_samples)
    print("Life is amazing")

    return good_samples


def get_bootstrap_data(data):
    """Function that wiggles s125 by assoc uncertainties

    Parameters
    ----------
    data  : pandas array
            Pandas data array object

    Returns
    -------
    data : pandas array
            Pandas data array object
    """
    new_data = data.copy()
    new_data.s125 = np.random.normal(data.s125.tolist(), data.s125_error.tolist())
    return new_data



def merge_and_select_data(filenames, output, number_of_files=10000):
    """ Function that reads the original IceCube data and creates a smaller file 
        containing just basic info after the selection
        
    Parameters
    ----------
          filenames : array of file names 
             output : output file name
    number_of_files : number of files to process
    
    Returns
    -------
    store: pandas HDFStore
    
   
    """
    store = pd.HDFStore(output,complib='blosc')
    for i, file in enumerate(filenames):
        print(i, file)
        s = pd.HDFStore(file)
        dataGen = s.select('Laputop')
        dataGen.drop(dataGen.columns.difference(['Run', 'Event','x','y','z','zenith','azimuth','time']), 1, inplace=True)
        dataRec = s.select('LaputopParams')
        dataRec.drop(dataRec.columns.difference(['s125','beta','chi2','ndf']), 1, inplace=True)
        dataCuts = s.select('IT73AnalysisIceTopQualityCuts')
        dataCuts.drop(['Run', 'Event','SubEvent','SubEventStream'], 1, inplace=True)
        dataGen = dataGen.dropna(axis=1, how='all')
        dataRec = dataRec.dropna(axis=1, how='all')
        data = pd.concat([dataGen, dataRec, dataCuts], axis=1, join_axes=[dataGen.index])
        data = data.query('exists!=0 & IceTopMaxSignalInside!=0 & IceTop_reco_succeeded!=0'
                         ' & IceTop_StandardFilter!=0 & IceTopMaxSignalInside !=0 '
                         ' & Laputop_FractionContainment!=0 & BetaCutPassed!=0 &s125>1')
        data.drop(data.columns.difference(['Run','Event','x','y','z','zenith','azimuth',
                                          's125','beta','chi2','ndf']), axis=1,inplace=True)
    
        store.append(key ='data', value=data,  format='t',chunksize=200000)
        s.close()
        if i>number_of_files:
            break

    return store