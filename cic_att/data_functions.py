
from __future__ import division, print_function
import numpy as np
import pandas as pd
import scipy as sp

import math

import emcee

from utils import random_pl, get_signal_ref, get_s125
from likelihood import lnlike, lnprior, lnprob


def generate_toy_data(events, inp):
    """Function that generates simulated data given predefined input

    Parameters
    ----------
    events    : int
                Number of events to simulate
    inp : dictionary containing            
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
    energy = random_pl(inp['minE'], inp['maxE'], inp['gamma'], events)
    s38 = get_signal_ref(inp['A'], inp['B'], energy)
    cos2 = np.random.uniform(np.cos(math.radians(inp['maxTheta']))**2, 1, events)
    s125 = get_s125(cos2, inp['alpha'], inp['beta'], s38)

    # Define & fill pandas data array object
    data = pd.DataFrame()
    data['energy'] = energy
    data['cos2'] = cos2
    data['s38'] = s38
    data['zenith'] = np.arccos(np.sqrt(data.cos2))
    data['zenith_er'] = np.random.uniform(math.radians(0.5), math.radians(1.5), events)
    data['s125'] = s125+ np.random.randn(len(s125))*s125*0.1
    data['s125_error'] = 0.1*s125
    data['I'] = 0

    
    #data.drop(data.columns.difference(['s125', 's125_error',"cos2", "I","zenith"]), 1, inplace= True)
    #data= data.loc[data.s125>4]
    #data= data.loc[data.cos2>0.5]  
    #data.reset_index(inplace=True)
    return data


def set_intensity_old(data, n_bins):
    """Function that calculates the intensity for each group
      ***** very slow *****
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
    return (data,groups)


def set_intensity(data, n_bins, variable='s125'):
    """Function that calculates the intensity for each group
        
    Parameters
    ----------
    data  : pandas array
           Pandas data array object
    n_bins : int
           number of bins for grouping

    Returns
    -------
    data : pandas array
    groups: the groups that were creating acording with the binning
    """
    # Define cosine squared bins
    min_cos2 = data.cos2.min()
    cos2_bins = np.linspace(min_cos2, 1, n_bins, endpoint=True)

    # sort data by S125 values
    data.sort_values([variable], ascending = False, inplace = True)

    # Bin and group by cosine ^ 2
    ind = np.digitize(data['cos2'], cos2_bins)
    groups = data.groupby(ind)
    for name, group in groups:
        initial_ind = group.I.index.tolist()
        test = group.copy()
        test.sort_values([variable], ascending= False, inplace = True)
        test.reset_index(inplace=True)
        test.drop(['index'], axis= 1, inplace=True)
        data.loc[initial_ind, 'I'] = test.index.tolist()
    return (data, groups)

    
    

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
    vals      : the values in the DataFrame that correspond to a certain intensity
                Pandas data array object 
    """
    # Define centers of bins
    min_cos2 = data.cos2.min()
    bins = np.linspace(min_cos2, 1., n_bins, endpoint=True)
    bin_centers = np.diff(bins)/2+bins[0:-1]
    bin_width = np.diff(bins)/2.
    # Get data values at intensity
    vals = data.loc[data.I == intensity].copy()
    vals['cos2_index'] = 0 
    for value in vals.cos2:
        idx = np.searchsorted(bins, value, side="left")
        vals.cos2.loc[vals["cos2"] == value] = bins[idx]-bin_width[0]
         
        
    s125_fit = np.asarray(vals.s125.tolist())
    s125_fit_error = np.asarray(vals.s125_error.tolist())
    # introduce checks for intensity
    return vals




def get_attenuation_parameters(init_params, s125_fit, s125_fit_error, cos2):
    """Function that performs 

    Parameters
    ----------
    init_params     : list (float)
                      initial parameters
    s125_fit        : array (float)
                      s125 values to fit
    s125_fit_error  : array (float)
                      associated uncertainties
    cos2            : array (float)
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
    y = get_s125(cos2, a_true, b_true, f_true)

    # Recenter cosine**2 to 25 deg
    cos_ref = np.cos(math.radians(25))**2
    cos2 = cos2 - cos_ref
    # Minimize negative log-likelihood function
    #nll = lambda *args: -lnlike(*args)
    #result = sp.optimize.minimize(nll, [a_true, b_true, f_true], args=(cos2, y, s125_fit_error))

    # Prep for emcee
    ndim, nwalkers = 3, 50
    # Set starting positions wiggled normally around min nll estimate
    pos = [init_params+ np.random.randn(ndim)*0.2 for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(cos2, y, s125_fit_error), threads=10)

    # Run emcee, get samples
    sampler.run_mcmc(pos, 2000)
    all_samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    good_samples = []
    # Remove burn-in and samples outside of allowed prior bounds
    for i in range(0, 2000-50):
        if lnprior(all_samples[i]) != -np.inf:
            good_samples.append(all_samples[i])
    good_samples = np.asarray(good_samples)
  
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

from tqdm import tqdm


def provide_bootstrap_data(data, samples, bins, intensity):
    """Function that provides bootstrap data
        
    Parameters
    ----------
    data  : pandas array, initial data
           Pandas data array object
    samples : int
           number of of samples to be drawn
    bins: int 
          as usual number of bins
    intensity: int      
             the intensity at which the attenuation is obtained
    Returns
    -------
    result : pandas array
             values from the bootstrap
    """
  

    result = []
    for i in tqdm(range(samples)):
        rand_data = get_bootstrap_data(data)
        rand_data, groups = set_intensity(rand_data, bins)
        rand_vals = get_data_to_fit(rand_data, intensity, bins)
        result.append(rand_vals)    
        #print(rand_vals)
    return pd.concat(result)


def obtain_attenuation(data, n_bins, intensity, samples=200, doMCMC=False):

    results_dict = {}

    df = provide_bootstrap_data(data, samples, n_bins, intensity)
    groups = df.groupby(['cos2'])
    params_scipy, cov2 = sp.optimize.curve_fit(get_s125, groups.cos2.mean(), groups.s125.mean(),sigma = groups.s125.std())
    
    results_dict["params"] = params_scipy
    results_dict["err"] = np.sqrt(np.diag(cov2))

    sample_mcmc = []
    if (doMCMC):
        sample_mcmc = get_attenuation_parameters(params_scipy, groups.s125.mean(), groups.s125.std(), groups.cos2.mean())

    results_dict["mcmc"] = sample_mcmc

    return (results_dict, groups)
    
    
    
