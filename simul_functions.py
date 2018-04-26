import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize

import matplotlib.pyplot as plt
import math

#functions needed for stage
def random_pl(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)



def get_signal_ref(A,B,energy):
    return ( energy / A )**(1/B)




def get_s125(cos2, alpha, beta, signal_ref):
    #function for computing s125
    cos_ref = np.cos(math.radians(38))**2
    x = cos2 - cos_ref
    return signal_ref*(1 + alpha*x + beta*x**2)



def generate_toy_data(events, minE, maxE, gamma, A, B, alpha, beta):
    energy = random_pl(minE, maxE, gamma, events) 
    s38 = get_signal_ref(A, B, energy)
    cos2 = np.random.rand(events) 
    s125 = get_s125(cos2, alpha, beta, s38)

    data=pd.DataFrame()
    data['energy'] = energy
    data['cos2'] = cos2
    data['s38'] = s38
    data['zenith'] = np.arccos(np.sqrt(data.cos2))
    data['zenith_er'] = np.random.uniform(math.radians(0.5), math.radians(1.5), events)
    data['s125'] = s125
    data['s125_error'] = np.abs(np.random.uniform(0.05, 0.1, events)*s125)
    data['I'] = 0
    return data

def get_positive_random(value, error):
    vals = -10    
    while value.any()<0:
        vals = np.random.normal(value, error)
    return vals        

def set_intensity(data, number_of_bins):
    #function that calculates the intensity for each group  
    data = data.sort_values(['s125'])
    min_cos2 = data.cos2.min() 
    bins = np.linspace(min_cos2, 1, number_of_bins, endpoint = True )
    ind = np.digitize(data['cos2'],bins)
    groups = data.groupby(ind)
                      
    for name, group in groups:
        values = group['s125'].apply(lambda x: group[group['s125']>x].count())
        data.loc[group.I.index.tolist(), 'I']= values.I 
    return data


def get_data_to_fit(data, intensity, nr_of_bins):
    bins = np.linspace(0.05,0.95, nr_of_bins, endpoint = True)
    val = data.loc[data.I == intensity]
    s125_fit = np.asarray(val.s125.tolist())
    s38_fit=np.asarray(val.s38.tolist())
    s125_fit_error=np.asarray(val.s125_error.tolist())
    # introduce checks for intensity 
    return (s125_fit, bins, s38_fit, s125_fit_error )

def get_attenuation_parameters(s125, cos2, performMCMC):
    parameters, cov2 =sp.optimize.curve_fit(get_s125, cos2, s125)
    if performMCMC:
        #implement MCMC with initial parameters from curve_fit
        print("Life is amazing")
    return (parameters, cov2)

def lnlike(params, cos2, y,f,yerr):
    a,b=params
    model = f*(b*cos2**2 + a*cos2+1)
    return -0.5*np.sum((y-model)**2/yerr**2 + np.log(yerr**2))

import scipy.optimize as op

def lnprior(params):
    a,b=params
    if -2.0<b<0. and 0.<a<1.5:
        return 0.0
    return -np.inf

def lnprob(params, cos2, y,f, yerr):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params,cos2,y,f,yerr)

def get_attenuation_parameters2(s125_fit,s38_fit,s125_fit_error, bins, performMCMC):
    parameters, cov2 =sp.optimize.curve_fit(get_s125, bins, s125_fit)
    ndim, nwalkers = 2, 100
    a2=parameters[0]
    b2=parameters[1]
    #alpha=0.919
    #beta=-1.13
    y=get_s125(bins, a2,b2, s38_fit)
    nll = lambda *args: -lnlike(*args)
    result= op.minimize(nll, [a2,b2], args=(bins, y,s38_fit,s125_fit_error))
    a_ml, b_ml= result["x"]
    import emcee
    if performMCMC:
        #emcee
        pos = [result["x"]*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(bins,y,s38_fit,s125_fit_error))

        sampler.run_mcmc(pos, 500)
        sample= sampler.chain[:, 50:, :].reshape((-1,ndim))
        print("Life is amazing")
        for a, b in sample[np.random.randint(len(sample), size=100)]:
            plt.plot(bins, s38_fit*(b*bins**2+a*bins+1), color="k", alpha=0.1)
        plt.plot(bins, s38_fit*(b2*bins**2+a2*bins+1), color="r", lw=2, alpha=0.8)
        plt.errorbar(bins, y, yerr=s125_fit_error, fmt=".k")
        
    return (parameters, cov2, sample)
        
def get_bootstrap_data(data):
    new_data = data.copy()
    new_data.s125 = np.random.normal(data.s125.tolist(), data.s125_error.tolist())  
    return new_data
    
def bootstrap_graphs(bootstrap_values, alpha, beta):
    bootstrap_val=np.asarray(bootstrap_values).transpose()

    fig, axes=plt.subplots(nrows=2,ncols=3, figsize=(10,5))
    ax0,ax1,ax2,ax3,ax4,ax5 =axes.flatten()
    
    ax0.hist(bootstrap_val[0], bins=10, normed=False)
    ax0.hist(alpha, bins=70, normed=False)
    ax0.set_title('alpha_bootstrap')
    ax1.hist(bootstrap_val[1], bins=10, normed=False)
    ax1.hist(beta, bins=70, normed=False)
    ax1.set_title('beta_bootstrap')
    ax2.hist(bootstrap_val[2], bins=10, normed=False)
    ax2.set_title('signal_ref_bootstrap')

    ax3.plot(bootstrap_val[1],bootstrap_val[0],lw=0,marker='o')
    ax3.plot(beta,alpha, lw=0, marker='o')
    ax3.set_xlabel('beta_bootstrap')
    ax3.set_ylabel('alpha_bootstrap')
    ax4.plot(bootstrap_val[2],bootstrap_val[0],lw=0,marker='o')
    ax4.set_xlabel('signal_ref_bootstrap')
    ax4.set_ylabel('alpha_bootstrap')
    ax5.plot(bootstrap_val[2],bootstrap_val[1],lw=0,marker='o')
    ax5.set_xlabel('signal_ref_bootstrap')
    ax5.set_ylabel('beta_bootstrap')
    fig.tight_layout()
    plot=plt.figure()
    plt.show()
    return plot

def rms_for_initial_vals(bootstrap_values, bootstrap_values_2):
    values_array=np.asarray(bootstrap_values)
    values_2=np.asarray(bootstrap_values_2)
    values_array_T=values_array.transpose()
    values_2_T=values_2.transpose()

    mean_values=[]
    sigma=[]
    rms=[]
    nb_columns=len(values_array_T)
    for i in range(0,nb_columns):
        mean_values.append(np.mean(values_array_T[i]))
        rms.append(np.sqrt(np.sum(values_2_T[i])/len(values_2_T[i]))) 
        sigma.append(np.sqrt(rms[i]**2-mean_values[i]**2))
    return sigma

def get_random_vars(N):
    E0 = 10**15
    E1 = 10**18
    gamma = -2.5

    A = 10**12
    B = 1.2
    b=0.919
    a=-1.13
    c=1

    E = rndm(E0, E1, gamma, N) 
    
    S_i_ref=S_ref(A,B,E)
    cos_2=np.random.rand(N)
    S=S_i(a,b,c,cos_2,S_i_ref)
    data=pd.DataFrame()
    data['E'] = E
    data['S_ref'] = S_i_ref
    data['cos2'] = cos_2
    data['S'] = S
    data['th'] = np.arccos(np.sqrt(data.cos2))
    data['lgE'] = np.log10(data.E)
    data['lgS'] = np.log10(data.S)
    data['lgS_ref'] = np.log10(data.S_ref)
    data = data.sort_values(['lgS'])
    data['I'] = 0
    return data

    
    
    

