import numpy as np
import pandas as pd
import scipy as sp

import math

def random_pl(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)



def get_signal_ref(A,B,energy):
    return ( energy / A )**(1/B)




def get_s125(cos2, alpha, beta, signal_ref):
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
    bins = np.linspace(0, 1, number_of_bins, endpoint = True )
    ind = np.digitize(data['cos2'],bins)
    groups = data.groupby(ind)
                      
    for name, group in groups:
        values = group['s125'].apply(lambda x: group[group['s125']>x].count())
        data.loc[group.I.index.tolist(), 'I']= values.I 
    return data


def get_data_to_fit(data, intensity, nr_of_bins):
    bins = np.linspace(0.05,0.95, nr_of_bins, endpoint = True)
    val = data.loc[data.I == intensity]
    s125_fit = val.s125.tolist()    
    # introduce checks for intensity 
    return (s125_fit, bins)

def get_attenuation_parameters(s125, cos2, performMCMC):
    parameters, cov2 =sp.optimize.curve_fit(get_s125, cos2, s125)
    if performMCMC:
        #implement MCMC with initial parameters from curve_fit
        print("Life is amazing")
    return (parameters, cov2)    
        
def get_bootstrap_data(data):
    new_data = data.copy()
    new_data.s125 = np.random.normal(data.s125.tolist(), data.s125_error.tolist())  
    return new_data
    
    
    


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

    
    
    
def get_random_vars2(N,theta,S125):
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
    data=pd.DataFrame()
    data['E'] = E
    data['S_ref'] = S_i_ref
    data['cos2'] = np.cos(theta)**2
    data['S'] = S125
    data['th'] = np.arccos(np.sqrt(data.cos2))
    data['lgE'] = np.log10(data.E)
    data['lgS'] = np.log10(data.S)
    data['lgS_ref'] = np.log10(data.S_ref)
    data = data.sort_values(['lgS'])
    data['I'] = 0
    bins = np.linspace(0, 1, 11, endpoint = True )
    ind = np.digitize(data['cos2'],bins)
    groups = data.groupby(ind)
                      
    for name, group in groups:
        values = group['lgS'].apply(lambda x: group[group['lgS']>x].count())
        data.loc[group.I.index.tolist(), 'I']= values.I 
        
    data['sqrt_I']=np.sqrt(data.I)
    return data
