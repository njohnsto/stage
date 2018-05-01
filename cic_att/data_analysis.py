import pandas as pd
import dask.dataframe as dd
import glob as glob
import numpy as np
from scipy.stats import kde

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import math
from data_functions import obtain_attenuation


plt.rc('font', size=16)
plt.rcParams['figure.figsize'] = (12.0, 10.0)    # resize plots

df = pd.DataFrame()
df = pd.read_hdf('/data/IceCube/merged2015.h5', key='data')
df['cos2'] = np.cos(df.zenith)**2
df['I'] = 0
df['s125_error'] = 0.1*df['s125']

df.drop(df.columns.difference(['s125', 's125_error',"cos2", "I","zenith"]), 1, inplace= True)
df= df.loc[df.s125>25]
df= df.loc[df.cos2>0.5]  
df.reset_index(inplace=True)
print('You have selected {} events'.format(df.s125.count()))

n_bins = 10
intensity = 200
samples = 100

fit_results, fitted_data = obtain_attenuation(df, n_bins, intensity, samples )

groups = fitted_data
sample = fit_results
cos_ref = np.cos(math.radians(38))**2
cos2 = np.linspace(0.5, 1, 20)-cos_ref

a_mcmc,  b_mcmc,  sref_mcmc  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                  zip(*np.percentile(sample, [16, 50, 84],
                                      axis=0)))

#(mid_value, +error, -error)
print("a = %f + %f - %f\n"%(a_mcmc[0],a_mcmc[1],a_mcmc[2]))
print("b = %f + %f - %f\n"%(b_mcmc[0],b_mcmc[1],b_mcmc[2]))
print("s_ref = %f + %f - %f\n"%(sref_mcmc[0],sref_mcmc[1],s38_mcmc[2]))

# Plot a subset of the samples
plt.errorbar(groups.cos2.mean()-cos_ref, groups.s125.mean(), yerr= groups.s125.std().tolist(), fmt=".k")
for a, b, f in sample[np.random.randint(len(sample), size=250)]:
    plt.plot(cos2, f * (b * cos2**2 + a * cos2 + 1), color="b", alpha=0.03)
    plt.plot(cos2, s38_mcmc[0] * (b_mcmc[0] * cos2**2 + a_mcmc[0] * cos2 + 1), color="b", lw=0.5, alpha=0.8)
plt.errorbar(groups.cos2.mean()-cos_ref, groups.s125.mean(), yerr= groups.s125.std().tolist(), fmt=".k")

