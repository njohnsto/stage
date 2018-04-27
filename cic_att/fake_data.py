import numpy as np
import scipy as sp
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from data_functions import generate_toy_data, get_bootstrap_data, set_intensity, get_data_to_fit, get_s125
from data_functions import get_attenuation_parameters, get_attenuation_parameters2
import corner


# Plotting defaults
fsize = 18


# input parameters for the simualtions
minE = 10**15 #eV
maxE = 10**18 #eV
gamma = -2.5 # slope of the spectrum

#relation between energy and s38
A = 10**12
B = 1.2

#attenuation true numbers
alpha = 0.919
beta = -1.13

# Maximum zenith angle in degrees
maxTheta = 50.

#number of events to generate
events = 10000
#number of bins for the zenith
n_bins = 8

data = generate_toy_data(events, minE, maxE, gamma, A, B, alpha, beta, maxTheta)


fig = plt.figure(figsize=(10, 6))
mpl.rc("font", family="serif", size=fsize)
ax = fig.add_subplot(111)
ax.grid(True)
ax.hist(data.cos2, bins=100)
ax.set_xlabel(r'$\cos^2(\theta)$')
ax.set_ylabel('Frequency')
fig.savefig('zenith_dist.png')


#### fit the attenuation curve
data = set_intensity(data, n_bins)
#print(data)

### input value the reference intesity
intensity = 100
s125_fit, bins2, s38_fit, s125_fit_error = get_data_to_fit(data, intensity, n_bins)
print(bins2)
values, cov = get_attenuation_parameters(s125_fit, bins2)
print(values)




a1 = values[0]
a2 = values[1]
a3 = values[2]
y  = get_s125(bins2, a1, a2, a3)
y2 = get_s125(bins2, alpha, beta, a3)

fig = plt.figure(figsize=(10, 6))
mpl.rc("font", family="serif", size=fsize)
ax = fig.add_subplot(111)
ax.grid(True)
ax.plot(bins2, s125_fit, lw=0, marker='o')
ax.plot(bins2, y, color="r")
ax.plot(bins2, y2)
ax.set_xlabel(r'$\cos^2(\theta)$')
ax.set_ylabel(r'$S_{125}$')
fig.savefig('first_fit_s125.png')


    
fig = plt.figure(figsize=(10, 6))
mpl.rc("font", family="serif", size=fsize)
ax = fig.add_subplot(111)
ax.grid(True)
ax.plot(np.log10(data.s125), data.I, lw=0, marker='o')
ax.set_yscale('log')
ax.set_xlim(1, 4)
ax.set_xlabel(r'$\log S_{125}$')
ax.set_ylabel(r'Intensity')
fig.savefig('intensity_vs_s125.png')



### number of boostrap samples as input
samples = 2

fig = plt.figure(figsize=(10, 6))
mpl.rc("font", family="serif", size=fsize)
ax = fig.add_subplot(111)
ax.grid(True)

a = [alpha, beta, 1]
bootstrap_values = []
bootstrap_values_2 = []
for j in range(0, samples):
    new_data = get_bootstrap_data(data)
    new_data=set_intensity(new_data, n_bins)
    s125_fit, bins, s38_fit, s125_fit_error = get_data_to_fit(new_data, intensity, n_bins)
    vals, cov_bt = get_attenuation_parameters(s125_fit, bins)
    bootstrap_values.append(vals)
    bootstrap_values_2.append(vals**2)
    ax.plot(bins, s125_fit, lw=0, marker='o')
    ax.plot(bins, get_s125(bins, vals[0], vals[1], vals[2]))
    print(vals)

ax.set_xlabel(r'$\log S_{125}$')
ax.set_ylabel(r'Intensity')
fig.savefig('bootstrap_s125_vs_zenith.png')




params_scipy, cov2 = sp.optimize.curve_fit(get_s125, bins, s125_fit)
# Use as guess
a_true = params_scipy[0]
b_true = params_scipy[1]
f_true = params_scipy[2]

sample = get_attenuation_parameters2(params_scipy, s125_fit, s38_fit, s125_fit_error, bins)




cos_ref = np.cos(math.radians(38))**2
cos2 = bins - cos_ref


fig = plt.figure(figsize=(10, 6))
mpl.rc("font", family="serif", size=fsize)
ax = fig.add_subplot(111)
ax.grid(True)

# Plot a subset of the samples
for a, b, f in sample[np.random.randint(len(sample), size=100)]:
    ax.plot(cos2, f * (b * cos2**2 + a * cos2 + 1), color="k", alpha=0.1)
    ax.plot(cos2, f_true * (b_true * cos2**2 + a_true * cos2 + 1), color="r", lw=2, alpha=0.8)
    ax.errorbar(cos2, y, yerr=s125_fit_error, fmt=".k")

ax.set_xlabel(r'$\log S_{125}$')
ax.set_ylabel(r'Intensity')
fig.savefig('mcmc_samples.png')



fig = plt.figure(figsize=(10, 6))
mpl.rc("font", family="serif", size=fsize)
ax = fig.add_subplot(111)
ax.grid(True)

fig = corner.corner(sample, labels=["$a$", "$b$", "$s38$"], truths=[alpha, beta, a3])
fig.savefig("corner_mcmc.png")




a_mcmc, b_mcmc, s38_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                            zip(*np.percentile(sample, [16, 50, 84],
                                                axis=0)))

#(mid_value, +error, -error)
print("a = %f + %f - %f\n" % (a_mcmc[0], a_mcmc[1], a_mcmc[2]))
print("b = %f + %f - %f\n" % (b_mcmc[0], b_mcmc[1], b_mcmc[2]))
print("s38 = %f + %f - %f\n" % (s38_mcmc[0], s38_mcmc[1], s38_mcmc[2]))
