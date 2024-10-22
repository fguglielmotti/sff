from scipy.stats import norm
from math import sqrt, exp
import scipy
import math
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
from statistics import mean
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.integrate import quad
from scipy.stats import t
from scipy.optimize import minimize
from scipy.special import kv, gamma  
from scipy.fft import fftshift, ifft


df1=4

def student_t_ES(df, loc=0, scale=1, ESlevel=0.05):
    return -scipy.stats.t.pdf(scipy.stats.t.ppf(ESlevel, df), df, loc, scale)/scipy.stats.t.cdf(scipy.stats.t.ppf(ESlevel, df), df, loc, scale)*(df+scipy.stats.t.ppf(ESlevel, df)**2)/(df-1)

true_ES = student_t_ES(df1)

sample_data = np.random.standard_t(df1, 500)

def mle_student_t(data):
    def neg_log_likelihood(params):
        df, loc, scale = params
        return -np.sum(t.logpdf(data, df, loc, scale))
    
    initial_params = [10, np.mean(data), np.std(data)]
    bounds = [(2, None), (None, None), (1e-6, None)]
    result = minimize(neg_log_likelihood, initial_params, bounds=bounds)
    return {'df': result.x[0], 'loc': result.x[1], 'scale': result.x[2]}

#Generate new set of data samples using the MLE parameters
mle_params = mle_student_t(sample_data)
df_mle, loc_mle, scale_mle = mle_params['df'], mle_params['loc'], mle_params['scale']
simulated_samples = t.rvs(df_mle, loc_mle, scale_mle, size=500)
print(simulated_samples)

#Compute the 90% CI for the ES for the simulated samples, generated using the MLE parameters
def parametric_bootstrap_CI(df, loc, scale, ESlevel=0.05, B=500, CI=0.90, n=500):
    ES_samples = []
    
    for _ in range(B):
        bootstrap_sample = t.rvs(df, loc, scale, size=n)
        VaR = np.percentile(bootstrap_sample, ESlevel * 100)
        ES = mean(bootstrap_sample[bootstrap_sample < VaR])
        ES_samples.append(ES)
    
    lower_bound = np.percentile(ES_samples, (1 - CI) / 2 * 100)
    upper_bound = np.percentile(ES_samples, (1 + CI) / 2 * 100)

    
    return lower_bound, upper_bound, ES_samples

parametric_ES_CI_lower, parametric_ES_CI_upper, parametric_ES_samples = parametric_bootstrap_CI(df_mle, loc_mle, scale_mle)

print(f"90% Confidence Interval via parametric Bootstrapping on simulated t-distribution for Expected Shortfall: ({parametric_ES_CI_lower:.4f}, {parametric_ES_CI_upper:.4f})")


#Q5c
#Choose with replacement 500 samples for each bootstrap and calculate the ES and VaR for each bootstrap, using only one set of data generated from the MLE t-distribution
def nonparametric_bootstrap_CI(data, ESlevel=0.05, B=500, CI=0.90, n=500):
    ES_samples = []
    
    for _ in range(B):
        bootstrap_sample = np.random.choice(data, n, replace=True)
        VaR = np.percentile(bootstrap_sample, ESlevel * 100)
        ES = mean(bootstrap_sample[bootstrap_sample < VaR])
        ES_samples.append(ES)
    
    lower_bound = np.percentile(ES_samples, (1 - CI) / 2 * 100)
    upper_bound = np.percentile(ES_samples, (1 + CI) / 2 * 100)

    return lower_bound, upper_bound, ES_samples

nonparametric_ES_CI_lower, nonparametric_ES_CI_upper, nonparametric_ES_samples = nonparametric_bootstrap_CI(simulated_samples)

print(f"90% Confidence Interval via nonparametric Bootstrapping on simulated t-distribution for Expected Shortfall: ({nonparametric_ES_CI_lower:.4f}, {nonparametric_ES_CI_upper:.4f})")


plt.figure(figsize=(12, 6))
plt.boxplot([nonparametric_ES_samples, parametric_ES_samples], labels=['Nonparametric Bootstrap ES', 'Parametric Bootstrap ES'])
plt.axhline(y=true_ES, color='r', linestyle='--', label=f'True ES (Using Bootstrap): {true_ES:.4f}')
plt.title('Comparison of Nonparametric and Parametric Bootstrap ES Values')
plt.ylabel('Expected Shortfall (ES)')
plt.legend()
plt.grid(True)
plt.show()


#%%
#Question 6 
#Run 1000 times and check coverage probability
def check_coverage_probability(df, loc, scale, true_ES, data, iterations=100):
    nonparametric_coverage = 0
    parametric_coverage = 0
    for _ in tqdm(range(iterations)):
        nonparametric_ES_CI_lower, nonparametric_ES_CI_upper, _ = nonparametric_bootstrap_CI(data)
        parametric_ES_CI_lower, parametric_ES_CI_upper, _ = parametric_bootstrap_CI(df, loc, scale)
        
        if nonparametric_ES_CI_lower <= true_ES <= nonparametric_ES_CI_upper:
            nonparametric_coverage += 1
        if parametric_ES_CI_lower <= true_ES <= parametric_ES_CI_upper:
            parametric_coverage += 1
            
    nonparametric_coverage_prob = nonparametric_coverage / iterations
    parametric_coverage_prob = parametric_coverage / iterations
    
    return nonparametric_coverage_prob, parametric_coverage_prob

#Checking the coverage of the 90% CI for the ES using the nonparametric and parametric bootstrap methods, given the true ES using Bootstrap method
nonparametric_coverage_prob, parametric_coverage_prob = check_coverage_probability(df_mle, loc_mle, scale_mle, true_ES, simulated_samples)

print(f"Nonparametric Bootstrap Coverage Probability: {nonparametric_coverage_prob:.4f}")
print(f"Parametric Bootstrap Coverage Probability: {parametric_coverage_prob:.4f}")