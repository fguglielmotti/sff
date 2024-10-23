from scipy.stats import norm
from math import sqrt, exp
import scipy
import math
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
from statistics import mean
import scipy.stats as stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.integrate import quad
from scipy.stats import t
from scipy.optimize import minimize
from scipy.special import kv, gamma  
from scipy.fft import fftshift, ifft


df1=10

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
    return result.x[0], result.x[1], result.x[2]

def expected_shortfall(data, ESlevel=0.05):
    sorted_data = np.sort(data)
    index = int(ESlevel * len(data))
    ES = np.mean(sorted_data[:index])
    return ES

#Compute the 90% CI for the ES for the simulated samples, generated using the MLE parameters
def parametric_bootstrap_CI(data, ESlevel=0.05, B=500, CI=0.90, n=500):
    #Generate new set of data samples using the MLE parameters
    df_mle, loc_mle, scale_mle = mle_student_t(data)
    ES_samples = []
    for _ in range(B):
        bootstrap_sample = t.rvs(df_mle, loc_mle, scale_mle, size=n)
        ES = expected_shortfall(bootstrap_sample)
        ES_samples.append(ES)
    
    lower_bound = np.percentile(ES_samples, (1 - CI) / 2 * 100)
    upper_bound = np.percentile(ES_samples, (1 + CI) / 2 * 100)

    return lower_bound, upper_bound, ES_samples

# Compute the 90% CI for the ES for the simulated samples, generated using the MLE parameters
parametric_ES_CI_lower, parametric_ES_CI_upper, parametric_ES_samples = parametric_bootstrap_CI(sample_data)

# Plot histogram for the distribution of the expected shortfalls
plt.figure(figsize=(12, 6))
plt.hist(parametric_ES_samples, bins=30, alpha=0.7, label='Parametric Bootstrap ES')
plt.axvline(x=true_ES, color='r', linestyle='--', label=f'True ES: {true_ES:.4f}')
plt.title('Distribution of Parametric Bootstrap Expected Shortfalls')
plt.xlabel('Expected Shortfall (ES)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

print(f"90% Confidence Interval via parametric Bootstrapping on simulated t-distribution for Expected Shortfall: ({parametric_ES_CI_lower:.4f}, {parametric_ES_CI_upper:.4f})")


#Q5c
#Choose with replacement 500 samples for each bootstrap and calculate the ES and VaR for each bootstrap
def nonparametric_bootstrap_CI(data, ESlevel=0.05, B=500, CI=0.90, n=500):
    ES_samples = []
    for _ in range(B):
        bootstrap_sample = np.random.choice(data, n, replace=True)
        ES = expected_shortfall(bootstrap_sample)
        ES_samples.append(ES)
    
    lower_bound = np.percentile(ES_samples, (1 - CI) / 2 * 100)
    upper_bound = np.percentile(ES_samples, (1 + CI) / 2 * 100)

    return lower_bound, upper_bound, ES_samples

nonparametric_ES_CI_lower, nonparametric_ES_CI_upper, nonparametric_ES_samples = nonparametric_bootstrap_CI(sample_data)

# Plot histogram for the distribution of the expected shortfalls
plt.figure(figsize=(12, 6))
plt.hist(nonparametric_ES_samples, bins=30, alpha=0.7, label='Nonparametric Bootstrap ES')
plt.axvline(x=true_ES, color='r', linestyle='--', label=f'True ES: {true_ES:.4f}')
plt.title('Distribution of Nonparametric Bootstrap Expected Shortfalls')
plt.xlabel('Expected Shortfall (ES)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

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
def check_coverage_probability(true_ES, df, n=500, iterations=500):
    nonparametric_coverage = 0
    parametric_coverage = 0
    len_parametric = []
    len_nonparametric =[]
    for _ in tqdm(range(iterations)):
        data=stats.t.rvs(df, loc=0, scale=1.0, size=n)
        nonparametric_ES_CI_lower, nonparametric_ES_CI_upper, _ = nonparametric_bootstrap_CI(data)
        parametric_ES_CI_lower, parametric_ES_CI_upper, _ = parametric_bootstrap_CI(data)
        len_parametric.append(parametric_ES_CI_upper - parametric_ES_CI_lower)
        len_nonparametric.append(nonparametric_ES_CI_upper - nonparametric_ES_CI_lower)
        if nonparametric_ES_CI_lower <= true_ES <= nonparametric_ES_CI_upper:
            nonparametric_coverage += 1
        if parametric_ES_CI_lower <= true_ES <= parametric_ES_CI_upper:
            parametric_coverage += 1

    nonparametric_coverage_prob = nonparametric_coverage / iterations
    parametric_coverage_prob = parametric_coverage / iterations
    avg_len_parametric = np.mean(len_parametric)
    avg_len_nonparametric = np.mean(len_nonparametric)
    
    return nonparametric_coverage_prob, parametric_coverage_prob, avg_len_parametric, avg_len_nonparametric

#Checking the coverage of the 90% CI for the ES using the nonparametric and parametric bootstrap methods, given the true ES using Bootstrap method
nonparametric_coverage_prob, parametric_coverage_prob, avg_len_parametric, avg_len_nonparametric = check_coverage_probability(true_ES, df1)

print(f"Nonparametric Bootstrap Coverage Probability: {nonparametric_coverage_prob:.4f}")
print(f"Parametric Bootstrap Coverage Probability: {parametric_coverage_prob:.4f}")
print(f"Average Length of Nonparametric Bootstrap CI: {avg_len_nonparametric:.4f}")
print(f"Average Length of Parametric Bootstrap CI: {avg_len_parametric:.4f}")

#Why does the question say we expect the parametric length to be shorter, but here we have parametric length longer than nonparametric length?
#This main issue is that I assumed that the nonparametric is carried out on the simulated dataset using MLE, but we should use the actual data set?