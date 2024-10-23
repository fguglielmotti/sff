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
import scipy.stats as stats
from scipy.stats import t
from scipy.optimize import minimize
from scipy.special import kv, gamma  
from scipy.fft import fftshift, ifft

random.seed(1234)
np.random.seed(1234)

#%%

#Question 1
#Characteristic function found in the Gaunt article (already cited in overleaf)
def cf_student_t(t, nu):
    numerator = kv(nu / 2, np.sqrt(nu) * np.abs(t)) * (np.sqrt(nu) * np.abs(t)) ** (nu / 2)
    denominator = gamma(nu / 2) * (2 ** (nu / 2 - 1))    
    return numerator / denominator

def integrand(t, x, df):
    return (1/(math.pi*2))*np.exp(-1j * t * x) * cf_student_t(t, df)


def inverse_fourier_transform(x, df, a=-3, b=3, n=600):
    t_values = np.linspace(a, b, n)
    integrand_values = [integrand(t, x, df) for t in t_values]
    integral = scipy.integrate.trapezoid(integrand_values, t_values)
    return integral.real

def plot_distributions(df):
    
    if df<0:
        return "ERROR: degrees of freedom cannot be negative."

    x_values = np.linspace(-3,3,600) if df>25 else np.linspace(-6,6,600)
    x_size = 3 if df>25 else 6
    y_values = [inverse_fourier_transform(x, df) for x in tqdm(x_values)]
    t_values = [scipy.stats.t.pdf(x, df) for x in x_values]
    plt.figure(figsize=(10, x_size))
    plt.plot(x_values, y_values, label='Inversion Formula PDF', linewidth=2)
    plt.plot(x_values, t_values, label="Student's t-distribution", color='red', linewidth=2, linestyle= 'dotted')
    plt.title('Plot of Inversion Formula PDF and Student\'s t-distribution from ' + str(x_size) + ' to ' + str(x_size) + ', alpha = 0.01, df=' + str(df))
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()
    diff = [(y - t) for y, t in zip(y_values, t_values)]
    plt.plot(x_values, diff, label = 'Distance', color='green',linewidth=2)
    plt.title('Plot of the Error of the Estimated Distribution')
    plt.show()
    return

plot_distributions(20)

#%%
#Question 2
df1 = 10
df2 = 20
x_values = np.linspace(-3, 3, 600)

#a)
def joint_pdf(x, y, df1, df2):
    return scipy.stats.t.pdf(x, df1) * scipy.stats.t.pdf(y, df2)

def integrand(s, df1, df2):
    return lambda x: joint_pdf(x, s - x, df1, df2)

def numerical_integration(func, a, b, n=10000):
    x = np.linspace(a, b, n+1)
    y = func(x)
    h = (b - a) / n
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral

conv_pdf_numerical = [numerical_integration(integrand(s, df1, df2), -3, 3) for s in x_values]

#b)
n = 500000
t1_samples = np.random.standard_t(df1, n)
t2_samples = np.random.standard_t(df2, n)
sum_samples = t1_samples + t2_samples

#c)
def integrand_joint(t, x, df1, df2):
    return (1/(math.pi*2))*np.exp(-1j * t * x) * cf_student_t(t, df1) * cf_student_t(t, df2)#CFt(t, lam1, alp, bet, delt1, mu) * CFt(t, lam2, alp, bet, delt2, mu) #This is not the CF of t, need to find it somewhere in his book

def inverse_fourier_transform_joint(x, df1, df2, a=-3, b=3, n=600):
    t_values = np.linspace(a, b, n)
    integrand_values = [integrand_joint(t, x, df1, df2) for t in t_values]
    integral = scipy.integrate.trapezoid(integrand_values, t_values)
    return integral.real

pdf_values = [inverse_fourier_transform_joint(x, df1, df2) for x in tqdm(x_values)]

plt.figure(figsize=(10, 6))
plt.plot(x_values, pdf_values, label='PDF from Inverse Fourier Transform', linewidth=3, color='green')
plt.plot(x_values, conv_pdf_numerical, label='Convolution of two t-distributions (Numerical Integration)', linewidth=3, color='red', linestyle='dotted')
sns.kdeplot(sum_samples, label='Simulated Sum of t-distributions', color='blue', linewidth=3, linestyle='dotted')
plt.title('Comparison of PDF from Inverse Fourier Transform, Simulated Sum, and Numerical Integration of two t-distributions with df1=20 and df2=10')
plt.xlabel('x')
plt.ylabel('Probability')
plt.xlim(-3, 3)
plt.legend()
plt.grid(True)
plt.show()

#%%
#Question 3 ES and VaR page 446, defining losses as a negative value, hence ES = E[L|L<VaR]
def ESt(df, c): #(Here c is the VaR at quantile alpha)
    return -scipy.stats.t.pdf(c, df1)/scipy.stats.t.cdf(c, df1)*(df1+c**2)/(df1-1)

#VaR is the alpha quantile of the t-distribution:
def VaRt(df1, alpha):
    return scipy.stats.t.ppf(alpha, df1)

#print(VaRt(df1, 0.05), ESt(df1, VaRt(df1, 0.05)))

#%%
#Question 4
def student_t_ES(df, loc=0, scale=1, ESlevel=0.05):
    return -scipy.stats.t.pdf(scipy.stats.t.ppf(ESlevel, df), df, loc, scale)/scipy.stats.t.cdf(scipy.stats.t.ppf(ESlevel, df), df, loc, scale)*(df+scipy.stats.t.ppf(ESlevel, df)**2)/(df-1)
true_ES = student_t_ES(df1)

print("True ES (Using the formula) at df1:", true_ES)

#%%
#Question 5 CHECK IF 5a IS CORRECT, because if we use bootstrap, unless we have large samples and large bootstrap replications, we will not get the true ES

sample_data = np.random.standard_t(df1, 500)

#5a) Perform bootstrap on the t-distribution, then output (true ES) of the samples
def bootstrap_ES(df, n=500, B=500):
    ES_samples = []
    for _ in range(B):
        sample_data = np.random.standard_t(df, n)
        VaR = np.percentile(sample_data, 5)
        ES = mean(sample_data[sample_data < VaR])
        ES_samples.append(ES)
    return mean(ES_samples)

print("The true ES (Using Bootstrap) at df1:", bootstrap_ES(df1))

# def bootstrap_ES(df, t_values, loc=0, scale=1, ESlevel=0.05, n=500, B=500):
#     if df<=1:
#         return "Degrees of freedom must be above 1"
#     bootstrap_samples = np.random.choice(t_values, (B, n), replace=True)
#     ES_samples = []
#     for sample in bootstrap_samples:
#         VaR = np.percentile(sample, ESlevel * 100)
#         ES = mean(sample[sample < VaR])
#         ES_samples.append(ES)
#     return np.percentile(ES_samples, ESlevel * 100)

# print("The true ES (Using Bootstrap) at df1=20:", bootstrap_ES(df1, sample_data))

#5b) Assume the underlying distribution is student T, find the 90% CI of ES
def student_t_ES(df, loc=0, scale=1, ESlevel=0.05):
    return -scipy.stats.t.pdf(scipy.stats.t.ppf(ESlevel, df), df, loc, scale)/scipy.stats.t.cdf(scipy.stats.t.ppf(ESlevel, df), df, loc, scale)*(df+scipy.stats.t.ppf(ESlevel, df)**2)/(df-1)

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

# Example to compute the 90% CI for the ES for the simulated samples, generated using the MLE parameters generated by sample data
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

# Example to compute the 90% CI for the ES using nonparametric bootstrapping on the sample data (True T-distribution)
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
plt.boxplot([nonparametric_ES_samples, parametric_ES_samples], tick_labels=['Nonparametric Bootstrap ES', 'Parametric Bootstrap ES'])
plt.axhline(y=true_ES, color='r', linestyle='--', label=f'True ES (Using Bootstrap): {true_ES:.4f}')
plt.title('Comparison of Nonparametric and Parametric Bootstrap ES Values')
plt.ylabel('Expected Shortfall (ES)')
plt.legend()
plt.grid(True)
plt.show()


#%%
#Question 6 #Still need a little bit of work or explaination
#Run 1000 times and check coverage probability
def check_coverage_probability(true_ES, df, n=500, iterations=1000):
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

#THIS MARC MF CANT EXPLAIN SHIT <- delete this

#However, this is not what he explain today in class where we generate a shit ton of MLEs, but rather what we thought of in the beginning