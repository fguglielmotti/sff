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
import random

#Question 1
def K(z,x):
    return scipy.special.kv(z,x, out=None)

def y_x(x, mu, delt):
    return sqrt(delt**2+(x-mu)**2)

def CFt(x, lam, alp, bet, delt, mu):
    return ((alp**2-bet**2)**(lam/2)*y_x(x, mu, delt)**(lam-1/2))/(sqrt(2*math.pi)*alp**(lam-1/2)*delt**lam*K(lam, delt*(sqrt(alp**2-bet**2))))*K(lam-1/2, alp*y_x(x, mu, delt))*exp(bet*(x-mu))

def plot_distributions(df):
    
    if df<0:
        return "ERROR: degrees of freedom cannot be negative."
    
    #Using the parameters from the book:
    lam = -df/2
    alp = 0.01
    bet = 0
    delt = sqrt(df)
    mu = 0
    x_values = np.linspace(-3,3,600) if df>25 else np.linspace(-6,6,600)
    x_size = 3 if df>25 else 6
    y_values = [CFt(x, lam, alp, bet, delt, mu) for x in x_values]
    t_values = [scipy.stats.t.pdf(x, df) for x in x_values]
    plt.figure(figsize=(10, x_size))
    plt.plot(x_values, y_values, label='CFt(x)', linewidth=2)
    plt.plot(x_values, t_values, label="Student's t-distribution", color='red', linewidth=2, linestyle= 'dotted')
    plt.title('Plot of CFt(x) and Student\'s t-distribution from ' + str(x_size) + ' to ' + str(x_size) + ', alpha = 0.01, df=' + str(df))
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()
    return

plot_distributions(20)

#Question 2
df1 = 20
df2 = 10
x_values = np.linspace(-3, 3, 600)

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

#c) This part need a bit of work I have no idea how to fix it
def integrand(t, x, df1, df2):
    lam1 = -df1 / 2
    lam2 = -df2 / 2
    alp = 0.01
    bet = 0
    delt1 = sqrt(df1)
    delt2 = sqrt(df2)
    mu = 0
    return np.exp(-1j * t * x) * CFt(t, lam1, alp, bet, delt1, mu) * CFt(t, lam2, alp, bet, delt2, mu)


def inverse_fourier_transform(x, df1, df2, a=-3, b=3, n=2400):
    t_values = np.linspace(a, b, n)
    integrand_values = [integrand(t, x, df1, df2) for t in t_values]
    integral = scipy.integrate.trapezoid(integrand_values, t_values)
    return integral.real

pdf_values = [inverse_fourier_transform(x, df1, df2) for x in tqdm(x_values)]

plt.figure(figsize=(10, 6))
plt.plot(x_values, pdf_values, label='PDF from Inverse Fourier Transform', linewidth=2, color='green')
plt.plot(x_values, conv_pdf_numerical, label='Convolution of two t-distributions (Numerical Integration)', linewidth=2, color='red')
sns.kdeplot(sum_samples, label='Simulated Sum of t-distributions', color='blue', linewidth=2)
plt.title('Comparison of PDF from Inverse Fourier Transform, Simulated Sum, and Numerical Integration of two t-distributions with df1=20 and df2=10')
plt.xlabel('x')
plt.ylabel('Probability')
plt.xlim(-3, 3)
plt.legend()
plt.grid(True)
plt.show()

#Question 3 ES and VaR page 446, defining losses as a negative value, hence ES = E[L|L<VaR]
def ESt(df, c): #(Here c is the VaR at quantile alpha)
    return -scipy.stats.t.pdf(c, df1)/scipy.stats.t.cdf(c, df1)*(df1+c**2)/(df1-1)

#VaR is the alpha quantile of the t-distribution:
def VaRt(df1, alpha):
    return scipy.stats.t.ppf(alpha, df1)

#print(VaRt(df1, 0.05), ESt(df1, VaRt(df1, 0.05)))

#Question 4
def student_t_ES(df, loc=0, scale=1, ESlevel=0.05):
    return -scipy.stats.t.pdf(scipy.stats.t.ppf(ESlevel, df), df, loc, scale)/scipy.stats.t.cdf(scipy.stats.t.ppf(ESlevel, df), df, loc, scale)*(df+scipy.stats.t.ppf(ESlevel, df)**2)/(df-1)

#e.g.
print("True ES (Using the formula) at df1=20:", student_t_ES(df1))

#Question 5 CHECK IF 5a IS CORRECT
#5a) Perform bootstrap on the t-distribution, then output ES from the samples?
def bootstrap_ES(df, loc=0, scale=1, ESlevel=0.05, n=500, B=500):
    t_values = np.random.standard_t(df, n)
    bootstrap_samples = np.random.choice(t_values, (B, n), replace=True)
    ES_samples = []
    for sample in bootstrap_samples:
        VaR = np.percentile(sample, ESlevel * 100) #5% of the data
        ES = mean(sample[sample < VaR]) #Average of the data below VaR
        ES_samples.append(ES)
    return mean(ES_samples)
print("The true ES (Using Bootstrap) at df1=20:", bootstrap_ES(df1))

#5b) Assume the underlying distribution is student T, find the 90% CI of ES

def mle_student_t(data):
    def neg_log_likelihood(params):
        df, loc, scale = params
        return -np.sum(t.logpdf(data, df, loc, scale))
    
    initial_params = [10, np.mean(data), np.std(data)]
    bounds = [(2, None), (None, None), (1e-6, None)]
    result = minimize(neg_log_likelihood, initial_params, bounds=bounds)
    return {'df': result.x[0], 'loc': result.x[1], 'scale': result.x[2]}

# Generate new set of data samples using the MLE parameters
sample_data = np.random.standard_t(df1, 500)

mle_params = mle_student_t(sample_data)
df_mle, loc_mle, scale_mle = mle_params['df'], mle_params['loc'], mle_params['scale']
simulated_samples = t.rvs(df_mle, loc_mle, scale_mle, size=500)

# Compute the 90% confidence interval for the expected shortfall for the simulated samples
def bootstrap_ES_CI(data, ESlevel=0.05, B=500, CI=0.90):
    n = len(data)
    ES_samples = []
    
    for _ in range(B):
        bootstrap_sample = np.random.choice(data, n, replace=True)
        VaR = np.percentile(bootstrap_sample, ESlevel * 100)
        ES = mean(bootstrap_sample[bootstrap_sample < VaR])
        ES_samples.append(ES)
    
    lower_bound = np.percentile(ES_samples, (1 - CI) / 2 * 100)
    upper_bound = np.percentile(ES_samples, (1 + CI) / 2 * 100)
    
    return lower_bound, upper_bound

ES_CI_lower, ES_CI_upper = bootstrap_ES_CI(simulated_samples)

print(f"90% Confidence Interval for Expected Shortfall: ({ES_CI_lower:.4f}, {ES_CI_upper:.4f})")



#Q5c
VaR = np.percentile(simulated_samples, 0.05)

ES = mean(simulated_samples[simulated_samples < VaR])

def nonparametric_bootstrap_CI(data, ESlevel=0.05, B=500):
    n = len(data)
    VaR_samples = []
    ES_samples = []
    
    for _ in range(B):
        bootstrap_sample = np.random.choice(data, n, replace=True)
        VaR = np.percentile(bootstrap_sample, ESlevel * 100)
        ES = mean(bootstrap_sample[bootstrap_sample < VaR])
        VaR_samples.append(VaR)
        ES_samples.append(ES)
    
    VaR_CI = np.percentile(VaR_samples, [5, 95])
    ES_CI = np.percentile(ES_samples, [5, 95])
    
    return VaR_CI, ES_CI, ES_samples

VaR_CI, ES_CI, nonparametric_ES_samples = nonparametric_bootstrap_CI(simulated_samples)