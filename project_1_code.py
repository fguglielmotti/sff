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

#Question 1
def K(z,x):
    return scipy.special.kv(z,x, out=None)

def y_x(x, mu, delt):
    return sqrt(delt**2+(x-mu)**2)

def CFt(x, lam, alp, bet, delt, mu):
    return ((alp**2-bet**2)**(lam/2)*y_x(x, mu, delt)**(lam-1/2))/(sqrt(2*math.pi)*alp**(lam-1/2)*delt**lam*K(lam, delt*(sqrt(alp**2-bet**2))))*K(lam-1/2, alp*y_x(x, mu, delt))*exp(bet*(x-mu))

# Parameters given in the book
df = 20
lam = -df/2
alp = 0.01
bet = 0
delt = sqrt(df)
mu = 0

x_values = np.linspace(-3, 3, 600)
y_values = [CFt(x, lam, alp, bet, delt, mu) for x in x_values]
t_values = [scipy.stats.t.pdf(x, df) for x in x_values]

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='CFt(x)', linewidth=2)
plt.plot(x_values, t_values, label="Student's t-distribution", color='red', linewidth=2, linestyle='--')
plt.title('Plot of CFt(x) and Student\'s t-distribution from -3 to 3 with alpha = 0.5, df=20 (Cannot see difference with smaller alpha)')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend() 
plt.grid(True)
plt.show()

#Question 2
df1 = 20
df2 = 10

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
print("True ES at df1=20:", student_t_ES(df1))

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
print("Bootstrapped ES at df1=20:", bootstrap_ES(df1))

#5b)