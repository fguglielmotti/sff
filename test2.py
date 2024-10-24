import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def calculate_true_es(df, loc=0, scale=1.0, es_level=0.05):
    if df <= 1:
        raise ValueError("Degrees of freedom (df) must be greater than 1")
    c = stats.t.ppf(es_level, df) #Critical value (quantile) beyond which we want to calculate expected shortfall
    phi_v_c = stats.t.pdf(c, df) #P.D.F. at c for the Student's t-dist.
    Phi_v_c = stats.t.cdf(c, df) #C.D.F. value at c for the Student's t-dist.
    es = - (phi_v_c / Phi_v_c) * ((df + c**2) / (df - 1)) #Formula
    return es * scale + loc #Scaling and location adjustment

def calculate_data_es(data, es_level=0.05): #Same as in 5a
    sorted_data = np.sort(data)
    cutoff_index = int(len(sorted_data) * es_level)
    VaR = sorted_data[cutoff_index] #VaR is the value at the cutoff index
    es = np.mean(sorted_data[:cutoff_index])
    return es

def nonparametric_bootstrap_ci(data, n=500, B=500, es_level=0.05, ci_level=0.90):
    bootstrap_es_values = []
    for _ in range(B):
        resample = np.random.choice(data, size=n, replace=True) #Sampling as discussed in 5a
        es = calculate_data_es(resample, es_level=es_level)
        bootstrap_es_values.append(es)

    #Confidence interval
    lower_bound = np.percentile(bootstrap_es_values, (1 - ci_level) / 2 * 100)
    upper_bound = np.percentile(bootstrap_es_values, (1 + ci_level) / 2 * 100)

    return lower_bound, upper_bound, bootstrap_es_values

def main(df=4, n=500, B=500, es_level=0.05, ci_level=0.90, sim=10):
    true_es = calculate_true_es(df, es_level=es_level)
    nonparam_convergence = []
    nonparam_ci_length = []

    for k in tqdm(range(25, 501, 1)):

        nonparametric_coverage_count = 0
        nonparametric_lengths = []
        n_values = []

        for i in range(sim):
            data = stats.t.rvs(df, loc=0, scale=1.0, size=n)

            lower_ci_nonparam, upper_ci_nonparam, nonparametric_es_values = nonparametric_bootstrap_ci(data, n=k, B=B, es_level=es_level, ci_level=ci_level)
            nonparametric_lengths.append(upper_ci_nonparam - lower_ci_nonparam)
            if lower_ci_nonparam <= true_es <= upper_ci_nonparam:
                nonparametric_coverage_count += 1

        nonparametric_coverage_prob = nonparametric_coverage_count / sim
        avg_nonparametric_length = np.mean(nonparametric_lengths)

        nonparam_convergence.append(nonparametric_coverage_prob)
        nonparam_ci_length.append(avg_nonparametric_length)
        n_values.append(k)

        print(f"Empirical Coverage Probability for Nonparametric Bootstrap for n = {k:.3f} CI: {nonparametric_coverage_prob:.3f}")
        print(f"Average Length of Nonparametric Bootstrap for n = {k:.3f} CI: {avg_nonparametric_length:.3f}")


    plt.figure(figsize=(12, 6))

    #Coverage
    plt.subplot(1, 2, 1)
    plt.plot(n_values, nonparam_convergence, label="Coverage Probability")
    plt.xlabel('n')
    plt.ylabel('Coverage Probability')
    plt.title('Nonparametric Bootstrap Coverage Probability vs n')
    plt.grid(True)
    plt.legend()

    #Average length of CI
    plt.subplot(1, 2, 2)
    plt.plot(n_values, nonparam_ci_length, label="Average CI Length", color='orange')
    plt.xlabel('n')
    plt.ylabel('Average CI Length')
    plt.title('Nonparametric Bootstrap CI Length vs n')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

main()