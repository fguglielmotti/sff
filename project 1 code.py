import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

#PROBLEM 1
# Function to calculate the inversion formula for Student's t-distribution
def inversion_formula_t_density(df, x_values):
    density = []
    for x in x_values:
        # Using the inversion formula to approximate the density
        gamma_term = np.math.gamma((df + 1) / 2) / (np.math.gamma(df / 2) * np.sqrt(np.pi * df))
        density_value = gamma_term * (1 + (x**2 / df)) ** (- (df + 1) / 2)
        density.append(density_value)
    return np.array(density)

# Function to plot the densities
def plot_densities(df):
    if df <= 0:
        print("Degree of freedom must be positive.")
        return
    
    # Setting up the x-axis range depending on the degree of freedom
    if df > 5:
        x_values = np.linspace(-3, 3, 500)
    else:
        x_values = np.linspace(-6, 6, 500)
    
    # Exact t-distribution density
    exact_density = t.pdf(x_values, df)
    
    # Inversion formula density
    inversion_density = inversion_formula_t_density(df, x_values)
    
    # Plotting the densities
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, exact_density, label='Exact Density', linewidth=2.0, color='blue')
    plt.plot(x_values, inversion_density, label='Inversion Formula Density', linewidth=2.0, color='red', linestyle='--')
    plt.title(f'Student\'s t-Distribution Densities (df = {df})')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optional: plot relative percentage error
    relative_error = np.abs((exact_density - inversion_density) / exact_density) * 100
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, relative_error, label='Relative Percentage Error', color='green', linewidth=2.0)
    plt.title(f'Relative Percentage Error in Densities (df = {df})')
    plt.xlabel('x')
    plt.ylabel('Relative Error (%)')
    plt.grid(True)
    plt.show()

# Input and execution
df = 5
#df = float(input("Enter the degree of freedom (must be positive): "))
plot_densities(df)

#END OF PROBLEM 1

#PROBLEM 2
