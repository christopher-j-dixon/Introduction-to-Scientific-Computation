#################################################################
## Functions to carry out numerical integration
#################################################################

#################################################################
## Imports
## - No further imports should be necessary
## - If you wish to import non-standard modules, ask Ed if that 
## - is acceptable
#################################################################
import numpy as np
import matplotlib.pyplot as plt

#################################################################
## Functions to be completed by student
#################################################################

#%% Q1 code
def composite_trapezium(a, b, n, f):
    """
    To approximate the definite integral of the function f(x) over the interval [a,b]
    using the composite trapezium rule with n subintervals.   

    Parameters:
    -----------
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    n : int
        The number of subintervals to use in the composite trapezoidal rule.
    f : function
        The function to integrate.
    
    Returns
    -------
    integral_approx : float
        The approximation of the integral.

    Examples
    --------
    >>> integral_approx = composite_trapezium(0, 1, 10, lambda x: x**2)
    """
    
    x = np.linspace(a, b, n+1)  # Construct the quadrature points
    h = (b - a) / n

    # Construct the quadrature weights
    weights = h * np.ones(n+1)
    weights[[0, -1]] = h / 2

    integral_approx = np.sum(f(x) * weights)

    return integral_approx


#%% Q2a code
def romberg_integration(a, b, n, f, level):
    """
    To approximate the definite integral of the function f(x) over the interval [a,b]
    using Romberg integration with specified levels of extrapolation.   

    Parameters:
    -----------
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    n : int
        The number of subintervals to use in the composite trapezoidal rule.
    f : function
        The function to integrate.
    level : int
        The level of extrapolation.
    
    Returns
    -------
    integral_approx : float
        The approximation of the integral.
    """
    
    R = np.zeros((level, level), np.float64)
    
    for i in range(level):
        N = n * 2**i
        R[i, 0] = composite_trapezium(a, b, N, f)
        
        for j in range(1, i + 1):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
            
    integral_approx = R[-1, -1]
    
    return np.float64(integral_approx)


#%% Q2b code
def compute_errors(N, no_n, levels, no_levels, f, a, b, true_val):
    """
    Computes the errors of Romberg integration for different levels of extrapolation 
    and different numbers of subintervals where the true value of the integral is known.

    Integral of function: f(x) = 1/x over [1, 2]
    -------------------------------------------------
    - As the number of subintervals (n) increases, the error decreases, and the 
      integral approximation tends towards the true value.
    - As the level of extrapolation increases, the error also decreases. This 
      improves the accuracy of the integral approximation.

    Parameters:
    -----------
    N : list
        List of numbers of subintervals.
    no_n : int
        Number of different subintervals to test.
    levels : list
        List of levels of extrapolation.
    no_levels : int
        Number of different levels of extrapolation to test.
    f : function
        The function to integrate.
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    true_val : float
        The true value of the integral.
    
    Returns
    -------
    error_matrix : numpy.ndarray
        Matrix containing the approximation errors.
    fig : matplotlib.figure.Figure
        Figure showing the errors plotted against the number of subintervals.
    """
    
    error_matrix = np.zeros((no_levels, no_n))

    for i, level in enumerate(levels):
        for j, n in enumerate(N):
            integral_approx = romberg_integration(a, b, n, f, level)
            error = abs(true_val - integral_approx)
            error_matrix[i, j] = error

    fig, ax = plt.subplots()
    for i, level in enumerate(levels):
        ax.loglog(N, error_matrix[i], marker='o', label=f'Level {level}')
    ax.set_xlabel('Number of subintervals (n)')
    ax.set_ylabel('Error')
    ax.legend()
    plt.show()

    return np.float64(error_matrix), fig

################
#%% Q1 Test
################

# Initialise
f = lambda x: np.sin(x)
a = 0
b = np.pi
n = 10

# Run the function
integral_approx = composite_trapezium(a, b, n, f)

print("\n################################")
print("Q1 TEST OUTPUT:\n")
print("integral_approx =\n")
print(integral_approx)
print("")

################
#%% Q2a Test
################

# Initialise
f = lambda x: np.sin(x)
a = 0
b = np.pi
n = 2

print("\n################################")
print("Q2a TEST OUTPUT:\n")
print("")

# Test code
for level in range(1, 5):
    # Run test 
    integral_approx = romberg_integration(a, b, n, f, level)
    print(f"level = {level}, integral_approx = {integral_approx}")

################
#%% Q2b
################
N = [1, 2, 4, 8]
levels = [1, 2, 3, 4]
true_val = 2.0

error_matrix, fig = compute_errors(N, 4, levels, 4, f, a, b, true_val)

print("\n################################")
print("Q2b TEST OUTPUT:\n")
print("Error =\n")
print(error_matrix)
