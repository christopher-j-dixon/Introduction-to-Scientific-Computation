#################################################################
## Functions to compute some approximation errors
#################################################################

#################################################################
## Imports
## - No further imports should be necessary
## - If you wish to import a non-standard modules, ask if that 
## - is acceptable
#################################################################
import numpy as np
import matplotlib.pyplot as plt
import approximations as approx  # import previously written functions
#################################################################

#################################################################
## Functions to be completed by student
#################################################################

#%% Q3 code

def interpolation_errors(a, b, n, P, f):
    """
    Computes the maximum interpolation error for a range of polynomial degrees.

    Parameters
    ----------
    a : float
        Start of the interval.
    b : float
        End of the interval.
    n : int
        Number of polynomial degrees.
    P : list or numpy.ndarray
        Degrees of the polynomials.
    f : function
        The function to interpolate.

    Returns
    -------
    error_matrix : numpy.ndarray of shape (n,)
        Array containing the maximum interpolation errors.
    fig : matplotlib.figure.Figure
        Figure showing the max interpolation error for different polynomial degrees.

    Examples
    --------
    >>> error_matrix, fig = interpolation_errors(0, 1, 10, np.arange(1, 11), lambda x: np.exp(2*x))
    """
    x = np.linspace(a, b, 2000)
    error_matrix = np.zeros(n)
    
    for i, p in enumerate(P):
        interpolant, _ = approx.poly_interpolation(a, b, p, n, x, f, produce_fig=False)
        fx = f(x)
        error_matrix[i] = np.max(np.abs(interpolant - fx))
    
    fig, ax = plt.subplots()
    ax.semilogy(P, error_matrix)
    ax.set_xlabel('Polynomial degree')
    ax.set_ylabel('Max error')
    ax.set_title('Max interpolation error for different polynomial degrees')
        
    return error_matrix, fig

#%% Q7 code 

def derivative_errors(x, P, m, H, n, f, fdiff):
    """
    Computes the error in the derivative approximation for various polynomial degrees and step sizes.

    Parameters
    ----------
    x : float
        Point at which to evaluate the derivative.
    P : list or numpy.ndarray
        Degrees of the polynomials.
    m : int
        Number of polynomial degrees.
    H : list or numpy.ndarray
        Step sizes.
    n : int
        Number of step sizes.
    f : function
        The function to differentiate.
    fdiff : function
        The exact derivative of the function f.

    Returns
    -------
    E : numpy.ndarray of shape (m, n)
        Matrix containing the errors in the derivative approximation.
    fig : matplotlib.figure.Figure
        Figure showing the error plot for different polynomial degrees and step sizes.

    Examples
    --------
    >>> E, fig = derivative_errors(0, np.array([2, 4, 6]), 3, np.array([1/4, 1/8, 1/16]), 3, lambda x: np.exp(2*x), lambda x: 2*np.exp(2*x))
    """
    E = np.zeros((m, n))
    fig, ax = plt.subplots()
    
    for i, p in enumerate(P):
        h_values = []
        E_values = []
    
        for j, h in enumerate(H):
            app = approx.approximate_derivative(x, p, h, p // 2, f)
            exact_diff = fdiff(x)
            error = np.abs(exact_diff - app)
            E[i][j] = error
            h_values.append(h)
            E_values.append(error)
    
        ax.loglog(h_values, E_values, marker='o', label=f'p={p}')
    
    ax.set_xlabel('Step size (h)')
    ax.set_ylabel('Error (E)')
    ax.set_title('Error plot for different polynomial degrees')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    
    return E, fig

#################################################################
## Test Code ##
## You are highly encouraged to write your own tests as well,
## but these should be written in a separate file
#################################################################
print("\nAny outputs above this line are due to importing approximations.py.\n")

################
#%% Q3 Test
################

# Initialise
n = 5
P = np.arange(1, n+1)
a = -1
b = 1
f = lambda x: 1 / (x + 2)

# Run the function
error_matrix, fig = interpolation_errors(a, b, n, P, f)

print("\n################################")
print('Q3 TEST OUTPUT:\n')

print("error_matrix = \n")
print(error_matrix)


################
#%% Q7 Test
################

# Initialise
P = np.array([2, 4, 6])
H = np.array([1/4, 1/8, 1/16])
x = 0
f = lambda x: 1 / (x + 2)
fdiff = lambda x: -1 / ((x + 2) ** 2)

# Run the function
E, fig = derivative_errors(x, P, 3, H, 3, f, fdiff)

print("\n################################")
print("Q7 TEST OUTPUT:\n")

print("E = \n")
print(E)
plt.show()
