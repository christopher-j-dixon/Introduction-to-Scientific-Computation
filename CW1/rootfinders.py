"""
CW1 rootfinders.py

Author: Christopher Dixon
"""

import numpy as np
import matplotlib.pyplot as plt

def plot(f, Min, Max, n):
    """
    Plot the function f over the interval [Min, Max] with n points.

    Parameters
    ----------
    f : Function
        The function to be plotted.
    Min : Real number
        The minimum value of the interval.
    Max : Real number
        The maximum value of the interval.
    n : Integer
        The number of points in the interval.
    """
    x = np.linspace(Min, Max, num=n)
    plt.plot(x, f(x))  # Plot the function
    plt.plot(x, np.zeros(n), color='r')  # Plot the x-axis
    if Min * Max < 0:
        plt.plot(np.zeros(n), f(x), color='r')  # Plot the y-axis if it intersects with the x-axis

def bisection(f, a, b, Nmax):
    """
    Bisection Method: Returns a numpy array of the sequence of approximations 
    obtained by the bisection method.

    Parameters
    ----------
    f : Function
        Input function for which the zero is to be found.
    a : Real number
        Left side of interval.
    b : Real number
        Right side of interval.
    Nmax : integer
        Number of iterations to be performed.

    Returns
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """
    # Initialise the array with zeros
    p_array = np.zeros(Nmax)

    # Start loop
    for i in np.arange(Nmax):
        # Bisect the interval
        p = (a + b) / 2
        # Store the midpoint
        p_array[i] = p
        # Check the sign and update the interval
        if f(p) * f(a) > 0:
            a = p
        else:
            b = p

    # Return the array of approximations
    return p_array

def fixedpoint_iteration(g, p0, Nmax):
    """
    Fixed Point Iteration Method: Returns a numpy array of the sequence of 
    approximations obtained by the fixed point iteration method.

    Parameters
    ----------
    g : Function
        Input function for which the 'fixed point' is to be found. 
    p0 : Real number
        Initial estimation of the 'fixed point'.
    Nmax : Integer
        Number of iterations to be performed.

    Returns
    -------
    p_array : numpy.ndarray, shape (Nmax,)
        Array containing the sequence of approximations.
    """
    # Initialise the array with zeros
    p_array = np.zeros(Nmax)
    # Compute the initial approximation
    p_array[0] = g(p0)

    # Iterate to compute further approximations
    for i in range(1, Nmax):
        p_array[i] = g(p_array[i-1])

    # Return the array of approximations
    return p_array

def fixedpoint_iteration_stop(g, p0, Nmax, TOL):
    """
    Fixed Point Iteration Method with Stopping Criterion: Returns a numpy array 
    of the sequence of approximations obtained by the fixed point iteration method.

    Parameters
    ----------
    g : Function
        Input function for which the 'fixed point' is to be found. 
    p0 : Real number
        Initial estimation of the 'fixed point'.
    Nmax : Integer
        Number of iterations to be performed.
    TOL : Real number
        Tolerance stopping criterion.

    Returns
    -------
    p_array : numpy.ndarray
        Array containing the sequence of approximations.
    """
    # Initialise the list with the initial approximation
    p_array = [g(p0)]
    # Compute the second approximation
    p_array.append(g(p_array[0]))

    i = 1
    # Iterate until the maximum number of iterations or tolerance is met
    while i < Nmax and abs(p_array[i] - p_array[i - 1]) > TOL:
        i += 1
        p_array.append(g(p_array[i - 1]))

    # Return the array of approximations
    return np.array(p_array)

def newton_stop(f, dfdx, p0, Nmax, TOL):
    """
    Newton-Raphson Method with Stopping Criterion: Returns a numpy array of the 
    sequence of approximations obtained by the Newton-Raphson method.

    Parameters
    ----------
    f : Function
        Input function for which the zero is to be found. 
    dfdx : Function 
        The differential function (df(x)/dx) of the input function (f(x))
    p0 : Real number
        Initial estimation of the 'root point'
    Nmax : Integer
        Number of iterations to be performed.
    TOL : Real number
        Tolerance stopping criterion.
        
    Returns
    -------
    p_array : numpy.ndarray
        Array containing the sequence of approximations.
    """
    # Initialise the list with the initial approximation
    p_array = [p0 - f(p0) / dfdx(p0)]
    # Compute the second approximation
    p_array.append(p_array[0] - f(p_array[0]) / dfdx(p_array[0]))

    i = 1
    # Iterate until the maximum number of iterations or tolerance is met
    while i < Nmax and abs(p_array[i] - p_array[i - 1]) > TOL:
        i += 1
        p_array.append(p_array[i - 1] - f(p_array[i - 1]) / dfdx(p_array[i - 1]))

    # Return the array of approximations
    return np.array(p_array)

def plot_convergence(p, f, dfdx, g, p0, Nmax):
    """
    Plot the convergence behavior of fixed-point iteration and Newton-Raphson method.

    Parameters
    ----------
    p : Real number
        Exact solution for comparison.
    f : Function
        Function for Newton-Raphson method.
    dfdx : Function
        Derivative of function f.
    g : Function
        Function for fixed-point iteration.
    p0 : Real number
        Initial approximation.
    Nmax : Integer
        Number of iterations to be performed.
    """
    TOL = 10**(-16)

    # Fixed-point iteration
    p_array_fp = fixedpoint_iteration(g, p0, Nmax)
    e_array_fp = np.abs(p - p_array_fp)
    n_array_fp = 1 + np.arange(np.shape(p_array_fp)[0])

    # Newton-Raphson root iteration
    p_array_nr = newton_stop(f, dfdx, p0, Nmax, TOL)
    e_array_nr = np.abs(p - p_array_nr)
    n_array_nr = np.arange(np.shape(p_array_nr)[0]) + 1

    # Prepare figure
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("|p-p_n|")
    ax.set_title("Convergence behaviour")
    ax.grid(True)

    # Plot fixed-point iteration errors
    ax.plot(n_array_fp, e_array_fp, "o", label="FP iteration", linestyle="--")
    # Plot Newton-Raphson errors
    ax.plot(n_array_nr, e_array_nr, "o", label="NR iteration", linestyle="--")

    # Add legend
    ax.legend()
    return fig, ax

def optimize_FPmethod(f, c_array, p0, TOL): 
    """
    Optimize the fixed-point iteration method by finding the optimal value of c.

    Parameters
    ----------
    f : Function
        Input function for which the zero is to be found.
    c_array : numpy.ndarray
        Array of candidate c values.
    p0 : Real number
        Initial estimation of the 'fixed point'.
    TOL : Real number
        Tolerance stopping criterion.

    Returns
    -------
    c_opt : Real number
        Optimal value of c.
    n_opt : Integer
        Number of iterations for the optimal c.
    """
    # Define the transformation function
    def g(x, c):
        return x - c * f(x)
    
    size = np.zeros(len(c_array))

    # Iterate over each candidate c value
    for i, c in enumerate(c_array):
        p_array = fixedpoint_iteration_stop(lambda x: g(x, c), p0, 100, TOL)
        size[i] = len(p_array)
    
    # Find the optimal c and corresponding number of iterations
    c_opt = c_array[np.argmin(size)]
    n_opt = int(np.min(size))
    
    return c_opt, n_opt

# Check if the module is run as the main program

#%% Question 2
import numpy as np
import matplotlib.pyplot as plt
import rootfinders as rf

# Initialise
f = lambda x: x**3 + x**2 - 2*x - 2
a = 1
b = 2
Nmax = 5

# Run bisection
p_array = rf.bisection(f, a, b, Nmax)
print(p_array)

#%% Question 3
# Initialise
g = lambda x: 1 - 1/2 * x**2
p0 = 1
Nmax = 5

# Run fixed-point iteration
p_array = rf.fixedpoint_iteration(g, p0, Nmax)
print(p_array)

#%% Question 4
# Initialise
TOL = 10**(-2)

# Run fixed-point iteration with stopping criterion
p_array = rf.fixedpoint_iteration_stop(g, p0, Nmax, TOL)
print(p_array)

#%% Question 5
# Initialise
f = lambda x: np.cos(x) - x
dfdx = lambda x: -np.sin(x) - 1
p0 = 1
Nmax = 5
TOL = 10**(-3)

# Run Newton-Raphson method with stopping criterion
p_array = rf.newton_stop(f, dfdx, p0, Nmax, TOL)
print(p_array)

#%% Question 6
# Initialise
f = lambda x: x - np.cos(x)
dfdx = lambda x: 1 + np.sin(x)
g = lambda x: np.cos(x)
p0 = 1
Nmax = 20
p = np.float64(0.73908513321516064165531207047)

# Plot convergence
fig, ax = rf.plot_convergence(p, f, dfdx, g, p0, Nmax)

#%% Question 7
# Initialise
f = lambda x: x**3 + x**2 - 2*x - 2
p0 = 1
TOL = 10**(-6)
c_array = np.linspace(0.01, 0.4, 40)

# Find optimal c
c_opt, n_opt = rf.optimize_FPmethod(f, c_array, p0, TOL)
print(n_opt)
print(c_opt)

plt.show()
