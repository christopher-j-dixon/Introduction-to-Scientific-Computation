#################################################################
## Functions to find a range of approximations using polynomials
#################################################################

#################################################################
## Imports
## - No further imports should be necessary
## - If you wish to import non-standard modules, ask if that 
## - is acceptable
#################################################################
import numpy as np
import matplotlib.pyplot as plt
import lagrange_polynomials as lp  # previously written functions

#################################################################
## Functions to be completed by student
#################################################################

#%% Q2 code

def poly_interpolation(a, b, p, n, x, f, produce_fig):
    """
    Perform polynomial interpolation of a function f.

    Parameters
    ----------
    a : float
        Start of interval [a, b].
    b : float
        End of interval [a, b].
    p : int
        Degree of the polynomial interpolant.
    n : int
        Number of points in the evaluation set.
    x : numpy.ndarray
        Array of n evaluation points.
    f : function
        Function to interpolate.
    produce_fig : bool
        If True, produce a plot of the interpolation.

    Returns
    -------
    interpolant : numpy.ndarray
        Array of interpolated values.
    fig : matplotlib.figure.Figure or None
        Plot of the interpolation if produce_fig is True, otherwise None.
    """
    xhat = np.linspace(a, b, p+1)
    lag_matrix, error_flag = lp.lagrange_poly(p, xhat, n, x, tol=1.0e-10)
    
    if error_flag:
        raise ValueError("Nodal points are not distinct within the given tolerance.")
    
    fx = f(xhat)
    interpolant = np.sum(lag_matrix * fx[:, np.newaxis], axis=0)
    
    if produce_fig:
        fig, ax = plt.subplots()
        ax.plot(x, f(x), label='f(x)')
        ax.plot(x, interpolant, label='Interpolant')
        ax.set_title('f(x) vs Interpolant')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        return interpolant, fig
    else:
        return interpolant, None

#%% Q4 code

def poly_interpolation_2d(p, a, b, c, d, X, Y, n, m, f, produce_fig):
    """
    Perform 2D polynomial interpolation of a function f.

    Parameters
    ----------
    p : int
        Degree of the polynomial interpolant.
    a : float
        Start of interval [a, b] for x.
    b : float
        End of interval [a, b] for x.
    c : float
        Start of interval [c, d] for y.
    d : float
        End of interval [c, d] for y.
    X : numpy.ndarray
        Grid points in the x direction.
    Y : numpy.ndarray
        Grid points in the y direction.
    n : int
        Number of points in the x direction.
    m : int
        Number of points in the y direction.
    f : function
        Function to interpolate.
    produce_fig : bool
        If True, produce a contour plot of the interpolation.

    Returns
    -------
    interpolant : numpy.ndarray
        Array of interpolated values.
    fig : matplotlib.figure.Figure or None
        Contour plot of the interpolation if produce_fig is True, otherwise None.
    """
    xhat = np.linspace(a, b, p+1)
    yhat = np.linspace(c, d, p+1)
    
    lag_x, error_flag_x = lp.lagrange_poly(p, xhat, n, X[0, :], tol=1.0e-10)
    lag_y, error_flag_y = lp.lagrange_poly(p, yhat, m, Y[:, 0], tol=1.0e-10)
    
    if error_flag_x or error_flag_y:
        raise ValueError("Nodal points are not distinct within the given tolerance.")
    
    interpolant = np.zeros((m, n))
    for i in range(p+1):
        for j in range(p+1):
            interpolant += f(xhat[i], yhat[j]) * np.outer(lag_y[j], lag_x[i])
    
    if produce_fig:
        fig, ax = plt.subplots()
        contour = ax.contour(X, Y, interpolant)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('2D polynomial interpolation plot')
        fig.colorbar(contour, ax=ax)
        return interpolant, fig
    else:
        return interpolant, None

#%% Q6 code

def approximate_derivative(x, p, h, k, f):
    """
    Approximate the derivative of a function f at point x using polynomial interpolation.

    Parameters
    ----------
    x : float
        Point at which to evaluate the derivative.
    p : int
        Degree of the polynomial interpolant.
    h : float
        Step size.
    k : int
        Index of the nodal point that coincides with x.
    f : function
        Function whose derivative is to be approximated.

    Returns
    -------
    deriv_approx : numpy.float64
        Approximation of the derivative of f at x.
    """
    xhat = np.linspace(x - h*k, x - h*k + h*p, p+1, dtype=np.float64)
    lagrange_poly, error_flag = lp.deriv_lagrange_poly(p, xhat, 1, np.array([x]), tol=1.0e-10)
    
    if error_flag:
        raise ValueError("Nodal points are not distinct within the given tolerance.")
    
    return np.float64(np.dot(lagrange_poly.reshape(-1), f(xhat)))

#################################################################
## Test Code ##
## You are highly encouraged to write your own tests as well,
## but these should be written in a separate file
#################################################################
print("\nAny outputs above this line are due to importing lagrange_polynomials.py.\n")

################
#%% Q2 Test
################

# Initialise
a = 0.5
b = 1.5
p = 3
n = 10
x = np.linspace(0.5, 1.5, n)
f = lambda x: np.exp(x) + np.sin(np.pi * x)

# Run the function
interpolant, fig = poly_interpolation(a, b, p, n, x, f, True)

print("\n################################")
print('Q2 TEST OUTPUT:\n')
print("interpolant = \n")
print(interpolant)

################
#%% Q4 Test
################

f = lambda x, y: np.exp(x**2 + y**2)
n = 4
m = 3
a = 0
b = 1
c = -1
d = 1 
x = np.linspace(a, b, n)
y = np.linspace(c, d, m)
X, Y = np.meshgrid(x, y)

interpolant, fig = poly_interpolation_2d(11, a, b, c, d, X, Y, n, m, f, True)

print("\n################################")
print('Q4 TEST OUTPUT:\n')
print("interpolant = \n")
print(interpolant)

################
#%% Q6 Test
################

print("\n################################")
print("Q6 TEST OUTPUT:\n")

# Initialise
p = 3
h = 0.1
x = 0.5
f = lambda x: np.cos(np.pi * x) + x

for k in range(4):
    # Run test 
    deriv_approx = approximate_derivative(x, p, h, k, f)
    print("k = " + str(k) + ", deriv_approx = " + str(deriv_approx))