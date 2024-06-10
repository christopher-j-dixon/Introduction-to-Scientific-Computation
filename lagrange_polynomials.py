#################################################################
## Functions to compute Lagrange polynomials
#################################################################

#################################################################
## Imports
## - No further imports should be necessary
## - If you wish to import non-standard modules, ask if that 
## - is acceptable
#################################################################
import numpy as np

#################################################################
## Functions to be completed by student
#################################################################

#%% Q1 code
def lagrange_poly(p, xhat, n, x, tol):
    """
    Evaluates at the points x, the p+1 Lagrange polynomials associated with the 
    nodal/interpolating points xhat.

    Parameters
    ----------
    p : int
        Degree of the polynomial (assumed positive).
    xhat : numpy.ndarray of shape (p+1,)
        Nodal points upon which the Lagrange polynomials are defined.
    n : int or integer array
        Number of points at which to evaluate the interpolant.
        If n is int, n0=n, else n0 = n[0], n1 = n[1], etc.
    x : numpy.ndarray of shape (n0, n1, ...)
        Points at which to evaluate the interpolant.
    tol : float
        Tolerance for which floating point numbers x and y are 
        considered equal: |x-y| < tol.

    Returns
    -------
    lagrange_matrix : numpy.ndarray of shape (p+1, n0, n1, ...)
        Matrix of evaluated Lagrange polynomials.
    error_flag : int
        0 if points are distinct, 1 if points are not distinct (error).

    Examples
    --------
    >>> lagrange_matrix, error_flag = lagrange_poly(3, np.array([-1,0,1,2]), 5, np.linspace(5), 1.0e-10)
    """
    if xhat.shape != (p+1,):
        return None, None  # Premature exit if xhat is not of correct length

    l_matrix_shape = np.concatenate(([p+1], x.shape))
    lagrange_matrix = np.ones(l_matrix_shape)  # Preallocate for speed
    error_flag = 0  # Initially, no error

    # Build up the polynomials one term at a time
    for k, xhat_k in enumerate(xhat):
        for m, xhat_m in enumerate(xhat):
            if m != k:  # Make sure we don't divide by zero
                if np.abs(xhat_k - xhat_m) < tol:  # Nodes regarded as equal
                    error_flag = 1
                    return lagrange_matrix, error_flag  # Immediate return
                
                lagrange_matrix[k] *= (x - xhat_m) / (xhat_k - xhat_m)  # Update lagrange matrix

    return lagrange_matrix, error_flag

#%% Q5 code
def deriv_lagrange_poly(p, xhat, n, x, tol):
    """
    Evaluates the derivatives of the Lagrange polynomials at the points x.

    Parameters
    ----------
    p : int
        Degree of the polynomial (assumed positive).
    xhat : numpy.ndarray of shape (p+1,)
        Nodal points upon which the Lagrange polynomials are defined.
    n : int or integer array
        Number of points at which to evaluate the interpolant.
    x : numpy.ndarray of shape (n,)
        Points at which to evaluate the interpolant.
    tol : float
        Tolerance for which floating point numbers x and y are 
        considered equal: |x-y| < tol.

    Returns
    -------
    deriv_lagrange_matrix : numpy.ndarray of shape (p+1, n)
        Matrix of evaluated derivatives of Lagrange polynomials.
    error_flag : int
        0 if points are distinct, 1 if points are not distinct (error).

    Examples
    --------
    >>> deriv_lagrange_matrix, error_flag = deriv_lagrange_poly(3, np.array([-1,0,1,2]), 6, np.linspace(-1,1,6), 1.0e-10)
    """
    if (not np.allclose(np.unique(xhat), xhat, rtol=0, atol=tol)) or (len(xhat) != p+1):
        error_flag = 1
        return np.zeros((p+1, n)), error_flag
    
    error_flag = 0
    deriv_lagrange_matrix = np.zeros((p+1, n))
    
    for i, xhat_i in enumerate(xhat):
        numerator_summation = 0
        
        for j, xhat_j in enumerate(xhat):
            if j != i: 
                numerator = np.prod([(x - xhat_k) for k, xhat_k in enumerate(xhat) if k != i and k != j], axis=0)
                denominator = xhat_i - xhat_j
                
                if np.abs(xhat_i - xhat_j) < tol:
                    error_flag = 1
                    return np.zeros((p+1, n)), error_flag
                
                numerator_summation += numerator / denominator
        
        deriv_lagrange_matrix[i] = numerator_summation
    
    return deriv_lagrange_matrix, error_flag

#################################################################
## Test Code ##
## You are highly encouraged to write your own tests as well,
## but these should be written in a separate file
#################################################################

################
#%% Q1 Test
################

# Initialise
p = 3
xhat = np.linspace(0.5, 1.5, p+1)
n = 7
x = np.linspace(0, 2, n)
tol = 1.0e-10

# Run the function
lagrange_matrix, error_flag = lagrange_poly(p, xhat, n, x, tol)

print("\n################################")
print("Q1 TEST OUTPUT:\n")
print("lagrange_matrix =\n")
print(lagrange_matrix)
print("")
print("error_flag = " + str(error_flag))

################
#%% Q5 Test
################

# Initialise
p = 3
xhat = np.linspace(-0.5, 0.5, p+1)
n = 6
x = np.linspace(-1, 1, n)
tol = 1.0e-12

# Run the function
deriv_lagrange_matrix, error_flag = deriv_lagrange_poly(p, xhat, n, x, tol)

print("\n################################")
print("Q5 TEST OUTPUT:\n")
print("deriv_lagrange_matrix =\n")
print(deriv_lagrange_matrix)
print("")
print("error_flag = " + str(error_flag))
