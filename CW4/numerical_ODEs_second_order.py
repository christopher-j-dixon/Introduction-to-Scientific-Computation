#################################################################
## Functions to carry out numerical solution of second order IVPS
#################################################################

#################################################################
## Imports
## - No further imports should be necessary
## - If you wish to import non-standard modules, ask Ed if that 
## - is acceptable
#################################################################
import numpy as np

#################################################################
## Functions to be completed by student
#################################################################

#%% Q4a code

def adams_bashforth_2(f, a, b, alpha, beta, n, method):
    """
    Solves a second-order IVP using the Euler method or the 2-step Adams-Bashforth method.

    Parameters:
    -----------
    f : function
        The right-hand side function of the second-order differential equation y'' = f(t, y, y').
    a : float
        The initial value of t.
    b : float
        The final value of t.
    alpha : float
        The initial value of y at t = a.
    beta : float
        The initial value of y' at t = a.
    n : int
        The number of steps.
    method : int
        The numerical method to use (1 for Euler, 2 for 2-step Adams-Bashforth).

    Returns
    -------
    t : numpy.ndarray
        Array of time points.
    y : numpy.ndarray
        Array of approximate solution values at the time points in t.
    """
    
    m = n + 1
    t = np.linspace(a, b, m)
    y = np.zeros((m, 2))
    y[0] = np.array([alpha, beta])

    def f_system(t, y):
        # Define the system of first-order equations
        dydt = np.array([y[1], f(t, y[0], y[1])])
        return dydt

    for i in range(n):
        if method == 1:  # Euler method
            h = (b - a) / n
            y[i + 1] = y[i] + h * f_system(t[i], y[i])
        elif method == 2:  # 2-step Adams-Bashforth method
            if i == 0:  # Use Euler method for the first step
                h = (b - a) / n
                y[i + 1] = y[i] + h * f_system(t[i], y[i])
            else:
                h = (b - a) / n
                y[i + 1] = y[i] + (h / 2) * (3 * f_system(t[i], y[i]) - f_system(t[i - 1], y[i - 1]))

    return t, y[:, 0]

#%% Q4b code
  
def compute_ode_errors(n_vals, no_n, a, b, alpha, beta, f, true_y):
    """
    Computes the final error for each step size using the Euler method and the 
    2-step Adams-Bashforth method for a second-order ODE.

    Parameters:
    -----------
    n_vals : list
        List of different numbers of steps to use in the numerical methods.
    no_n : int
        Number of different step sizes to test.
    a : float
        The initial value of t.
    b : float
        The final value of t.
    alpha : float
        The initial value of y at t = a.
    beta : float
        The initial value of y' at t = a.
    f : function
        The right-hand side function of the second-order differential equation y'' = f(t, y, y').
    true_y : function
        The true solution of the differential equation.

    Returns
    -------
    error_y : numpy.ndarray
        Array containing the final errors for each method and step size.
    """
    
    error_y = np.zeros((2, no_n))

    for i, n in enumerate(n_vals):
        t_eul, y_eul = adams_bashforth_2(f, a, b, alpha, beta, n, 1)
        t_ab, y_ab = adams_bashforth_2(f, a, b, alpha, beta, n, 2)
        
        eul_error = true_y(t_eul[-1]) - y_eul[-1]
        ab_error = true_y(t_ab[-1]) - y_ab[-1]
        
        error_y[0, i] = np.abs(eul_error)
        error_y[1, i] = np.abs(ab_error)

    return error_y

#################################################################

# Define the second-order ODE
f = lambda t, y0, y1: (2 + np.exp(-t)) * np.cos(t) - y0 - 2 * y1
true_y = lambda t: np.exp(-t) - np.exp(-t) * np.cos(t) + np.sin(t)

a = 0
b = 1
alpha = 0
beta = 1

################
#%% Q4a Test
################

n = 40

# Compute the numerical solutions
t_euler, y_euler = adams_bashforth_2(f, a, b, alpha, beta, n, 1)
t_ab, y_ab = adams_bashforth_2(f, a, b, alpha, beta, n, 2)

print("\n################################")
print("Q4a TEST OUTPUT (last few values of solutions):\n")

# Print the last few points of each solution for comparison
print("  t      True      Euler    Adams-Bashforth")
print("--------------------------------------------")
for i in range(-4, 0):
    print("{:.2f}   {:.6f}   {:.6f}   {:.6f}".format(t_euler[i], 
          true_y(t_euler[i]), y_euler[i], y_ab[i]))

################
#%% Q4b Test
################

no_n = 6
n_vals = 4 * 2**np.arange(no_n)

errors_y = compute_ode_errors(n_vals, no_n, a, b, alpha, beta, f, true_y)

print("\n################################")
print("Q4b TEST OUTPUT:\n")

print("errors_y = \n")
print(errors_y)

############################################################
