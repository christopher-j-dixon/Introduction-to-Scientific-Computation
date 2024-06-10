#################################################################
## Functions to carry out numerical solution of first order IVPS
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

#%% Q3 code

def adams_bashforth(f, a, b, ya, n, method):
    """
    Approximates the solution y(t) for an initial value problem over the interval [a, b]
    using the specified numerical method (Euler or 2-step Adams-Bashforth).

    Parameters:
    -----------
    f : function
        The right-hand side function of the differential equation y' = f(t, y).
    a : float
        The initial value of t.
    b : float
        The final value of t.
    ya : float
        The initial value of y at t = a.
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
    
    t = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    y[0] = ya
    h = (b - a) / n  # Step size
    
    if method == 1:  # Euler method
        for i in range(n):
            y[i + 1] = y[i] + h * f(t[i], y[i])
            
    elif method == 2:  # 2-step Adams-Bashforth method
        # Use Euler method to obtain the second value
        y[1] = y[0] + h * f(t[0], y[0])
        
        for i in range(1, n):
            y[i + 1] = y[i] + (h / 2) * (3 * f(t[i], y[i]) - f(t[i - 1], y[i - 1]))
            
    return t, y

################
#%% Q3 Test
################

# Initialise
a = 0
b = 2
ya = 0.5
n = 40

# Define the ODE and the true solution
f = lambda t, y: y - t**2 + 1
y_true = lambda t: (t + 1)**2 - 0.5 * np.exp(t)

# Compute the numerical solutions
t_euler, y_euler = adams_bashforth(f, a, b, ya, n, 1)
t_ab, y_ab = adams_bashforth(f, a, b, ya, n, 2)

print("\n################################")
print("Q3 TEST OUTPUT (last few values of solutions):\n")

# Print the last few points of each solution for comparison
print("  t      True      Euler    Adams-Bashforth")
print("--------------------------------------------")
for i in range(-4, 0):
    print("{:.2f}   {:.6f}   {:.6f}   {:.6f}".format(t_euler[i], 
          y_true(t_euler[i]), y_euler[i], y_ab[i]))
