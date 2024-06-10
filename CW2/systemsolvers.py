"""
CW2 systemsolvers

Author: Christopher Dixon
"""

import numpy as np
import matplotlib.pyplot as plt

def no_pivoting(A, b, n, c):
    """
    Returns an array representing the augmented matrix M arrived at by
    starting from the augmented matrix [A b] and performing forward
    elimination without row interchanges until all of the entries below the
    main diagonal in the first c columns are 0.
    
    Parameters
    ----------
    A : numpy.ndarray of shape (n,n)
        Array representing the square matrix A.
    b : numpy.ndarray of shape (n,1)
        Array representing the column vector b.
    n : integer
        Integer that is at least 2.
    c : integer
        Positive integer that is at most n-1.
    
    Returns
    -------
    M : numpy.ndarray of shape (n,n+1)
        Array representing the augmented matrix M.
    """
    M = np.hstack((A, b))  # Create the initial augmented matrix
    
    for i in range(c):
        for j in range(i+1, n):
            m = M[j, i] / M[i, i]
            M[j, i] = 0
            M[j, i+1:n+1] = M[j, i+1:n+1] - m * M[i, i+1:n+1]
    
    return M

def backward_substitution(M, n):
    """
    Returns an array representing the solution x of Ux=v computed using
    backward substitution where U is an upper triangular matrix.
    
    Parameters
    ----------
    M : numpy.ndarray of shape (n,n+1)
        Array representing the augmented matrix [U v].
    n : integer
        Integer that is at least 2.
        
    Returns
    -------
    x : numpy.ndarray of shape (n,1)
        Array representing the solution x.
    """
    x = np.zeros([n, 1])
    
    x[n-1] = M[n-1, n] / M[n-1, n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = (M[i, n] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]
    
    return x

def no_pivoting_solve(A, b, n):
    """
    Returns an array representing the solution x to Ax=b computed using
    forward elimination with no pivoting followed by backward substitution.
    
    Parameters
    ----------
    A : numpy.ndarray of shape (n,n)
        Array representing the square matrix A.
    b : numpy.ndarray of shape (n,1)
        Array representing the column vector b.
    n : integer
        Integer that is at least 2.
    
    Returns
    -------
    x : numpy.ndarray of shape (n,1)
        Array representing the solution x.
    """
    M = no_pivoting(A, b, n, n-1)
    x = backward_substitution(M, n)
    return x

#%% Gaussian elimination with scaled partial pivoting

def find_max(M, s, n, i):
    """
    Find the index of the row with the maximum scaled value in column i.

    Parameters
    ----------
    M : numpy.ndarray
        Augmented matrix of shape (n, n+1).
    s : numpy.ndarray
        Scale factors of shape (n, ).
    n : int
        Size of the matrix.
    i : int
        Column index.

    Returns
    -------
    int
        Index of the row with the maximum scaled value.
    """
    m = 0
    p = 0
    
    for j in range(i, n):
        if abs(M[j, i] / s[j]) > m:
            m = abs(M[j, i] / s[j])
            p = j
    
    return p

def scaled_partial_pivoting(A, b, n, c):
    """
    Perform scaled partial pivoting on the augmented matrix [A|b].

    Parameters
    ----------
    A : numpy.ndarray
        Square matrix of shape (n, n).
    b : numpy.ndarray
        Column vector of shape (n, 1).
    n : int
        Size of the matrix.
    c : int
        Column index to stop the forward elimination prematurely.

    Returns
    -------
    numpy.ndarray
        Augmented matrix after forward elimination.
    
    Raises
    ------
    ValueError
        If no unique solution exists.
    """
    def pivot(M, s, n, i):
        p = find_max(M, s, n, i)
        s[i], s[p] = s[p], s[i]
        M[[i, p], :] = M[[p, i], :]

    M = np.hstack((A, b))  # Create augmented matrix
    s = np.amax(np.abs(M), axis=1)  # Scale factors

    for i in range(c):
        pivot(M, s, n, i)
        for j in range(i + 1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:] -= factor * M[i, i:]

    if M[n-1, n-1] == 0:
        raise ValueError('No unique solution')
    
    return M

def spp_solve(A, b, n):
    """
    Solve the system of equations Ax = b using scaled partial pivoting.

    Parameters
    ----------
    A : numpy.ndarray
        Square matrix of shape (n, n).
    b : numpy.ndarray
        Column vector of shape (n, 1).
    n : int
        Size of the matrix.

    Returns
    -------
    numpy.ndarray
        Solution vector x of shape (n, 1).
    """
    M = scaled_partial_pivoting(A, b, n, n - 1)
    x = backward_substitution(M, n)
    return x

#%% PLU factorisation

def PLU(A, n):
    """
    Perform PLU factorisation of the matrix A.

    Parameters
    ----------
    A : numpy.ndarray
        Square matrix of shape (n, n).
    n : int
        Size of the matrix.

    Returns
    -------
    P : numpy.ndarray
        Permutation matrix of shape (n, n).
    L : numpy.ndarray
        Lower triangular matrix of shape (n, n).
    U : numpy.ndarray
        Upper triangular matrix of shape (n, n).
    """
    P = np.identity(n)
    L = np.zeros((n, n))
    U = np.copy(A)
    
    for i in range(n):
        u = [s for s in range(i, n) if abs(U[s, i]) > 1e-15]
        s = min(u) if u else i
        
        if s != i:
            P[[i, s], :], L[[i, s], :], U[[i, s], :] = P[[s, i], :], L[[s, i], :], U[[s, i], :]
        
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]
    
    P = P.T
    L += np.identity(n)
    
    return P, L, U

#%% Jacobi method

def Jacobi(A, b, n, x0, N):
    """
    Perform the Jacobi method for solving the system of linear equations Ax = b.

    Parameters
    ----------
    A : numpy.ndarray
        Square matrix of shape (n, n).
    b : numpy.ndarray
        Column vector of shape (n, 1).
    n : int
        Size of the matrix.
    x0 : numpy.ndarray
        Initial approximation vector of shape (n, 1).
    N : int
        Number of iterations.

    Returns
    -------
    numpy.ndarray
        Approximations of x at each iteration, shape (n, N+1).
    """
    x = np.zeros((n, N + 1))
    x[:, 0] = x0.flatten()
    
    for k in range(1, N + 1):
        for i in range(n):
            sum_ = sum(-A[i, j] * x[j, k-1] for j in range(n) if j != i)
            x[i, k] = (sum_ + b[i]) / A[i, i]
    
    return x

#%% Jacobi plot

def Jacobi_plot(A, b, n, x0, N):
    """
    Plot the convergence of the Jacobi method.

    Parameters
    ----------
    A : numpy.ndarray
        Square matrix of shape (n, n).
    b : numpy.ndarray
        Column vector of shape (n, 1).
    n : int
        Size of the matrix.
    x0 : numpy.ndarray
        Initial approximation vector of shape (n, 1).
    N : int
        Number of iterations.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes._axes.Axes
        Figure and axes of the plot.
    """
    x = no_pivoting_solve(A, b, n)
    J = Jacobi(A, b, n, x0, N)
    
    k_array = np.arange(N + 1)
    l_inf = [np.linalg.norm(x - J[:, k].reshape(-1, 1), np.inf) for k in k_array]
    l_two = [np.linalg.norm(x - J[:, k].reshape(-1, 1), 2) for k in k_array]
    
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.grid(True)
    
    ax.plot(k_array, l_inf, "s", label="$||x-x^{(k)}||_\\infty$")
    ax.plot(k_array, l_two, "o", label="$||x-x^{(k)}||_2$")
    
    ax.legend()
    return fig, ax
