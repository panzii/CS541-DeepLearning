import numpy as np 

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return np.dot(A, B) - C

def problem_1c (A, B, C):
    return A*B + C.T

def problem_1d (x, y):
    return x.T @ y

def problem_1e (A, i):
    return np.sum(A[i, ::2])

def problem_1f (A, c, d):
    return np.mean(A[np.nonzero((A >= c) & (A <= d))])

def problem_1g (A, k):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    sorted_eigenvalues_idx = np.argsort(np.abs(eigenvalues))[::-1]
    largest_eigenvalues_idx = sorted_eigenvalues_idx[0:k]
    filtered_eigenvectors = eigenvectors[:, largest_eigenvalues_idx]
    return filtered_eigenvectors

def problem_1h (x, k, m, s):
    n = len(x)
    z = np.ones(n)
    mean = x + m*z
    covariance = s*np.eye(n)
    return np.random.multivariate_normal(mean, covariance, k).T

def problem_1i (A):
    idx = list(range(0, len(A)))
    np.random.shuffle(idx)
    return A[:, idx]

def problem_1j (x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean)/std

def problem_1k (x, k):
    return np.repeat(np.atleast_2d(x).T, k, axis=1)

def problem_1l (X, Y):
    X_expanded = X[:, :, np.newaxis]
    Y_expanded = Y[:, np.newaxis, :]

    difference = X_expanded - Y_expanded
    squared_diff_sum = np.sum(difference**2, axis=0)
    distance_matrix = np.sqrt(squared_diff_sum) 
    
    return distance_matrix
