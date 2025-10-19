import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar

def second_smallest_eigenvalue(graph):
    """
    Compute the second smallest eigenvalue (Fiedler value) of the Laplacian matrix
    of an undirected graph given its adjacency matrix.

    Parameters:
    -----------
    graph : np.ndarray
        Adjacency matrix of the graph (n x n), must be symmetric.

    Returns:
    --------
    float
        The second smallest eigenvalue of the Laplacian matrix.
    """
    # Check if symmetric (undirected graph)
    if not np.allclose(graph, graph.T):
        raise ValueError("Adjacency matrix must be symmetric for an undirected graph.")

    # Degree matrix
    degrees = np.sum(graph, axis=1)
    degree_matrix = np.diag(degrees)

    # Laplacian matrix
    laplacian = degree_matrix - graph

    # Eigenvalues: use eigvalsh (for symmetric matrix)
    eigenvalues = np.linalg.eigvalsh(laplacian)

    # Sort in ascending order, return second smallest
    eigenvalues_sorted = np.sort(eigenvalues)
    return eigenvalues_sorted[1]


# compute rho for each iteration (algorithm 2 in the paper)
def compute_rho(graph, sigma_cdp, sigma_cor, C):
    """
    Compute the rho value for the current iteration based on the graph and parameters.
    This function computes the rho value using Theorem 1 in The Privacy Power of Correlated Noise in Decentralized Learning.

    Args:
        graph (np.ndarray): The adjacency matrix of the graph.
        sigma_cdp (float): The sigma value for the local differential privacy.
        sigma_cor (float): The sigma value for the global differential privacy.
        C (float): The constant value used in the computation.

    Returns:
        float: The computed rho value for the current iteration.
    """
    a1 = second_smallest_eigenvalue(graph)
    n = graph.shape[0]

    term1 = 1 / ((n - 1) * sigma_cdp)
    term2_numerator = 1 - 1 / (n - 1)
    term2_denominator = sigma_cdp + a1 * sigma_cor
    term2 = term2_numerator / term2_denominator

    rho = 2 * C**2 * (term1 + term2)

    return rho


# compute rho for all iterations
def compute_rho_all(n_iter, graph, sigma_cdp, sigma_cor, C):
    """
    Compute the rho values for all iterations.
    """
    rho_current = compute_rho(graph, sigma_cdp, sigma_cor, C)
    return n_iter * rho_current


# compute epsilon using \epsilon= \rho + 2 \cdot \sqrt{-\rho \cdot \log_e \delta}
def compute_epsilon(n_iter, graph, sigma_cdp, sigma_cor, C):
    """
    Compute the epsilon value for the given delta.
    """
    rho = compute_rho_all(n_iter, graph, sigma_cdp, sigma_cor, C)
    delta = 1e-5

    epsilon = rho + 2 * np.sqrt(-rho * np.log(delta))
    return epsilon


# compute mu-GDP using
def comnpute_mu_gdp(n_iter, graph, sigma_fdp, sigma_cor, C):
    """
    Compute the mu value for the given epsilon and delta.
    """
    a1 = second_smallest_eigenvalue(graph)
    n = graph.shape[0]

    term1 = 1 / ((n - 1) * sigma_fdp)
    term2_numerator = 1 - 1 / (n - 1)
    term2_denominator = sigma_fdp + a1 * sigma_cor
    term2 = term2_numerator / term2_denominator

    mu = C * np.sqrt(2*(term1 + term2))

    return mu * np.sqrt(n_iter)


# given epsilon, sigma_cor, sigma_fdp, and C, compute delta
def delta_eqn(n_iter, graph, sigma_fdp, sigma_cdp, sigma_cor, C, epsilon):
    """
    Compute the delta value for the given epsilon and delta.
    """
    mu = comnpute_mu_gdp(n_iter, graph, sigma_fdp, sigma_cor, C)
    epsilon = compute_epsilon(n_iter, graph, sigma_cdp, sigma_cor, C)

    terma = norm.cdf(-epsilon / mu + mu / 2)
    termb = np.exp(epsilon) * norm.cdf(-epsilon / mu - mu / 2)

    return terma - termb

lower = 1e-7
upper = 12

# search for sigma_fdp using bisection method to match delta_eqn = delta
def compute_sigma_fdp(n_iter, graph, sigma_cdp, sigma_cor, C, delta=1e-5, bracket=(lower, upper)):
    """
    Solves for sigma_fdp such that delta_eqn = delta
    """
    epsilon = compute_epsilon(n_iter, graph, sigma_cdp, sigma_cor, C)

    def root_function(sigma_fdp):
        return delta_eqn(n_iter, graph, sigma_fdp, sigma_cdp, sigma_cor, C, epsilon) - delta
    # print(root_function(lower), root_function(upper))

    sol = root_scalar(root_function, bracket=bracket, method='brentq')
    if sol.converged:
        return sol.root
    else:
        raise RuntimeError("Root finding for sigma_gdp2 did not converge")