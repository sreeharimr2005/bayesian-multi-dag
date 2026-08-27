import numpy as np
from scipy.stats import kendalltau

# Empirical posterior edge inclusion probability
def edge_probability(adj_matrices: np.ndarray, edge: tuple[int, int]) -> float:
    T = adj_matrices.shape[0]
    sum = 0

    for adj_matrix in adj_matrices:
        sum += adj_matrix[edge[0], edge[1]]

    return sum / T

# Average Hamming distance
def hamming_distance(predicted_adj_matrices: np.ndarray, adj_matrices_hat: np.ndarray) -> float:
    K = adj_matrices_hat.shape[0]

    sum = 0
    for k in range(K):
        predicted_adj_matrix_k = predicted_adj_matrices[k]
        adj_matrix_hat_k = adj_matrices_hat[k]

        sum += np.abs(predicted_adj_matrix_k - adj_matrix_hat_k).sum()

    return sum / K

# Posterior mean rank correlation (Kendall’s tau)
def rank_correlation(sigmas_predicted: np.ndarray, sigma_true: np.ndarray) -> float:
    T = sigmas_predicted.shape[0]

    unnormalized_tau_star = 0
    for t in range(T):
        tau_t = kendalltau(sigma_true, sigmas_predicted[t]).statistic
        unnormalized_tau_star += tau_t

    return unnormalized_tau_star / T

def true_positive_rate(predicted_adj_matrix: np.ndarray, true_adj_matrix: np.ndarray) -> float:
    predicted_num_edges = predicted_adj_matrix.sum() / 2

    return  (predicted_adj_matrix * true_adj_matrix).sum() / predicted_num_edges

def false_discovery_rate(predicted_adj_matrix: np.ndarray, true_adj_matrix: np.ndarray) -> float:
    num_edges = true_adj_matrix.sum() / 2

    return ((1 - predicted_adj_matrix) * true_adj_matrix).sum() / num_edges