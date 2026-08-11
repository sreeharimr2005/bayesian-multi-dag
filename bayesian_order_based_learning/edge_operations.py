from score_function import Score

import networkx as nx
import numpy as np

# 0 1 2 3   ---->   0 1 2 3
# 0 2 3 1   ---->   0 3 1 2
# inv[sigma[i]] = i
def get_sigma_inverse(sigma: np.ndarray) -> np.ndarray:
    inv = sigma.copy()
    inv[sigma] = np.arange(len(sigma))

    return inv

def forward_edge_selection(d: int, X: np.ndarray, j: int, score: Score, sigma: np.ndarray) -> set[int]:
    S_j = set()

    while True:
        sigma_inverse = get_sigma_inverse(sigma)
        P_j_sigma = set(sigma_inverse[sigma_inverse < j])

        i_star = max(
            P_j_sigma - S_j,
            key=lambda i: score.local_score(X, j, S_j | {i}) - score.local_score(X, j, S_j)
        )

        if (score.local_score(X, j, S_j | {i_star}) - score.local_score(X, j, S_j) > 0) and len(S_j) <= d:
            S_j.add(i_star)
        else:
            break

    return S_j

def backward_edge_deletion(X: np.ndarray, j: int, score: Score, S_j: set[int]) -> set[int]:
    while not(len(S_j) == 0):
        i_star = max(
            S_j,
            key=lambda i: score.local_score(X, j, S_j - {i}) - score.local_score(X, j, S_j)
        )

        if score.local_score(X, j, S_j - {i_star}) - score.local_score(X, j, S_j) > 0:
            S_j.remove(i_star)
        else:
            break

    return S_j

def edge_selection(d: int, X: np.ndarray, sigma: np.ndarray, score: Score) -> (list[set[int]], nx.DiGraph):
    p = X.shape[1]

    parent_sets: list[set[int]] = [set() for _ in range(p)]
    for j in range(p):
        S_j = forward_edge_selection(d, X, j, score, sigma)
        S_j = backward_edge_deletion(X, j, score, S_j)
        parent_sets[j] = S_j

    G_hat_sigma = nx.DiGraph()

    for i, parent_set in enumerate(parent_sets):
        for parent in parent_set:
            G_hat_sigma.add_edge(parent, i)

    return parent_sets, G_hat_sigma