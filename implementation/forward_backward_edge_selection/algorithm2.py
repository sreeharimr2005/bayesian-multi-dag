from edge_selection import forward_edge_selection, backward_edge_deletion
from score_function import Score

import numpy as np

def algorithm2(X: np.ndarray, sigma: np.ndarray, score: Score) -> list[set[int]]:
    p = X.shape[1]

    parent_sets = []
    for j in range(p):
        S_j = forward_edge_selection(j, score, sigma)
        S_j = backward_edge_deletion(j, score, S_j)
        parent_sets.append(S_j)

    return parent_sets