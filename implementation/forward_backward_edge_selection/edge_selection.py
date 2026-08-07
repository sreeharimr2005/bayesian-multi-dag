from score_function import Score

import numpy as np

# 0 1 2 3   ---->   0 1 2 3
# 0 2 3 1   ---->   0 3 1 2
# inv[sigma[i]] = i
def get_sigma_inverse(sigma: np.ndarray) -> np.ndarray:
    inv = sigma.copy()
    inv[sigma] = np.arange(len(sigma))

    return inv

def forward_edge_selection(j: int, score: Score, sigma: np.ndarray) -> set[int]:
    S_j = set()

    while True:
        sigma_inverse = get_sigma_inverse(sigma)
        P_j_sigma = set(sigma_inverse[sigma_inverse < j])

        i_star = max(
            P_j_sigma - S_j,
            key=lambda i: score.local_score(j, S_j | {i}) - score.local_score(j, S_j)
        )

        if score.local_score(j, S_j | {i_star}) - score.local_score(j, S_j) > 0:
            S_j.add(i_star)
        else:
            break

    return S_j

def backward_edge_deletion(j: int, score: Score, S_j: set[int]) -> set[int]:
    while not(len(S_j) == 0):
        i_star = max(
            S_j,
            key=lambda i: score.local_score(j, S_j - {i}) - score.local_score(j, S_j)
        )

        if score.local_score(j, S_j - {i_star}) - score.local_score(j, S_j) > 0:
            S_j.remove(i_star)
        else:
            break
