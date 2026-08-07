from math import log

import networkx as nx
import numpy as np

class Score:
    def __init__(self, X: np.ndarray, c_0: float, alpha: float, gamma: float, kappa: float):
        self.X = X
        self.c_0 = c_0
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa

    def local_score(self, j: int, candidates: set[int]) -> float:
        cardinality = len(candidates)
        n = self.X.shape[0]
        p = self.X.shape[1]

        first_half = -((self.c_0 * log(p)) + 0.5 * (log(1 + (self.alpha / self.gamma)))) * cardinality
        second_half = (-0.5 * ((self.alpha * n) + self.kappa) *
                       log(n * self._calculate_residual_variance(j, candidates)))

        return first_half + second_half

    def score(self, G: nx.DiGraph) -> float:
        return sum(self.local_score(j, set(G.predecessors(j))) for j in G.nodes())

    def _calculate_residual_variance(self, j: int, S: set[int]) -> float:
        n = self.X.shape[0]
        n_reciprocal = 1 / n
        X_j = self.X[:, j]
        X_S = self.X[:, list(S)]
        I_n = np.eye(n)

        if len(S) == 0:
            return n_reciprocal * float(X_j.T @ X_j)

        return n_reciprocal * float(
            X_j.T @
            (I_n - (X_S @ np.linalg.inv(X_S.T @ X_S) @ X_S.T)) @
            X_j
        )