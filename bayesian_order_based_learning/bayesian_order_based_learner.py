from forward_backward_edge_selection.score_function import Score
from forward_backward_edge_selection.edge_operations import edge_selection
from bayesian_order_based_learning.random_to_random_neighborhood import R2R

import networkx as nx
import numpy as np
import random

class OrderBasedLearner:
    def __init__(self,
                 X: np.ndarray,
                 sigma_0: np.ndarray,
                 c_0 : int = 3,
                 alpha: float = 0.99,
                 gamma: float = 0.01,
                 kappa: float = 0,
                 d: int | None = None,
                 T: int | None = None):
        self.X = X
        self.score = Score(c_0, alpha, gamma, kappa)
        self.sigma_0 = sigma_0

        if d is None:
            self.d = next(iter(X)).shape[1]
        else:
            self.d = d

        if T is None:
            self.T = 20 * (next(iter(X)).shape[1] ** 2)
        else:
            self.T = T

    def compute(self) -> (np.ndarray, list[list[nx.DiGraph]]):
        T = self.T
        K = len(self.X)
        sigma_prev = self.sigma_0

        G_hat_sigma_list_prev = [None for _ in range(K)]
        for k in range(K):
            X_k = self.X[k]

            _, G_hat_sigma_k = edge_selection(self.d, X_k, self.sigma_0, self.score)
            G_hat_sigma_list_prev[k] = G_hat_sigma_k

        pi_sigma_prev = self._compute_prior(G_hat_sigma_list_prev)

        sampled_orderings = []
        dags = []
        for t in range(T):
            sigma_curr = R2R.draw(R2R.get_neighborhood(sigma_prev))

            G_hat_sigma_list_curr = [None for _ in range(K)]
            for k in range(K):
                X_k = self.X[k]

                _, G_hat_sigma_k = edge_selection(self.d, X_k, sigma_curr, self.score)
                G_hat_sigma_list_curr[k] = G_hat_sigma_k

            pi_sigma_curr = self._compute_prior(G_hat_sigma_list_curr)

            a = min(pi_sigma_curr/pi_sigma_prev, 1)
            u = random.uniform(0, 1)

            if u <= a:
                sigma_prev = sigma_curr
                G_hat_sigma_list_prev = G_hat_sigma_list_curr

            sampled_orderings.append(sigma_prev)
            dags.append(G_hat_sigma_list_prev)

        return np.array(sampled_orderings), dags

    def _compute_prior(self, G_hat_sigma_k_list: list[nx.DiGraph]) -> float:
        pi_k_G_k_sigma = []
        for k, G_hat_sigma_k in enumerate(G_hat_sigma_k_list):
            pi_k_G_k_sigma.append(self.score.score(self.X[k], G_hat_sigma_k))

        return float(np.prod(pi_k_G_k_sigma))
