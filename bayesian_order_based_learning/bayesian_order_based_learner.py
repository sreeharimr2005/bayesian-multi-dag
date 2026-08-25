from score_function import Score
from edge_operations import edge_selection
from bayesian_order_based_learning.random_to_random_neighborhood import R2R

from concurrent.futures import ProcessPoolExecutor
import ray

from typing import Optional
from tqdm import tqdm
import networkx as nx
import numpy as np
import random

RAY_ADDRESS = None

@ray.remote
def _edge_selection_remote(d: int, X_k: np.ndarray, sigma: np.ndarray, score: Score):
    return edge_selection(d, X_k, sigma, score)

class OrderBasedLearner:
    def __init__(self,
                 X: list[np.ndarray],
                 sigma_0: np.ndarray,
                 c_0 : int = 3,
                 alpha: float = 0.99,
                 gamma: float = 0.01,
                 kappa: float = 0,
                 d: Optional[int] = None,
                 T: Optional[int] = None,
                 burn_in: Optional[float] = None,
                 verbose: bool = False):
        self.X = X
        self.score = Score(c_0, alpha, gamma, kappa)
        self.sigma_0 = sigma_0

        if d is None:
            self.d = X[0].shape[1]
        else:
            self.d = d

        if T is None:
            self.T = 20 * (X[0].shape[1] ** 2)
        else:
            self.T = T

        if burn_in is None:
            self.burn_in = self.T / 2
        else:
            self.burn_in = burn_in

        self.verbose = verbose

        if not ray.is_initialized():
            ray.init(address=RAY_ADDRESS, runtime_env={"working_dir": "/Users/sreehari_miniravi/Work/ResearchProjects/bayesian-multi-dag/bayesian_order_based_learning"})

        self._score_ref = ray.put(self.score)
        self._X_refs = [ray.put(self.X[k]) for k in range(len(self.X))]

    def compute(self) -> (np.ndarray, list[list[nx.DiGraph]], list[float]):
        T = self.T
        K = len(self.X)
        sigma_prev = self.sigma_0

        with ProcessPoolExecutor(max_workers=K) as executor:
            """
            G_hat_sigma_list_prev = [None for _ in range(K)]
            for k in range(K):
                X_k = self.X[k]

                _, G_hat_sigma_k = edge_selection(self.d, X_k, self.sigma_0, self.score)
                G_hat_sigma_list_prev[k] = G_hat_sigma_k
            """

            G_hat_sigma_list_prev = self._compute_graphs(executor, self.sigma_0)
            pi_sigma_prev = self.compute_posterior(G_hat_sigma_list_prev)

            sampled_orderings = []
            dags = []
            log_posteriors = [pi_sigma_prev]
            for t in tqdm(range(T), desc="MCMC Sampling wth R2R", disable=not self.verbose):
                #sigma_curr = R2R.draw(R2R.get_neighborhood(sigma_prev))
                sigma_curr = R2R.efficient_draw(sigma_prev)

                """
                G_hat_sigma_list_curr = [None for _ in range(K)]
                for k in range(K):
                    X_k = self.X[k]

                    _, G_hat_sigma_k = edge_selection(self.d, X_k, sigma_curr, self.score)
                    G_hat_sigma_list_curr[k] = G_hat_sigma_k
                """

                G_hat_sigma_list_curr = self._compute_graphs(executor, sigma_curr)
                pi_sigma_curr = self.compute_posterior(G_hat_sigma_list_curr)

                a = min(np.exp(pi_sigma_curr - pi_sigma_prev), 1)
                u = random.uniform(0, 1)

                if u <= a:
                    sigma_prev = sigma_curr
                    G_hat_sigma_list_prev = G_hat_sigma_list_curr
                    pi_sigma_prev = pi_sigma_curr
                    log_posteriors.append(pi_sigma_curr)
                else:
                    log_posteriors.append(log_posteriors[-1])

                if t >= self.burn_in:
                    sampled_orderings.append(sigma_prev)
                    dags.append(G_hat_sigma_list_prev)

        return np.array(sampled_orderings), dags, log_posteriors

    def compute_posterior(self, G_hat_sigma_k_list: list[nx.DiGraph]) -> float:
        pi_k_G_k_sigma = []
        for k, G_hat_sigma_k in enumerate(G_hat_sigma_k_list):
            pi_k_G_k_sigma.append(self.score.score(self.X[k], G_hat_sigma_k))

        return float(np.sum(pi_k_G_k_sigma))

    def _compute_graphs(self, executor, sigma: np.ndarray) -> list[nx.DiGraph]:
        """
        futures = [
            executor.submit(edge_selection, self.d, self.X[k], sigma, self.score)
            for k in range(self.X.shape[0])
        ]
        """

        futures = [
            _edge_selection_remote.remote(self.d, self._X_refs[k], sigma, self._score_ref)
            for k in range(len(self.X))
        ]

        results = ray.get(futures)
        return [result[1] for result in results]

    def shutdown(self):
        ray.shutdown()
