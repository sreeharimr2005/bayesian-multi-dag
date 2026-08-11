from bayesian_order_based_learning.bayesian_order_based_learner import OrderBasedLearner
from bayesian_order_based_learning.metrics import *

import numpy as np
import networkx as nx
import random

class Simulation:
    def __init__(self, sigma: np.ndarray, n_k: int, datasets: int, iterations: int):
        self.sigma = sigma
        self.n_k = n_k
        self.datasets = datasets
        self.iterations = iterations

        self.c_0 = None
        self.alpha = None
        self.gamma = None
        self.kappa = None
        self.d = None
        self.T = None
        self.burn_in = None

    def modify_hyperparameters(self, c_0: int, alpha: float, gamma: float, kappa: float, d: int, T: int, burn_in: float):
        self.c_0 = c_0
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.d = d
        self.T = T
        self.burn_in = burn_in

    def simulate(self) -> (list[float], float):
        p = len(self.sigma)
        error_variance = 1
        kwargs = {}

        if self.c_0 is not None:
            kwargs["c_0"] = self.c_0
        if self.alpha is not None:
            kwargs["alpha"] = self.alpha
        if self.gamma is not None:
            kwargs["gamma"] = self.gamma
        if self.kappa is not None:
            kwargs["kappa"] = self.kappa
        if self.d is not None:
            kwargs["d"] = self.d
        if self.T is not None:
            kwargs["T"] = self.T
        if self.burn_in is not None:
            kwargs["burn_in"] = self.burn_in

        generated_datasets = []
        G_hat_sigma_list = []
        for k in range(self.datasets):
            G_k = self._get_graph()
            G_hat_sigma_list.append(G_k)

            dataset_k = np.zeros((self.n_k, p))
            for node in G_k.nodes:
                if G_k.predecessors(node) is None:
                    dataset_k[:, node] = np.random.normal(0, error_variance, self.n_k)

                for parent in G_k.predecessors(node):
                    dataset_k[:, node] += G_k[parent][node]["weight"] * dataset_k[:, parent]

            generated_datasets.append(dataset_k)

        generated_datasets = np.array(generated_datasets)

        sigma_0 = np.arange(1, p + 1)
        obl = OrderBasedLearner(generated_datasets, sigma_0, **kwargs)

        predicted_orderings, predicted_dags = obl.compute()
        predicted_adj_matrices = np.array([
            [nx.to_numpy_array(G, dtype=np.dtype(int)) for G in predicted_dags_t]
            for predicted_dags_t in predicted_dags
        ])
        true_adj_matrices = np.array([
            nx.to_numpy_array(G, dtype=np.dtype(int)) for G in G_hat_sigma_list
        ])

        hd = [hamming_distance(predicted, true_adj_matrices) for predicted in predicted_adj_matrices]
        tau_star = rank_correlation(predicted_orderings, self.sigma)

        return hd, tau_star

    def _get_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        p = len(self.sigma)

        G.add_nodes_from(range(p))
        for j in range(p):
            for i in range(j):
                prob_edge = 3 / (2 * p - 2)
                u = random.uniform(0, 1)

                if u <= prob_edge:
                    edge_weight = random.uniform(0.5, 1.5)
                    if edge_weight >= 1:
                        edge_weight = edge_weight - 2
                    G.add_edge(self.sigma[i].item(), self.sigma[j].item(), weight=edge_weight)

        return G