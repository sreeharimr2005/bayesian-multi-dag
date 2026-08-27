from bayesian_order_based_learning.bayesian_order_based_learner import OrderBasedLearner

import numpy as np
import networkx as nx
import random

class Simulation:
    def __init__(self, sigma: np.ndarray, n_k: int, verbose: bool = False, K_list: list[int] = None):
        self.sigma = sigma
        self.n_k = n_k
        self.verbose = verbose

        self.c_0 = None
        self.alpha = None
        self.gamma = None
        self.kappa = None
        self.d = None
        self.T = None
        self.burn_in = None

        self._true_dags = None
        self._datasets = None

        if K_list is not None:
            self.K_list = K_list
        else:
            self.K_list = [1]

        self._true_posterior_list = [None] * len(self.K_list)

    def modify_hyperparameters(self,
                               c_0: int = None,
                               alpha: float = None,
                               gamma: float = None,
                               kappa: float = None,
                               d: int = None,
                               T: int = None,
                               burn_in: float = None):
        self.c_0 = c_0
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.d = d
        self.T = T
        self.burn_in = burn_in

    def simulate(self):
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

        K_max = max(self.K_list)
        generated_datasets = []
        true_dags = []

        for k in range(K_max):
            G_k = self._get_graph()
            true_dags.append(G_k)

            dataset_k = np.zeros((self.n_k, p))
            for node in nx.topological_sort(G_k):
                dataset_k[:, node] = np.random.normal(0, error_variance, self.n_k)

                for parent in G_k.predecessors(node):
                    dataset_k[:, node] += G_k[parent][node]["weight"] * dataset_k[:, parent]

            generated_datasets.append(dataset_k)

        self._true_dags = true_dags
        self._datasets = generated_datasets

        # generated_datasets = np.array(generated_datasets)

        sigma_0 = np.arange(0, p)

        predicted_orderings_list = [[] for _ in self.K_list]
        predicted_dags_list = [[] for _ in self.K_list]
        log_posteriors_list = [[] for _ in self.K_list]

        for idx, K in enumerate(self.K_list):
            print(f"\nK={K} being computed ...\n")

            datasets_K = generated_datasets[:K]
            true_dags_K = true_dags[:K]

            obl = OrderBasedLearner(datasets_K, sigma_0, verbose=self.verbose, **kwargs)
            self._true_posterior_list[idx] = obl.compute_posterior(true_dags_K)

            predicted_orderings_list[idx], predicted_dags_list[idx], log_posteriors_list[idx] = obl.compute()
            obl.shutdown()

        """
        predicted_adj_matrices = np.array([
            [nx.to_numpy_array(G, dtype=np.dtype(int)) for G in predicted_dags_t]
            for predicted_dags_t in predicted_dags
        ])
        true_adj_matrices = np.array([
            nx.to_numpy_array(G, weight=None, dtype=np.dtype(int)) for G in G_hat_sigma_list
        ])

        hd = [hamming_distance(predicted, true_adj_matrices) for predicted in predicted_adj_matrices]
        tau_star = rank_correlation(predicted_orderings, self.sigma)
        """
        return predicted_orderings_list, predicted_dags_list, log_posteriors_list

    def get_true_dags(self) -> list[list[nx.DiGraph]]:
        return self._true_dags

    def get_datasets(self):
        return self._datasets

    def get_true_posterior(self) -> list[float]:
        return self._true_posterior_list

    def _get_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        p = len(self.sigma)
        prob_edge = 3 / (2 * p - 2)

        G.add_nodes_from(range(p))
        for j in range(p):
            for i in range(j):
                u = random.uniform(0, 1)

                if u <= prob_edge:
                    edge_weight = random.choice([random.uniform(-1.0, -0.5), random.uniform(0.5, 1.0)])
                    G.add_edge(self.sigma[i].item(), self.sigma[j].item(), weight=edge_weight)

        return G