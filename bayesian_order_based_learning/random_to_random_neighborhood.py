from math import floor
import random
import numpy as np

class R2R:
    @staticmethod
    def get_neighborhood(sigma: np.ndarray) -> np.ndarray:
        r2r_sigma = []

        for i in range(len(sigma)):
            for j in range(len(sigma)):
                if i == j:
                    continue
                else:
                    sigma_j = sigma[j]
                    sigma_new = np.delete(sigma, j)
                    sigma_new = np.insert(sigma_new, i, sigma_j)
                    r2r_sigma.append(sigma_new)

        return np.unique(np.array(r2r_sigma), axis=0)

    @staticmethod
    def draw(r2r_sigma: np.ndarray) -> np.ndarray:
        idx = floor(random.uniform(0,1) * r2r_sigma.shape[0])

        return r2r_sigma[idx]

    @staticmethod
    def efficient_draw(sigma: np.ndarray) -> np.ndarray:
        i = 0
        j = 0

        while not i == j:
            i, j = random.sample(range(len(sigma)), 2)

        sigma_j = sigma[j]
        sigma_new = np.delete(sigma, j)
        sigma_new = np.insert(sigma_new, i, sigma_j)

        return sigma_new