from math import floor
import random
import numpy as np

class R2R:
    @staticmethod
    def get_neighborhood(sigma: np.ndarray) -> list[np.ndarray]:
        r2r_sigma = []

        for i in range(len(sigma)):
            for j in range(len(sigma)):
                if i == j:
                    continue
                elif i < j:
                    r2r_sigma.append(R2R._R2R_lesser_operation(sigma))
                elif i > j:
                    r2r_sigma.append(R2R._R2R_greater_operation(sigma))

        return r2r_sigma

    @staticmethod
    def draw(r2r_sigma: list[np.ndarray]) -> np.ndarray:
        idx = floor(random.uniform(0,1) * len(r2r_sigma))

        return r2r_sigma[idx]

    @staticmethod
    def _R2R_lesser_operation(sigma: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def _R2R_greater_operation(sigma: np.ndarray) -> np.ndarray:
        pass