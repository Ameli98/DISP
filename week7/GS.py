import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


class GS_solver:
    def __init__(self, vectors: list, weight=None, col=3, row=2) -> None:
        if weight is not None:
            self.weight = weight
        else:
            self.weight = np.ones_like(vectors[0])

        self.sample_points = np.arange(0, len(self.weight), 1)

        self.unit_vectors = [
            vectors[0] / sqrt(self.inner_product(vectors[0], vectors[0]))
        ]
        for vi in vectors[1:]:
            for unit_vector_j in self.unit_vectors:
                vi -= self.inner_product(vi, unit_vector_j) * unit_vector_j
                vi /= sqrt(self.inner_product(vi, vi))
            self.unit_vectors.append(vi)
        self.plot(col, row)

    def plot(self, col, row):
        for index, unit_vector in enumerate(self.unit_vectors):
            plt.subplot(col, row, index + 1)
            plt.plot(self.sample_points, unit_vector)
        plt.show()

    def inner_product(self, x: np.array, y: np.array) -> float:
        product = 0
        for xi, yi, wi in zip(x, y, self.weight):
            product += xi * yi * wi
        return product
