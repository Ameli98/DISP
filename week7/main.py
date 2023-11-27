import numpy as np
from GS import GS_solver

if __name__ == "__main__":
    Vec = np.arange(11, dtype="float64")
    Vectors = [Vec**i for i in range(5)]
    Norm = GS_solver(Vectors)
