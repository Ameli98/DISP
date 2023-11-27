import numpy as np
from numpy.linalg import svd


def PCA(RawData: np.array, Component: int) -> np.array:
    MinDim = min(*RawData.shape)
    assert Component <= MinDim

    Mean = np.mean(RawData, axis=0)
    Data = RawData - Mean
    U, S, Vh = svd(Data)

    MainComponent = np.zeros_like(RawData, dtype="float64")
    for i in range(Component):
        Ui, Vi = np.reshape(U[:, i], (U.shape[0], 1)), np.reshape(
            Vh[i], (1, Vh.shape[1])
        )
        MainComponent += S[i] * Ui @ Vi
    MainComponent += Mean
    return MainComponent


if __name__ == "__main__":
    RawData = np.array(
        [[2, -1, 3], [-1, 3, 5], [0, 2, 4], [4, -2, -1], [1, 0, 4], [-2, 5, 5]]
    )
    MainComponent = PCA(RawData, 2)
    print(f"RawData:\n{RawData}\n")
    print(f"MainComponent:\n{MainComponent}")
