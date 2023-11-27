import numpy as np


class MomentAnalysis2D:
    def __init__(self, Vector: np.array) -> None:
        assert Vector.ndim == 2
        self.Distribution = Vector / np.sum(Vector)
        Distribution_X = np.sum(self.Distribution, axis=1)
        Distribution_Y = np.sum(self.Distribution, axis=0)
        Mean = lambda x: np.sum(x * np.arange(x.size))
        self.XMEAN = Mean(Distribution_X)
        self.YMEAN = Mean(Distribution_Y)

    def Moment(self, Xth: int, Yth: int, Central: bool = True) -> float:
        Shape = self.Distribution.shape
        NX, NY = np.zeros(Shape), np.zeros(Shape)
        NX = NX * np.arange(Shape[0])
        NY = NY * np.arange(Shape[1]).reshape(1, Shape[1])
        if Central:
            NX -= self.XMEAN
            NY -= self.YMEAN
        NXY = (NX**Xth) * (NY**Yth)
        return np.sum(self.Distribution * NXY)


if __name__ == "__main__":
    # Elliptic image
    Elliptic = np.zeros((15, 15))
    RX, RY = 2, 4
    X0, Y0 = 4, 8
    L0, L1, L2, Linf = 0, 0, 0, 0
    for i in range(Elliptic.shape[0]):
        for j in range(Elliptic.shape[1]):
            if (i - X0) ** 2 // (RX**2) + (j - Y0) ** 2 // (RY**2) <= 1:
                Elliptic[i][j] = 1
                L0 += 1
                L1 += 1
                L2 += 1
                Linf = max(Linf, Elliptic[i][j])
    L2 = L2**0.5
    print("Elliptic Image:", Elliptic)
    print(f"L0 = {L0} \nL1 = {L1}\nL2 = {L2}\nLinf = {Linf}")

    EllipticMoment = MomentAnalysis2D(Elliptic)
    print(
        EllipticMoment.Moment(2, 0),
        EllipticMoment.Moment(1, 1),
        EllipticMoment.Moment(0, 2),
    )
