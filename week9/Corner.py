import cv2
import numpy as np
from scipy.signal import convolve2d as conv2d
import argparse

class CornerDetector:
    def __init__(
        self, Image: np.array, k: float = 0.04, W_AMP: int = 5, SIGMA: float = 2
    ) -> None:
        X, Y = np.zeros_like(Image), np.zeros_like(Image)
        X[0], X[-1] = Image[1], -Image[-2]
        Y[:, 0], Y[:, -1] = Image[:, 1], -Image[:, -2]
        X[1:-1] = Image[2:] - Image[:-2]
        Y[:, 1:-1] = Image[:, 2:] - Image[:, :-2]

        W = np.arange(-W_AMP, W_AMP + 1).reshape(1, -1) ** 2
        W = W + W.T
        W = np.exp(W * -0.5 / (SIGMA**2))

        A, B, C = (
            conv2d(X**2, W, "same"),
            conv2d(Y**2, W, "same"),
            conv2d(X * Y, W, "same"),
        )

        TR = A + B
        DET = A * B - C**2

        R = DET - k * (TR**2)

        Mask = np.full(R.shape, True)
        # 4 Corner
        Mask[0, 0] = R[0, 0] > max(R[0, 1], R[1, 0])
        Mask[-1, 0] = R[-1, 0] > max(R[-2, 0], R[-1, 1])
        Mask[0, -1] = R[0, -1] > max(R[1, -1], R[0, -2])
        Mask[-1, -1] = R[-1, -1] > max(R[-2, -1], R[-1, -2])

        # 4 Edge
        for m in range(Mask.shape[0])[1:-1]:
            Mask[m, 0] = R[m, 0] > max(R[m, 1], R[m - 1, 0], R[m + 1, 0])
            Mask[m, -1] = R[m, -1] > max(R[m, -2], R[m - 1, -1], R[m + 1, -1])
        for n in range(R.shape[1])[1:-1]:
            Mask[0, n] = R[0, n] > max(R[1, n], R[0, n - 1], R[0, n + 1])
            Mask[-1, n] = R[-1, n] > max(R[-2, n], R[-1, n - 1], R[-1, n + 1])
        Mask = Mask & (R > np.max(R) / 100)

        self.Result = np.zeros_like(Image, dtype="uint8")
        self.Result[Mask] = 255

    def ShowResult(self):
        cv2.imshow("Corner Result", self.Result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ImagePath", type = str)
    args = parser.parse_args()

    Image = cv2.imread(args.ImagePath)
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY).astype("float16")
    Detector = CornerDetector(Image)
    Detector.ShowResult()
