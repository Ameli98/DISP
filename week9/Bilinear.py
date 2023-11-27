import cv2
import numpy as np


class Bilinear:
    def __init__(self, Image: np.array, XScale: float, YScale: float) -> None:
        if Image.ndim == 2:
            Image = np.expand_dims(Image, axis=2)
        else:
            assert Image.ndim == 3

        Input = np.zeros((Image.shape[0] + 1, Image.shape[1] + 1, Image.shape[2]))
        Input[:-1, :-1, :] = Image

        Output = np.zeros(
            (
                int(np.floor(Image.shape[0] * XScale)),
                int(np.floor(Image.shape[1] * YScale)),
                Image.shape[2],
            )
        )
        for i in range(Output.shape[0]):
            for j in range(Output.shape[1]):
                m1, n1 = i / XScale, j / YScale
                m0, n0 = int(np.floor(m1)), int(np.floor(n1))
                a, b = m1 - m0, n1 - n0

                for k in range(Output.shape[2]):
                    Output[i, j, k] = (
                        (1 - a) * (1 - b) * Input[m0, n0, k]
                        + a * (1 - b) * Input[m0 + 1, n0, k]
                        + (1 - a) * b * Input[m0, n0 + 1, k]
                        + a * b * Input[m0 + 1, n0 + 1, k]
                    )

        Output = Output.astype("uint8")
        cv2.imshow("Bilinear result", Output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Image = cv2.imread("Lena.png")
    Image = Image.astype("float16")
    Bilinear(Image, 1.5, 1.6)
