from scipy.fft import fft2, ifft2
import numpy as np
import matplotlib.pyplot as plt
import cv2


def Fusion(
    LowFreqImage: np.array, HighFreqImage: np.array, ThresholdRatio: float
) -> np.array:
    assert LowFreqImage.shape == HighFreqImage.shape
    assert LowFreqImage.ndim == 2
    assert LowFreqImage.shape[0] == LowFreqImage.shape[1]

    LowFreqImage = fft2(LowFreqImage)
    HighFreqImage = fft2(HighFreqImage)

    Shape = LowFreqImage.shape
    Threshold = int(Shape[0] * ThresholdRatio)

    def FusionMasks() -> np.array:
        Mask = np.zeros(Shape)
        for i in range(Threshold):
            Mask[i, : Threshold + 1 - i] = 1
            Mask[Mask.shape[0] - 1 - i, : Threshold + 1 - i] = 1
            Mask[i, Mask.shape[1] - Threshold + i :] = 1
            Mask[Mask.shape[0] - 1 - i, Mask.shape[1] - Threshold + i :] = 1
        return (Mask, np.ones(Shape) - Mask)

    LowPass, HighPass = FusionMasks()
    FusionImage = LowFreqImage * LowPass + HighFreqImage * HighPass
    FusionImage = ifft2(FusionImage)

    return FusionImage.astype("uint8")


if __name__ == "__main__":
    image1 = cv2.imread("Lena.png")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype("float16")
    image2 = cv2.imread("Barbara.png")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype("float16")

    result = Fusion(image1, image2, 1 / 30)
    cv2.imshow("Fused Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
