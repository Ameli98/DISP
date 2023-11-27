from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d as conv2d
import numpy as np
import cv2


def Equalizer(BlurImage: np.array, Mask: np.array, CONST: float) -> np.array:
    HALFLEN = (Mask.shape[0] // 2, Mask.shape[1] // 2)

    ExtendMask = np.zeros_like(BlurImage)
    ExtendMask[: HALFLEN[0] + 1, : HALFLEN[1] + 1] = Mask[HALFLEN[0] :, HALFLEN[1] :]
    ExtendMask[-HALFLEN[0] :, : HALFLEN[1] + 1] = Mask[: HALFLEN[0], HALFLEN[1] :]
    ExtendMask[: HALFLEN[0] + 1, -HALFLEN[1] :] = Mask[HALFLEN[0] :, : HALFLEN[1]]
    ExtendMask[-HALFLEN[0] :, -HALFLEN[1] :] = Mask[: HALFLEN[0], : HALFLEN[1]]

    ExtendMask = fft2(ExtendMask)
    ExtendMask += CONST / np.conjugate(ExtendMask)
    ExtendMask = 1 / ExtendMask

    RecoverImage = fft2(BlurImage) * ExtendMask
    RecoverImage = np.abs(ifft2(RecoverImage)).astype("uint8")

    return RecoverImage


if __name__ == "__main__":
    # Make image blurred
    Image = cv2.imread("Lena.png")
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY).astype("float16")

    Mask = [np.arange(-10, 11).reshape(1, 21) ** 2 for i in range(21)]
    Mask = np.concatenate(Mask, axis=0)
    Mask += Mask.transpose()
    Mask = np.exp(-0.1 * Mask)
    Mask /= Mask.sum()

    NOISEAMP1, NOISEAMP2 = 5, 20
    Noise1 = NOISEAMP1 * (np.random.rand(*Image.shape) - 0.5)
    Noise2 = NOISEAMP2 * (np.random.rand(*Image.shape) - 0.5)

    BlurImage1 = conv2d(Image, Mask, "same") + Noise1
    BlurImage2 = conv2d(Image, Mask, "same") + Noise2

    CONST_A, CONST_B = 0.04, 0.1
    RecoverImage1a = Equalizer(BlurImage1, Mask, CONST_A)
    RecoverImage1b = Equalizer(BlurImage1, Mask, CONST_B)

    RecoverImage2a = Equalizer(BlurImage2, Mask, CONST_A)
    RecoverImage2b = Equalizer(BlurImage2, Mask, CONST_B)

    BlurImages = np.concatenate((BlurImage1, BlurImage2), axis=1)
    RecoverImagesa = np.concatenate((RecoverImage1a, RecoverImage2a), axis=1)
    RecoverImagesb = np.concatenate((RecoverImage1b, RecoverImage2b), axis=1)
    Result = np.concatenate(
        (BlurImages, RecoverImagesa, RecoverImagesb), axis=0
    ).astype("uint8")

    cv2.imshow("Recover Image", Result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
