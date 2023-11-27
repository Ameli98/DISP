import cv2
import numpy as np
from math import log10, sqrt


def InnerProduct(x: np.array, y: np.array) -> float:
    return np.mean(x * y)


def SecondMoment(x: np.array) -> float:
    return np.mean(x**2)


def MSE(x: np.array, y: np.array) -> float:
    difference = x - y
    return SecondMoment(difference)


def NRMSE(x: np.array, y: np.array) -> float:
    MSError = MSE(x, y)
    SecMom = SecondMoment(x)
    return sqrt(MSError / SecMom)


def PSNR(x: np.array, y: np.array) -> float:
    MAXVALUE = 255
    MSError = MSE(x, y)
    Ratio = (MAXVALUE**2) / MSError
    return 10 * log10(Ratio)


def CoVarience(x: np.array, y: np.array) -> float:
    XMean = np.mean(x)
    YMean = np.mean(y)
    XYMean = InnerProduct(x, y)
    return XYMean - XMean * YMean


def Varience(x: np.array) -> float:
    SecMom = SecondMoment(x)
    XMean = np.mean(x)
    return SecMom - (XMean**2)


def SSIM(
    x: np.array, y: np.array, c1: float = sqrt(1 / 255), c2: float = sqrt(1 / 255)
) -> float:
    LENGTH = 255

    XMean = np.mean(x)
    YMean = np.mean(y)
    C1Term = (c1 * LENGTH) ** 2
    MeanCoefficient = (2 * XMean * YMean + C1Term) / (XMean**2 + YMean**2 + C1Term)

    Cov = CoVarience(x, y)
    XVar = Varience(x)
    YVar = Varience(y)
    C2Term = (c2 * LENGTH) ** 2
    VarCoefficient = (2 * Cov + C2Term) / (XVar + YVar + C2Term)

    return MeanCoefficient * VarCoefficient


def DSSIM(
    x: np.array, y: np.array, c1: float = sqrt(1 / 255), c2: float = sqrt(1 / 255)
) -> float:
    return 1 - SSIM(x, y, c1, c2)


if __name__ == "__main__":
    image1 = cv2.imread("peppers256.bmp").astype("float16")
    image2 = cv2.imread("Peppers.png").astype("float16")
    print(f"MSE = {MSE(image1, image2)}")
    print(f"NRMSE = {NRMSE(image1, image2)}")
    print(f"PSNR = {PSNR(image1, image2)}")
    print(f"SSIM = {SSIM(image1, image2)}")
