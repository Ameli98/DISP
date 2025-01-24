import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def EroDila(Image: np.array, Dila: bool = False) -> np.array:
    Result = np.zeros_like(Image)
    func = max if Dila else min

    # 4 corners
    Result[0, 0] = func(Image[0, 0], Image[0, 1], Image[1, 0])
    Result[0, -1] = func(Image[0, -1], Image[0, -2], Image[1, -1])
    Result[-1, 0] = func(Image[-1, 0], Image[-1, 1], Image[-2, 0])
    Result[-1, -1] = func(Image[-1, -1], Image[-1, -2], Image[-2, -1])

    # 4 edges
    for m in range(Image.shape[0])[1:-1]:
        Result[m, 0] = func(Image[m, 0], Image[m, 1], Image[m - 1, 0], Image[m + 1, 0])
        Result[m, -1] = func(
            Image[m, -1], Image[m, -2], Image[m - 1, -1], Image[m + 1, -1]
        )
    for n in range(Image.shape[1])[1:-1]:
        Result[0, n] = func(Image[0, n], Image[1, n], Image[0, n - 1], Image[0, n + 1])
        Result[-1, n] = func(
            Image[-1, n], Image[-2, n], Image[-1, n - 1], Image[-1, n + 1]
        )

    # Inside indices
    for m in range(Image.shape[0])[1:-1]:
        for n in range(Image.shape[1])[1:-1]:
            Result[m, n] = func(
                Image[m, n],
                Image[m - 1, n],
                Image[m + 1, n],
                Image[m, n - 1],
                Image[m, n + 1],
            )
    return Result


def Closing(Image: np.array, times: int = 3) -> np.array:
    for i in range(times):
        Image = EroDila(Image, True)
    for i in range(times):
        Image = EroDila(Image)
    return Image


def Opening(Image: np.array, times: int = 3) -> np.array:
    for i in range(times):
        Image = EroDila(Image)
    for i in range(times):
        Image = EroDila(Image, True)
    return Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ImagePath", type = str)
    args = parser.parse_args()

    Image = cv2.imread(args.ImagePath)
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY).astype("float16")

    Image1 = np.concatenate((EroDila(Image), EroDila(Image, True)), axis=1)
    Image2 = np.concatenate((Opening(Image), Closing(Image)), axis=1)

    Result = np.concatenate((Image1, Image2), axis=0).astype("uint8")
    cv2.imshow("Result", Result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
