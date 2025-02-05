import cv2
import numpy as np
from numpy.linalg import inv
import argparse


class LightAdjuster:
    def __init__(self, ImagePath: str, alpha: float) -> None:
        self.BGRImage = cv2.imread(ImagePath)

        # BGR to YCbCr
        self.BGR2YTransform = np.array(
            [[0.114, 0.587, 0.299], [0.5, -0.331, -0.169], [-0.081, -0.419, 0.5]]
        )
        self.YImage = np.tensordot(self.BGRImage, self.BGR2YTransform, ((2), (1)))

        # Adjust intensity
        self.LightAdjust(alpha)

        # YCbCr to BGR
        self.BGRImage = np.tensordot(self.YImage, inv(self.BGR2YTransform), ((2), (1)))
        self.BGRImage = np.clip(self.BGRImage, 0, 255).astype("uint8")


        cv2.imshow("Lighten / Darken", self.BGRImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def LightAdjust(self, alpha: float) -> None:
        self.YImage[:, :, 0] /= 255
        self.YImage[:, :, 0] = self.YImage[:, :, 0] ** alpha
        self.YImage[:, :, 0] *= 255


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ImagePath", type = str)
    args = parser.parse_args()
    Picture = LightAdjuster(args.ImagePath, 2)
