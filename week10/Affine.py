import numpy as np
import cv2
import argparse


class AffineTransformer:
    def __init__(self, Image: np.array):
        if Image.ndim == 2:
            Image = np.expand_dims(Image, axis=2)
        else:
            assert Image.ndim == 3
        Shape = Image.shape[0] + 1, Image.shape[1] + 1, Image.shape[2]
        self.Input = np.zeros(Shape)
        self.Input[:-1, :-1] = Image

    def Bilinear3D(self, Position: np.array) -> np.array:
        x, y = Position[0], Position[1]
        m, n = int(np.ceil(x).item()), int(np.ceil(y).item())
        a, b = x - m, y - n

        def Interpolate(channel) -> np.array:
            return (
                (1 - a) * (1 - b) * self.Input[m, n, channel]
                + a * (1 - b) * self.Input[m + 1, n, channel]
                + (1 - a) * b * self.Input[m, n + 1, channel]
                + a * b * self.Input[m + 1, n + 1, channel]
            )

        Pixel = np.array([Interpolate(k) for k in range(self.Input.shape[2])]).reshape(self.Input.shape[2])
        
        return Pixel

    def InRange(self, Position: np.array) -> bool:
        x, y = Position[0], Position[1]
        return (
            x >= 0
            and x <= self.Input.shape[0] - 2
            and y >= 0
            and y <= self.Input.shape[1] - 2
        )

    def Rotate(self, Degree: float, CenterX, CenterY) -> np.array:
        Output = np.zeros_like(Image, dtype="uint8")
        Radius = Degree * np.pi / 180
        Transform = np.array(
            [[np.cos(Radius), np.sin(Radius)], [-np.sin(Radius), np.cos(Radius)]]
        )
        Transform = np.linalg.inv(Transform)
        for i in range(Image.shape[0]):
            for j in range(Image.shape[1]):
                Position = Transform @ np.array(
                    [[i - CenterX], [j - CenterY]]
                ) + np.array([[CenterX], [CenterY]])
                if self.InRange(Position):
                    Output[i, j] = self.Bilinear3D(Position)
        return Output

    def Sheer(self, CenterX: int, CenterY: int, ZetaX: float = 0, ZetaY: float = 0) -> np.array:
        
        Output = np.zeros_like(Image, dtype="uint8")
        Transform = np.array([[1, ZetaY], [ZetaX, 1]])
        Transform = np.linalg.inv(Transform)
        for i in range(Image.shape[0]):
            for j in range(Image.shape[1]):
                Position = Transform @ np.array(
                    [[i - CenterX], [j - CenterY]]
                ) + np.array([[CenterX], [CenterY]])
                if self.InRange(Position):
                    Output[i, j] = self.Bilinear3D(Position)
        return Output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ImagePath", type = str)
    args = parser.parse_args()

    Image = cv2.imread(args.ImagePath)
    Image = Image.astype("float16")

    Affine = AffineTransformer(Image)
    Rotated = Affine.Rotate(30, Image.shape[0] // 2, Image.shape[1] // 2)
    Sheered = Affine.Sheer(Image.shape[0] // 2, Image.shape[1] // 2, ZetaX=0.3)
    Result = np.concatenate((Rotated, Sheered), axis=0)

    cv2.imshow("Result", Result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
