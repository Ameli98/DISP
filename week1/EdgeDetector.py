import numpy as np
import cv2

class EdgeDetector():
    def __init__(self, ImagePath:str):
        self.image = cv2.imread(ImagePath)
        
        mode = input("Choose mode: 'Simple', 'Sobel', or 'Laplacian'.\n")
        self.operator = None

        if mode == "Simple":
            self.direction = input("Choose direction: 'H' for horizontal, 'V' for Vertical'.\n")
            edge = self.SimpleOperator()

        elif mode == "Sobel":
            self.direction = input("Choose direction: 'H' for horizontal, 'V' for Vertical', and 'D' for diagonal.\n")
            if self.direction == "H":
                self.operator = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4
            elif self.direction == "V":
                self.operator = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 4
            else:
                self.operator = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]) / 4
            edge = self.ConvDetection()

        else:
            self.operator = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8
            edge = self.ConvDetection()
            
        edge = np.absolute(edge) * self.image.shape[2]
        gray = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Detected Edge", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def SimpleOperator(self) -> np.ndarray:
        if self.direction == 'V':
            edge = self.image[1:, :] - self.image[:-1, :]
        else:
            edge = self.image[:, 1:] - self.image[:, :-1]
        return edge
    
    def ConvDetection(self):
        edge = np.zeros((self.image.shape[0] - self.operator.shape[0] + 1, self.image.shape[1] - self.operator.shape[1] + 1, self.image.shape[2]))
        for k in range(edge.shape[2]):
            for j in range(edge.shape[1]):
                for i in range(edge.shape[0]):
                    edge[i, j, k] =  np.sum(self.image[i:i+self.operator.shape[0], j:j+self.operator.shape[1], k] * self.operator)
        return edge.astype("uint8")
    
if __name__ == "__main__":
    ImageEdge = EdgeDetector("004.BMP")