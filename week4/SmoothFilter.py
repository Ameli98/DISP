import numpy as np
import matplotlib.pyplot as plt
from Filter import Filter

if __name__ == "__main__":
    TIME = np.arange(-50, 101)
    X0 = TIME * 0.1
    plt.subplot(331)
    plt.plot(TIME, X0)

    # Filters for short and long term feature
    SHORTSIGMA, SHORTLENGTH = 0.8, 4
    LONGSIGMA, LONGLENGTH = 0.4, 10

    # Case 1: Low noise
    An = 2
    Noise = (np.random.rand(*X0.shape) - 0.5) * An
    X1 = X0 + Noise
    plt.subplot(334)
    plt.plot(TIME, X1)

    Y1S = Filter(X1, SHORTSIGMA, SHORTLENGTH, False)
    plt.subplot(335)
    plt.plot(TIME, Y1S)

    Y1L = Filter(X1, LONGSIGMA, LONGLENGTH, False)
    plt.subplot(336)
    plt.plot(TIME, Y1L)

    # Case 2: High noise
    An = 10
    Noise = (np.random.rand(*X0.shape) - 0.5) * An
    X2 = X0 + Noise
    plt.subplot(337)
    plt.plot(TIME, X2)

    Y2S = Filter(X2, SHORTSIGMA, SHORTLENGTH, False)
    plt.subplot(338)
    plt.plot(TIME, Y2S)

    Y2L = Filter(X2, LONGSIGMA, LONGLENGTH, False)
    plt.subplot(339)
    plt.plot(TIME, Y2L)

    plt.show()
