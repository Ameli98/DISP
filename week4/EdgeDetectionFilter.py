import numpy as np
import matplotlib.pyplot as plt
from Filter import Filter


if __name__ == "__main__":
    TIME = np.arange(-30, 101, 1)
    OFFSET1 = 30
    X0 = np.zeros_like(TIME)
    X0[-10 + OFFSET1 : 21 + OFFSET1] = 1
    X0[50 + OFFSET1 : 81 + OFFSET1] = 1
    plt.subplot(331)
    plt.plot(TIME, X0)

    SHORTSIGMA, SHORTLENGTH = 0.5, 4
    LONGSIGMA, LONGLENGTH = 0.2, 10

    # Case 1 : low noise
    An = 0.2
    Noise = (np.random.rand(*X0.shape) - 0.5) * An
    X1 = X0 + Noise
    plt.subplot(334)
    plt.plot(TIME, X1)

    Y1S = Filter(X1, SHORTSIGMA, SHORTLENGTH)
    plt.subplot(335)
    plt.plot(TIME, Y1S)

    Y1L = Filter(X1, LONGSIGMA, LONGLENGTH)
    plt.subplot(336)
    plt.plot(TIME, Y1L)

    # Case 2 : high noise
    An = 0.5
    Noise = (np.random.rand(*X0.shape) - 0.5) * An
    X2 = X0 + Noise
    plt.subplot(337)
    plt.plot(TIME, X2)

    Y2S = Filter(X2, SHORTSIGMA, SHORTLENGTH)
    plt.subplot(338)
    plt.plot(TIME, Y2S)

    Y2L = Filter(X2, LONGSIGMA, LONGLENGTH)
    plt.subplot(339)
    plt.plot(TIME, Y2L)

    plt.show()
