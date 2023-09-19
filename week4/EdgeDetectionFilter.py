import numpy as np
import matplotlib.pyplot as plt
from Filter import Filter


if __name__ == "__main__":
    TIME = np.arange(-30, 101, 1)
    OFFSET1 = 30
    X0 = np.zeros_like(TIME)
    X0[-10+OFFSET1:21+OFFSET1] = 1
    X0[50+OFFSET1:81+OFFSET1] = 1
    plt.subplot(321)
    plt.plot(TIME, X0)

    # Case 1 : low noise -> short impulse
    An = 0.2
    Noise = (np.random.rand(*X0.shape) - 0.5) * An
    X1 = X0 + Noise
    plt.subplot(323)
    plt.plot(TIME, X1)

    Sigma = 0.5
    Length = 4
    Y1 = Filter(X1, Sigma, Length)
    plt.subplot(324)
    plt.plot(TIME, Y1)

    # Case 2 : high noise -> long impulse
    An = 0.5
    Noise = (np.random.rand(*X0.shape) - 0.5) * An
    X2 = X0 + Noise
    plt.subplot(325)
    plt.plot(TIME, X2)

    Sigma = 0.2
    Length = 10
    Y2 = Filter(X2, Sigma, Length)
    plt.subplot(326)
    plt.plot(TIME, Y2)

    plt.show()