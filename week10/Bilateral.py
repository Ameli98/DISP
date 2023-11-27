import numpy as np
import matplotlib.pyplot as plt
import ctypes as ct


def Bilateral(Signal: np.array, k1: float, k2: float, L: int) -> np.array:
    assert Signal.ndim == 1
    Output = np.zeros_like(Signal)

    for n in range(L):
        Guassian = -k1 * np.arange(-n, L + 1) ** 2
        Difference = -k2 * (Signal[n] - Signal[: n + L + 1]) ** 2
        Impulse = np.exp(Guassian + Difference)
        Impulse /= np.sum(Impulse)

        for i, m in enumerate(range(n + L + 1)):
            Output[n] += Signal[m] * Impulse[i]

    for n in range(L, Signal.size - L):
        Guassian = -k1 * np.arange(-L, L + 1) ** 2
        Difference = -k2 * (Signal[n] - Signal[n - L : n + L + 1]) ** 2
        Impulse = np.exp(Guassian + Difference)
        Impulse /= np.sum(Impulse)

        for i, m in enumerate(range(n - L, n + L + 1)):
            Output[n] += Signal[m] * Impulse[i]

    for n in range(Signal.size - L, Signal.size):
        Guassian = -k1 * np.arange(-L, Signal.size - n) ** 2
        Difference = -k2 * (Signal[n] - Signal[n - L :]) ** 2
        Impulse = np.exp(Guassian + Difference)
        Impulse /= np.sum(Impulse)

        for i, m in enumerate(range(n - L, Signal.size)):
            Output[n] += Signal[m] * Impulse[i]

    return Output


if __name__ == "__main__":
    # Create Signal
    X = np.concatenate((np.ones((50)), np.zeros((50))))
    NOISEAMP = 0.2
    Y = X + (np.random.rand(X.size) - 0.5) * NOISEAMP

    # Filter
    k1, k2, L = 0.1, 5, 3
    Z = Bilateral(Y, k1, k2, L)

    # Load C code
    dll = ct.cdll.LoadLibrary("Bilateral.dll")
    dll.Bilateral.argtypes = (
        np.ctypeslib.ndpointer(),
        ct.c_size_t,
        ct.c_double,
        ct.c_double,
        ct.c_int,
    )
    dll.Bilateral.restype = None

    # Filter by C
    W = np.copy(Y)
    dll.Bilateral(W, W.size, k1, k2, L)

    # Plot
    TIME = np.arange(100)
    plt.subplot(311)
    plt.plot(TIME, X)
    plt.subplot(312)
    plt.plot(TIME, Y)
    plt.subplot(313)
    plt.plot(TIME, Z)
    plt.show()
