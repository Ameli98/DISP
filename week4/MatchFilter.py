import numpy as np
import matplotlib.pyplot as plt
from Filter import MatchFilter


if __name__ == "__main__":
    # Construct signal
    Pattern = np.arange(-50, 51) / 50
    plt.subplot(311)
    plt.plot(Pattern)

    Signal = np.zeros((901,))
    Signal[100:201] = 1
    Signal[300:401] = Pattern
    Signal[500:601] = np.arange(50, -51, -1) / 50
    Signal[700:801] = np.sin(np.arange(0, 101) * np.pi / 50)
    plt.subplot(312)
    plt.plot(Signal)

    Result = MatchFilter(Signal, Pattern)
    plt.subplot(313)
    plt.plot(Result)
    plt.show()
