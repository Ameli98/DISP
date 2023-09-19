import wave
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt


if __name__ == "__main__":
    Wavefile = wave.open("SoundEffect.wav", "rb")
    LENGTH = Wavefile.getnframes()
    Frame = Wavefile.readframes(LENGTH)
    Audio = np.frombuffer(Frame, dtype=np.int16)
    Audio = Audio / max(abs(Audio))


    FS = Wavefile.getframerate()
    num_frame = Wavefile.getnframes()
    Audio = np.reshape(Audio, (num_frame, -1))

    TIME = np.arange(0, LENGTH) / FS
    plt.subplot(121)
    plt.plot(TIME, Audio[:,1])

    Spectrum = np.abs(fft(Audio[:,1])) / FS
    N0 = int(np.ceil(LENGTH) / 2)
    Spectrum = np.concatenate((Spectrum[N0:], Spectrum[:N0]))
    FREQ = np.linspace(N0-Spectrum.size, N0-1, Spectrum.size)/Spectrum.size*FS
    plt.subplot(122)
    plt.plot(FREQ, Spectrum)
    plt.show()
    
