import numpy as np
import matplotlib.pyplot as plt

def Filter(Signal:np.array, Decay:float, HalfLength:int, Detect:bool=True) -> np.array:
    TIME = np.arange(-HalfLength, HalfLength+1)
    ImpulseResponse = np.abs(TIME)  * (- Decay)
    ImpulseResponse = np.exp(ImpulseResponse)
    ImpulseResponse /= np.sum(ImpulseResponse)
    if Detect:
        ImpulseResponse[0:HalfLength] *= -1
        ImpulseResponse[HalfLength] = 0
    return np.convolve(Signal, ImpulseResponse, "same")
