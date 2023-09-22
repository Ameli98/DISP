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

def CORR(Signal:np.array, Pattern:np.array) -> np.array:
    return np.convolve(Signal, np.flip(Pattern), "same")

def MatchFilter(Signal:np.array, Pattern:np.array) -> np.array:
    Pattern = Pattern - np.mean(Pattern)
    PATTERNLENGTH = np.sum(Pattern ** 2)

    MeanFilter = np.ones_like(Pattern) / Pattern.size
    LOCALMEAN = CORR(Signal, MeanFilter)
    DIFF = Signal - LOCALMEAN
    SQUAREERROR = np.sum(DIFF ** 2)

    return CORR(Signal, Pattern) / ((PATTERNLENGTH * SQUAREERROR) ** 0.5)