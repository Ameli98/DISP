#include <stdlib.h>
#include <math.h>
#include <stdio.h>

void Bilateral(double *Signal, size_t SignalLength, double k1, double k2, int L) {
    size_t ARRAYSIZE = 2 * L + 1;

    double Output[SignalLength];
    for (int n = 0; n < SignalLength; n++) {
        Output[n] = 0;
    }
    double Decay[ARRAYSIZE], C;
    for (int n = 0; n < ARRAYSIZE; n++) {
        C = 0;
    }

    int DecayIndex;
    /* Get the window function for each n. */
    for (int n = 0; n < SignalLength; n++) {
        for (int m = n - L; m <= n + L; m++) {
            DecayIndex = m - (n - L);
            if (m < 0 || m >= SignalLength)
                continue;
            Decay[DecayIndex] = - k1 * (n - m) * (n - m) - k2 * (Signal[n] - Signal[m]) * (Signal[n] - Signal[m]);
            Decay[DecayIndex] = exp(Decay[DecayIndex]);
            C += Decay[DecayIndex];
        }
        C = 1 / C;
    

        /* Convolution */
        for (int m = n - L; m <= n + L; m++) {
            DecayIndex = m - (n - L);
            if (m < 0 || m >= SignalLength)
                continue;
            Output[n] += Signal[m] * Decay[DecayIndex];
        }
        Output[n] *= C; 
    }

    for(int n = 0; n < SignalLength; n++) {
        Signal[n] = Output[n];
    }
}