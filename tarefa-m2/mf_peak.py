# H(Z) = 1 + H0/2 * [1 - A2(Z)]

# A2(Z) = [-aB + (d - d*aB)Z^(-1) + Z^(-2)] / [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)]
# Y(Z)/X(Z) = [-aB + (d - d*aB)Z^(-1) + Z^(-2)] / [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)]
# Y(Z) * [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)] = [-aB + (d - d*aB)Z^(-1) + Z^(-2)] X(Z)
# Y[n] + (d - d*aB)Y[n-1] - aB*Y[n-2] = -aB*X[n] + (d - d*aB)X[n-1] + X[n-2]
# Y2[n] = -aB*X[n] + (d-d*aB)X[n-1] + X[n-2] - (d - d*aB)Y[n-1] + aB*Y[n-2]
# Y2[n] = -aB*X[n] + d(1-aB)X[n-1] + X[n-2] - d(1-aB)Y[n-1] + aB*Y[n-2]

# H(Z) = 1 + H0/2 * [1 - A2(Z)]
# H(Z) = 1 + k * [1 - [[-aB + (d - d*aB)Z^(-1) + Z^(-2)] / [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)]]]
# H(Z) = 1 + k * [[1 + (d - d*aB)Z^(-1) - aB*Z^(-2)] - [-aB + (d - d*aB)Z^(-1) + Z^(-2)] / [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)]]
# H(Z) = 1 + k * [1 + (d - d*aB)Z^(-1) - aB*Z^(-2) + aB - (d - d*aB)Z^(-1) - Z^(-2)] / [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)]]
# H(Z) = 1 + k * [1 + aB + [(d - d*aB)- (d - d*aB)]Z^(-1) - [aB + 1]Z^(-2)] / [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)]]
# H(Z) = 1 + k * [1 + aB - [aB + 1]Z^(-2)] / [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)]]
# H(Z) = 1 + [k(1 + aB) - k[aB + 1]Z^(-2)] / [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)]]
# H(Z) = [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)] + k(1 + aB) - k[aB + 1]Z^(-2)] / [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)]
# H(Z) = [1 + k(1 + aB) + (d - d*aB)Z^(-1) - (k[aB + 1] + aB)*Z^(-2)] / [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)]
# H(Z) = [1 + k(1 + aB)] + [d - d*aB]Z^(-1) - [k[aB + 1] + aB]*Z^(-2) / [1 + (d - d*aB)Z^(-1) - aB*Z^(-2)]
# H(Z) = b0 + b1Z^(-1) + b2*Z^(-2) / a0 + a1Z^(-1) + a2*Z^(-2)

# b0 = 1 + k(1 + aB)
# b1 = d - d*aB
# b2 = -k[aB + 1] - ab
# a0 = 1
# a1 = d-d*aB
# a2 = -aB

# d = - cos(2 * pi * fc / fs)

# H(Z) = b0 + b1Z^(-1) + b2*Z^(-2) / a0 + a1Z^(-1) + a2*Z^(-2)
# Y(Z)/X(Z) = b0 + b1Z^(-1) +b2*Z^(-2) / a0 + a1Z^(-1) + a2*Z^(-2)
# Y(Z) * [a0 + a1Z^(-1) + a2*Z^(-2)] = [b0 + b1Z^(-1) + b2*Z^(-2)] X(Z)
# a0Y[n] + a1Y[n-1] + a2Y[n-2] = b0X[n] + b1X[n-1] + b2X[n-2]
# a0Y[n] = b0X[n] + b1X[n-1] + b2X[n-2] - a1Y[n-1] - a2Y[n-2]
# Y[n] = [b0X[n] + b1X[n-1] + b2X[n-2] - a1Y[n-1] - a2Y[n-2] ] / a0


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

fs = 44100  # Hz
fc = 1000  # Hz
fb = 600  # Hz


def mf_peak(fc, G, fs, fb):
    V0 = 10 ** (G / 20)
    H0 = V0 - 1
    k = H0 / 2
    d = -np.cos((2 * np.pi * fc) / fs)

    # Prewarp and handle boost/cut case
    if G >= 0:
        aB = (np.tan(np.pi * fb / fs) - 1) / (np.tan(np.pi * fb / fs) + 1)
    else:
        aB = (np.tan(np.pi * fb / fs) - V0) / (np.tan(np.pi * fb / fs) + V0)

    # Derived from algebraic H(z)
    b0 = 1 + k * (1 + aB)
    b1 = d - d * aB
    b2 = -(k * (aB + 1) + aB)
    a0 = 1
    a1 = d - d * aB
    a2 = -aB

    # Numerator [a0, a1, a2], Denominator [1, b1, b2]
    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])

    return b, a


def plot_filter(b, a, fs, label):
    w, h = freqz(b, a, fs=fs)
    plt.plot(w, 20 * np.log10(abs(h)), label=label)


b_mf_peak_boost, a_mf_peak_boost = mf_peak(fc=fc, G=10, fs=fs, fb=fb)
b_mf_peak_cut, a_mf_peak_cut = mf_peak(fc=fc, G=-10, fs=fs, fb=fb)

plt.figure(figsize=(10, 6))
plot_filter(b_mf_peak_boost, a_mf_peak_boost, fs, "MF Peak Boost")
plot_filter(b_mf_peak_cut, a_mf_peak_cut, fs, "MF Peak Cut")

plt.axvline(fc, color="r", linestyle="--", label=f"Frequência de corte (fc={fc} Hz)")

plt.title("Shelving HF")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Ganho (dB)")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.show()
