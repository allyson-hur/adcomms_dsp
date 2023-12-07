# Plotting
import matplotlib
from matplotlib import rc

rc("animation", html="html5")
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from IPython.display import Audio


def real_fft_plot(
    t,
    x,
    t_axis,
    f_axis,
    t_label="$x(t)$",
    f_label="$X(f)$",
    ylim=1.25,
    xlim=100e3,
    abs_xlim=100.0,
):
    # Store only one half of the FFT (F{real} is always symmetric).
    X = fftshift(fft(x))
    f = fftshift(fftfreq(len(X), dt))
    t_axis.plot(t, x, label=t_label)
    t_axis.set_ylim(-ylim, ylim)
    t_axis.set_title(t_label)
    t_axis.set_xlabel("$t$")

    f_axis.plot(f[len(X) // 2 :], np.abs(X[len(X) // 2 :]), label=f_label)
    f_axis.set_title(f_label)
    f_axis.set_xlabel("$f (Hz)$")
    f_axis.set_xlim(0, xlim)
    f_axis.set_ylim(0, abs_xlim)


# Python helpers
from itertools import islice, zip_longest


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


# Science!
import scipy
import scipy.stats as stats
from scipy.signal import convolve, lfilter
from scipy.special import erf
from scipy.fft import fft, fftfreq, fftshift
import numpy as np

# Audio
import librosa
import librosa.display

# Hyperparameters/Constants/Distributions
epsilon = 1e-3  # What is "zero" anyway?

# Sampling Parameters
f_sample_hz = 1e6
dt = 1 / f_sample_hz

# Why don't we have to demodulate?
# Why do greater number of samples not improve the BER?
samples_per_symbol = 200
T_symbol = samples_per_symbol * dt

symbol_ones = np.ones((samples_per_symbol,))
symbol_zeros = np.zeros((samples_per_symbol,))

# Zero-mean Gaussian noise.
variance = 1e-3
noiseRV = stats.norm(loc=0.0, scale=np.sqrt(variance))

# Create a binary message.
message_txt = "Hello, world."
message_bits_txt = "".join([f"{ord(x):08b}" for x in message_txt])
print(f"{message_txt} -> {message_bits_txt}")
message_bits = np.array([int(b) for b in message_bits_txt])


def PAMGenerator(symbol, k=2, samples_per_symbol=10):
    amplitudes = np.linspace(-1, 1, k)
    if symbol < 0 or symbol > (k - 1):
        raise ValueError(f"{k}-PAM must have symbols in [0, {k-1}].")
    return amplitudes[symbol] * np.ones(
        samples_per_symbol,
    )


bits_per_symbol = 2
N_symbols = 2**bits_per_symbol

fig, ax = plt.subplots(2, N_symbols)
plt.subplots_adjust(hspace=1.05)
symbols_of_t = []
symbol_zeros = np.zeros((samples_per_symbol,))
for symbol in range(N_symbols):
    x = PAMGenerator(symbol, k=N_symbols, samples_per_symbol=samples_per_symbol)
    symbols_of_t.append(x)

    # Note that we are concatenating our waveform with zeros before and after -
    #  this will make the plot and FFT a little more obvious.
    x = np.concatenate((symbol_zeros, x, symbol_zeros))
    t = np.linspace(-1 * T_symbol, 2 * T_symbol, len(x))

    real_fft_plot(
        t,
        x,
        ax[0][symbol],
        ax[1][symbol],
        t_label="$x_{%d}[t]$" % symbol,
        f_label="$|X_{%d}(f)|$" % symbol,
    )

# plt.suptitle("Symbols and Fourier Transforms")
# plt.show()
# plt.close()

x_samples = samples_per_symbol * int(np.ceil(len(message_bits_txt) // bits_per_symbol))
x = np.zeros(
    x_samples,
)
t = np.linspace(0, dt * x_samples, x_samples)
for i, symbol in enumerate(batched(message_bits_txt, bits_per_symbol)):
    if not symbol:
        continue
    symbol = int("".join(symbol), 2)
    i_start = i * samples_per_symbol
    i_end = (i + 1) * samples_per_symbol
    x[i_start:i_end] = symbols_of_t[symbol]

# fig, ax = plt.subplots(2,1)
# plt.subplots_adjust(hspace=1.05)
# real_fft_plot(t, x, ax[0], ax[1], abs_xlim=1.1e3)
# plt.suptitle("x(t) and X(f)")
# plt.show()
# plt.close()

f_c = 50e3
x_modulated = x * np.cos(2 * np.pi * f_c * t)
fig, ax = plt.subplots(2, 1)
# plt.subplots_adjust(hspace=1.05)
# real_fft_plot(t, x_modulated, ax[0], ax[1], abs_xlim=1.1e3)
# plt.suptitle("Modulated x(t) and X(f)")
# plt.show()
# plt.close()

# Start with just additive noise, and no delays. Then make it more complicated!
y = x_modulated + noiseRV.rvs(len(x))
# y = x_modulated

fig, ax = plt.subplots(2, 1)
# plt.subplots_adjust(hspace=1.05)
# real_fft_plot(t, y, ax[0], ax[1], abs_xlim=1.1e3)
# plt.suptitle("Modulated y(t) and X(f)")
# plt.show()
# plt.close()

I_of_t = np.cos(2 * np.pi * f_c * t) * y
Q_of_t = np.sin(2 * np.pi * f_c * t) * y  # You can ignore this for now.
# plt.plot(t, I_of_t)
# plt.show()

cov = np.cov(symbols_of_t, rowvar=False)
eig_val, eig_vec = np.linalg.eig(cov)
principal_eig_vecs = eig_vec[0]

s0_coef = (symbols_of_t[0] @ principal_eig_vecs.T).real
s1_coef = (symbols_of_t[1] @ principal_eig_vecs.T).real
s2_coef = (symbols_of_t[2] @ principal_eig_vecs.T).real
s3_coef = (symbols_of_t[3] @ principal_eig_vecs.T).real

pca_coef_to_symbol = {
    s0_coef.real: [0, 0],
    s1_coef.real: [0, 1],
    s2_coef.real: [1, 0],
    s3_coef.real: [1, 1],
}

i = 0
m_hat = []

coefs = np.array([s0_coef, s1_coef, s2_coef, s3_coef])

while i < len(y):
    curr_chunk = y[i : i + samples_per_symbol]
    chunk_coef = (curr_chunk @ principal_eig_vecs.T).real
    m_hat.extend(pca_coef_to_symbol[coefs[np.argmin(np.abs(coefs - chunk_coef))]])
    i += samples_per_symbol


def bits2a(b):
    return "".join(chr(int("".join(x), 2)) for x in zip(*[iter(b)] * 8))


bitstring = "".join(map(str, m_hat))
string = bits2a(bitstring)
print(string)


tx_avg_power = np.mean(x_modulated**2.0)
noise_power = variance
snr = tx_avg_power / noise_power
ber = np.sum(m_hat != message_bits) / len(message_bits)

print(f"{tx_avg_power=}")
print(f"{noise_power=}")
print(f"{snr=}")
print(f"{ber=}")
