from itertools import islice

import scipy.stats as stats
import numpy as np

# Sampling Parameters
f_sample_hz = 1e6
dt = 1 / f_sample_hz
samples_per_symbol = 200
T_symbol = samples_per_symbol * dt

symbol_zeros = np.zeros((samples_per_symbol,))

# Zero-mean Gaussian noise.
variance = 1e-3
noiseRV = stats.norm(loc=0.0, scale=np.sqrt(variance))


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


# Create a binary message.
def generate_signal(message_txt: str):
    message_bits_txt = "".join([f"{ord(x):08b}" for x in message_txt])
    print(f"T: {message_txt} -> {message_bits_txt}")
    message_bits = np.array([int(b) for b in message_bits_txt])

    return message_bits, message_bits_txt


message_txt = "Hello, world."
message_bits, message_bits_txt = generate_signal(message_txt)


def PAMGenerator(symbol, k=2, samples_per_symbol=10):
    amplitudes = np.linspace(-1, 1, k)
    if symbol < 0 or symbol > (k - 1):
        raise ValueError(f"{k}-PAM must have symbols in [0, {k-1}].")
    return amplitudes[symbol] * np.ones(
        samples_per_symbol,
    )


def generate_symbols(bits_per_symbol: int):
    symbols_of_t = []
    symbol_zeros = np.zeros((samples_per_symbol,))
    N_symbols = 2**bits_per_symbol
    for symbol in range(N_symbols):
        x = PAMGenerator(symbol, k=N_symbols, samples_per_symbol=samples_per_symbol)
        symbols_of_t.append(x)

        # Note that we are concatenating our waveform with zeros before and after -
        # this will make the plot and FFT a little more obvious.
        x = np.concatenate((symbol_zeros, x, symbol_zeros))

    return symbols_of_t


bits_per_symbol = 2
symbols_of_t = generate_symbols(bits_per_symbol)


def translate_message_bits(bits_per_symbol, message_bits_txt):
    x_samples = samples_per_symbol * int(
        np.ceil(len(message_bits_txt) // bits_per_symbol)
    )
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

    return x, t


x, t = translate_message_bits(bits_per_symbol, message_bits_txt)


def transmit_signal(x, f_c=50e3):
    f_c = 50e3
    x_modulated = x * np.cos(2 * np.pi * f_c * t)
    y = x_modulated + noiseRV.rvs(len(x))

    return y, x_modulated


y, x_modulated = transmit_signal(x)


def compute_eigen_symbols(symbols_of_t):
    cov = np.cov(symbols_of_t, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov)
    principal_eig_vec = eig_vec[0]

    s0 = (symbols_of_t[0] @ principal_eig_vec.T).real
    s1 = (symbols_of_t[1] @ principal_eig_vec.T).real
    s2 = (symbols_of_t[2] @ principal_eig_vec.T).real
    s3 = (symbols_of_t[3] @ principal_eig_vec.T).real

    return np.array([s0, s1, s2, s3]), principal_eig_vec


coefs, principal_eig_vec = compute_eigen_symbols(symbols_of_t)

pca_coef_to_symbol = {
    coefs[0].real: [0, 0],
    coefs[1].real: [0, 1],
    coefs[2].real: [1, 0],
    coefs[3].real: [1, 1],
}


def decode_signal(pca_coef_to_symbol, principal_eig_vec, y):
    i = 0
    x_hat = []

    while i < len(y):
        curr_chunk = y[i : i + samples_per_symbol]
        chunk_coef = (curr_chunk @ principal_eig_vec.T).real
        x_hat.extend(pca_coef_to_symbol[coefs[np.argmin(np.abs(coefs - chunk_coef))]])
        i += samples_per_symbol

    return x_hat


x_hat = decode_signal(pca_coef_to_symbol, principal_eig_vec, y)


def bits2a(b):
    return "".join(chr(int("".join(x), 2)) for x in zip(*[iter(b)] * 8))


bitstring = "".join(map(str, x_hat))
string = bits2a(bitstring)
print(f"R: {string} -> {bitstring}")

tx_avg_power = np.mean(x_modulated**2.0)
noise_power = variance
snr = tx_avg_power / noise_power
ber = np.sum(x_hat != message_bits) / len(message_bits)

print(f"{tx_avg_power=}")
print(f"{noise_power=}")
print(f"{snr=}")
print(f"{ber=}")
