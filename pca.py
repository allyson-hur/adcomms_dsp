from itertools import islice
import scipy.stats as stats
import numpy as np
from functools import cached_property
import matplotlib.pyplot as plt


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def PAMGenerator(symbol, k=2, samples_per_symbol=10):
    amplitudes = np.linspace(-1, 1, k)
    if symbol < 0 or symbol > (k - 1):
        raise ValueError(f"{k}-PAM must have symbols in [0, {k-1}].")
    return amplitudes[symbol] * np.ones(
        samples_per_symbol,
    )


def bits2a(b):
    return "".join(chr(int("".join(x), 2)) for x in zip(*[iter(b)] * 8))


class PCARunner:
    F_SAMPLE = 1e6  # Hz
    SAMPLES_PER_SYMBOL = 200
    BITS_PER_SYMBOL = 2
    F_C = 50e3  # Hz

    def __init__(self, variance, message_txt):
        self.message_txt = message_txt
        self.variance = variance
        self.noiseRV = stats.norm(loc=0.0, scale=np.sqrt(variance))

    @cached_property
    def dt(self):
        return 1 / self.F_SAMPLE

    @cached_property
    def T_symbol(self):
        return self.SAMPLES_PER_SYMBOL * self.dt

    @cached_property
    def message_bits_txt(self):
        return "".join([f"{ord(x):08b}" for x in self.message_txt])

    @cached_property
    def message_bits(self):
        return np.array([int(b) for b in self.message_bits_txt])

    @cached_property
    def symbols_of_t(self):
        symbols_of_t = []
        N_symbols = 2**self.BITS_PER_SYMBOL
        for symbol in range(N_symbols):
            x = PAMGenerator(
                symbol, k=N_symbols, samples_per_symbol=self.SAMPLES_PER_SYMBOL
            )
            symbols_of_t.append(x)

        return symbols_of_t

    @cached_property
    def x(self):
        x_samples = self.SAMPLES_PER_SYMBOL * int(
            np.ceil(len(self.message_bits_txt) // self.BITS_PER_SYMBOL)
        )
        x = np.zeros(
            x_samples,
        )

        for i, symbol in enumerate(
            batched(self.message_bits_txt, self.BITS_PER_SYMBOL)
        ):
            if not symbol:
                continue
            symbol = int("".join(symbol), 2)
            i_start = i * self.SAMPLES_PER_SYMBOL
            i_end = (i + 1) * self.SAMPLES_PER_SYMBOL
            x[i_start:i_end] = self.symbols_of_t[symbol]

        return x

    @cached_property
    def t(self):
        x_samples = self.SAMPLES_PER_SYMBOL * int(
            np.ceil(len(self.message_bits_txt) // self.BITS_PER_SYMBOL)
        )
        return np.linspace(0, self.dt * x_samples, x_samples)

    @cached_property
    def x_modulated(self):
        return self.x * np.cos(2 * np.pi * self.F_C * self.t)

    @cached_property
    def y(self):
        return self.x_modulated + self.noiseRV.rvs(len(self.x))

    @cached_property
    def principal_eig_vec(self):
        cov = np.cov(self.symbols_of_t, rowvar=False)
        _, eig_vec = np.linalg.eig(cov)
        return eig_vec[0]

    @cached_property
    def s0_coef(self):
        return (self.symbols_of_t[0] @ self.principal_eig_vec.T).real

    @cached_property
    def s1_coef(self):
        return (self.symbols_of_t[1] @ self.principal_eig_vec.T).real

    @cached_property
    def s2_coef(self):
        return (self.symbols_of_t[2] @ self.principal_eig_vec.T).real

    @cached_property
    def s3_coef(self):
        return (self.symbols_of_t[3] @ self.principal_eig_vec.T).real

    @cached_property
    def symbol_coefs(self):
        return np.array([self.s0_coef, self.s1_coef, self.s2_coef, self.s3_coef])

    @cached_property
    def pca_coef_to_symbol(self):
        return {
            self.s0_coef.real: [0, 0],
            self.s1_coef.real: [0, 1],
            self.s2_coef.real: [1, 0],
            self.s3_coef.real: [1, 1],
        }

    @cached_property
    def x_hat(self):
        i = 0
        x_hat = []

        while i < len(self.y):
            curr_chunk = self.y[i : i + self.SAMPLES_PER_SYMBOL]
            chunk_coef = (curr_chunk @ self.principal_eig_vec.T).real
            x_hat.extend(
                self.pca_coef_to_symbol[
                    self.symbol_coefs[np.argmin(np.abs(self.symbol_coefs - chunk_coef))]
                ]
            )
            i += self.SAMPLES_PER_SYMBOL

        return x_hat

    @cached_property
    def decoded_bitstring(self):
        return "".join(map(str, self.x_hat))

    @cached_property
    def decoded_string(self):
        return bits2a(self.decoded_bitstring)

    @cached_property
    def tx_avg_power(self):
        return np.mean(self.x_modulated**2.0)

    @cached_property
    def noise_power(self):
        return self.variance

    @cached_property
    def snr(self):
        return self.tx_avg_power / self.noise_power

    @cached_property
    def ber(self):
        return np.sum(self.x_hat != self.message_bits) / len(self.message_bits)

    def display_error_metrics(self):
        print(f"{self.tx_avg_power=}")
        print(f"{self.noise_power=}")
        print(f"{self.snr=}")
        print(f"{self.ber=}")


def run_sweep():
    snrs = []
    bers = []

    for var in (vars := np.geomspace(1e-5, 1e2, 8)):
        pca = PCARunner(variance=var, message_txt="Hello, world.")
        snrs.append(pca.snr)
        bers.append(pca.ber)

    _, ax = plt.subplots(2, 1)

    ax[0].plot(vars, snrs)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set(xlabel="Variance", ylabel="Signal-to-Noise Ratio")
    ax[1].plot(vars, bers)
    ax[1].set_xscale("log")
    ax[1].set(xlabel="Variance", ylabel="Bit Error Rate")

    plt.show()


def main():
    pca = PCARunner(variance=1e-3, message_txt="Hello, world.")
    pca.display_error_metrics()


if __name__ == "__main__":
    run_sweep()
