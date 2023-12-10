from itertools import islice
import scipy.stats as stats
import numpy as np
from functools import cached_property
import matplotlib.pyplot as plt
from enum import Enum
from pathlib import Path


def batched(iterable, n):
    """
    Batch data into lists of a specified length.

    Parameters:
    - iterable: The input iterable to be batched.
    - n (int): The desired batch size.
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def PAMGenerator(symbol, k=2, samples_per_symbol=10):
    """
    Generate a PAM signal for a given symbol.

    Parameters:
    - symbol (int): The symbol value for which the PAM signal is generated.
    - k (int): The number of amplitude levels in the PAM signal.
    - samples_per_symbol (int): The number of samples per symbol in the generated signal.

    Returns:
    - pam_signal: The PAM signal for the specified symbol.
    """
    amplitudes = np.linspace(-1, 1, k)
    if symbol < 0 or symbol > (k - 1):
        raise ValueError(f"{k}-PAM must have symbols in [0, {k-1}].")
    return amplitudes[symbol] * np.ones(
        samples_per_symbol,
    )


def bits2a(b):
    """
    Convert a binary string to ASCII.
    
    Parameters:
    - b (str): The binary string to be converted.
    
    Returns:
    - string: The ASCII string.
    """
    return "".join(chr(int("".join(x), 2)) for x in zip(*[iter(b)] * 8))

# Plotting function
def real_fft_plot(t, x, t_axis, f_axis, t_label='$x(t)$', f_label='$X(f)$', ylim=1.25, xlim=100e3, abs_xlim=100.0):
	"""Plots the Fast Fourier Transform in the frequency domain. Stores only one half of the FFT (F{real} is always symmetric)."""
	X = fftshift(fft(x))
	f = fftshift(fftfreq(len(X), dt))
	t_axis.plot(t, x, label=t_label)
	t_axis.set_ylim(-ylim, ylim)
	t_axis.set_title(t_label)
	t_axis.set_xlabel('$t$')

	f_axis.plot(f[len(X)//2:], np.abs(X[len(X)//2:]), label=f_label)
	f_axis.set_title(f_label)
	f_axis.set_xlabel('$f (Hz)$')
	f_axis.set_xlim(0, xlim)
	f_axis.set_ylim(0, abs_xlim)


class TransferFunctions(Enum):
    """
    Contains the different transfer functions for our channel models that the 
    modulated signal is sent through. Each transfer function is represented
    as an enum member with a corresponding string value.
    """
    DiracDelta = "dirac_delta"
    Sinc = "sinc"
    Cosine = "cosine"
    Sine = "sine"
    Triangle = "triangle"


class PCARunner:
    """
    Contains functions for simulating a communication system that uses PCA for
    decoding modulated signals.
    """
    F_SAMPLE = 1e6  # Hz
    SAMPLES_PER_SYMBOL = 200
    BITS_PER_SYMBOL = 2
    F_C = 50e3  # Hz

    def __init__(self, variance, message_txt, transfer_func):
        """
        Initialize the class with parameters for the variance, message text,
        and a specified transfer function.

        Args:
            variance (float): the variance of the signal that affects noise.
            message_txt (str): the original text message to be transmitted.
            transfer_func (Enum): the transfer function applied to modulated signal.
        """
        self.message_txt = message_txt
        self.variance = variance
        self.noiseRV = stats.norm(loc=0.0, scale=np.sqrt(variance))
        self.transfer_func = transfer_func

    @cached_property
    def dt(self):
        """
        Calculate the sampling period.

        Returns:
            A float representing the sampling period.
        """
        return 1 / self.F_SAMPLE

    @cached_property
    def T_symbol(self):
        """
        Calculate the symbol period.

        Returns:
            A float representing the symbol period.
        """
        return self.SAMPLES_PER_SYMBOL * self.dt

    @cached_property
    def message_bits_txt(self):
        """
        Convert message text to a binary string.

        Returns:
            A string of the binary representation of the original message.
        """
        return "".join([f"{ord(x):08b}" for x in self.message_txt])

    @cached_property
    def message_bits(self):
        """
        Convert the binary representation of the original message into an array
        of integers, where each integer represents a binary bit in the message.

        Returns:
            An integer array representing the binary bits of the original message.
        """
        return np.array([int(b) for b in self.message_bits_txt])

    @cached_property
    def symbols_of_t(self):
        """
        Generate symbols and store them in a list.

        Returns:
            A list of the generated symbols.
        """
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
        """
        Divide the binary message into chunks, then convert each chunk into an
        integer. Then, find the PAM symbol corresopnding to the integer.

        Returns:
            An array of PAM symbols.
        """
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
        """
        Get the time values associated with each sample in the signal.

        Returns:
            An array representing the time values.
        """
        x_samples = self.SAMPLES_PER_SYMBOL * int(
            np.ceil(len(self.message_bits_txt) // self.BITS_PER_SYMBOL)
        )
        return np.linspace(0, self.dt * x_samples, x_samples)

    @cached_property
    def x_modulated(self):
        """
        Generate the modulated signal by multiplying the generated symbol by a 
        cosine wave at the carrier frequency.

        Returns:
            An array representing the modulated signal.
        """
        return self.x * np.cos(2 * np.pi * self.F_C * self.t)

    @cached_property
    def y(self):
        h_of_t = None
        if self.transfer_func == TransferFunctions.DiracDelta:
            h_of_t = np.ones(np.shape(self.t)[0])
        elif self.transfer_func == TransferFunctions.Sinc:
            h_of_t = np.sinc(self.t)
        elif self.transfer_func == TransferFunctions.Sine:
            h_of_t = np.sin(self.t)
        elif self.transfer_func == TransferFunctions.Cosine:
            h_of_t = np.cos(self.t)
        elif self.transfer_func == TransferFunctions.Triangle:
            l = np.shape(self.t)[0]
            h_of_t = np.zeros(l)
            h_of_t[: l // 2] = np.linspace(0, 1, l // 2)
            h_of_t[l // 2 :] = np.linspace(1, 0, l // 2)
        return self.x_modulated * h_of_t + self.noiseRV.rvs(len(self.x))

    @cached_property
    def principal_eig_vec(self):
        """
        Compute the principal eigenvector of the covariance matrix. This
        eigenvector represents the direction in the feature space where along
        which the data varies the most.

        Returns:
            The first row of the matrix of eigenvectors.
        """
        cov = np.cov(self.symbols_of_t, rowvar=False)
        _, eig_vec = np.linalg.eig(cov)
        return eig_vec[0]

    @cached_property
    def s0_coef(self):
        """
        Calculate the coefficient associated with the first symbol when
        projecting it onto the principal eigenvector.

        Returns:
            The coefficient associated with the first symbol.
        """
        return (self.symbols_of_t[0] @ self.principal_eig_vec.T).real

    @cached_property
    def s1_coef(self):
        """
        Calculate the coefficient associated with the second symbol when
        projecting it onto the principal eigenvector.

        Returns:
            The coefficient associated with the second symbol.
        """
        return (self.symbols_of_t[1] @ self.principal_eig_vec.T).real

    @cached_property
    def s2_coef(self):
        """
        Calculate the coefficient associated with the third symbol when
        projecting it onto the principal eigenvector.

        Returns:
            The coefficient associated with the third symbol.
        """
        return (self.symbols_of_t[2] @ self.principal_eig_vec.T).real

    @cached_property
    def s3_coef(self):
        """
        Calculate the coefficient associated with the fourth symbol when
        projecting it onto the principal eigenvector.

        Returns:
            The coefficient associated with the fourth symbol.
        """
        return (self.symbols_of_t[3] @ self.principal_eig_vec.T).real

    @cached_property
    def symbol_coefs(self):
        """
        Create an array of the four coefficients associated with the symbols.

        Returns:
            An array of the coefficients.
        """
        return np.array([self.s0_coef, self.s1_coef, self.s2_coef, self.s3_coef])

    @cached_property
    def pca_coef_to_symbol(self):
        """
        Map coefficients associated with specific symbols to corresponsding
        binary representations.

        Returns:
            A dictionary of the coefficients with corresponding binary 
            representations.
        """
        return {
            self.s0_coef.real: [0, 0],
            self.s1_coef.real: [0, 1],
            self.s2_coef.real: [1, 0],
            self.s3_coef.real: [1, 1],
        }

    @cached_property
    def x_hat(self):
        """
        Decode the received signal. The received signal is divided into chunks,
        then for each chunk, the corresponding coefficient is calculated. The
        associated symbol is determined based on the closest match between the
        calculated coefficient and the predefined symbol coefficients.

        Returns:
            A list of reconstructed symbols.
        """
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
        """
        Convert list of reconstructed symbols into a binary bitstring.

        Returns:
            A bitstring representing the reconstructed symbols.
        """
        return "".join(map(str, self.x_hat))

    @cached_property
    def decoded_string(self):
        """
        Convert the decoded bitstring to ASCII.

        Returns:
            The ASCII string representing the decoded message.
        """
        return bits2a(self.decoded_bitstring)

    @cached_property
    def tx_avg_power(self):
        """
        Calculate the average transmitted power.

        Returns:
            A float representing the average transmitted power.
        """
        return np.mean(self.x_modulated**2.0)

    @cached_property
    def noise_power(self):
        """
        Get the power of the additive noise.

        Returns:
            A float representing the noise power.
        """
        return self.variance

    @cached_property
    def snr(self):
        """
        Calculate the signal-to-noise ratio.

        Returns:
            A float representing the signal-to-noise ratio.
        """
        return self.tx_avg_power / self.noise_power

    @cached_property
    def ber(self):
        """
        Calculate the bit error rate.

        Returns:
            A float representing the bit error rate.
        """
        return np.sum(self.x_hat != self.message_bits) / len(self.message_bits)

    def display_error_metrics(self):
        """
        Display the calculated performance metrics.
        """
        print(f"{self.tx_avg_power=}")
        print(f"{self.noise_power=}")
        print(f"{self.snr=}")
        print(f"{self.ber=}")


def run_sweep():
    """
    Perform a parameter sweep across different levels of variance. Save and plot
    the signal-to-noise ratio and bit error rate for each level of variance and
    each transfer function. The images are saved in the "images" folder with 
    filenames based on the transfer functions.
    """
    for tf in TransferFunctions.__members__.values():
        snrs = []
        bers = []

        for var in (vars := np.geomspace(1e-5, 1e2, 8)):
            pca = PCARunner(
                variance=var,
                message_txt="Hello, world.",
                transfer_func=tf,
            )
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
        plt.suptitle(f"h(t): {tf.name}")

        plt.savefig(Path(__file__).parent / f"images/{tf.value}")


def main():
    pca = PCARunner(
        variance=1e-3,
        message_txt="Hello, world.",
        transfer_func=TransferFunctions.Triangle,
    )
    pca.display_error_metrics()


if __name__ == "__main__":
    run_sweep()
