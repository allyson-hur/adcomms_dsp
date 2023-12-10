from scipy.fft import fft, fftfreq, fftshift
import numpy as np

# Sampling Parameters
F_SAMPLE = 1e6  # Hz
dt = 1 / F_SAMPLE

def real_fft_plot(t, x, t_axis, f_axis, dt, t_label, f_label, ylim=1.25, xlim=100e3, abs_xlim=100.0):
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
