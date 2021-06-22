import numpy as np
from scipy import signal as scipy_signal
from stockwell import st
import pycwt


def get_snr(data, itp, dit=300):
    tmp_std = np.std(data[itp - dit:itp])
    if tmp_std > 0:
        return np.std(data[itp:itp + dit]) / tmp_std
    else:
        return 0


def prepare_dataset(signal, noise, snr, itp, transform_type):
    """ Function for adding noise over signal to provide data for training """

    T = 30  # secunde
    Fs = 100  # frecventa esantionare
    Ts = 1 / Fs
    freqs = [0, Fs / 2]  # Hz
    times = [0, T]  # secunde
    nperseg = 30
    nfft = 60
    nt = int(T / Ts)  # numar esantioane = 30 / (1/100) = 3000 esantioane
    scales = 0

    if len(signal) == nt:
        signal = signal
    else:
        signal = signal[itp:itp + nt]
        itp = 0

    np.nan_to_num(signal, nan=-50, posinf=50, neginf=-50)

    signal = 10 * (2 * (signal - np.amin(signal)) / (np.amax(signal) - np.amin(signal)) - 1)

    np.nan_to_num(noise, nan=-4, posinf=4, neginf=-4)

    noise = (noise - np.amin(noise)) / (np.amax(noise) - np.amin(noise))

    # Add white gaussian noise
    # SNR[dB] = 20 * log_10 (std_signal / std_noise) --> e gresita formula (5) din articol
    std_signal = np.std(signal)
    std_noise = std_signal / (10 ** (snr / 20))
    noise = std_noise * noise[:nt]
    #    np.random.randn(len(signal))  # noise ~ N(medie = 0, dispersie = sigma_noise) daca este zgomot Gaussian
    noisy_signal = signal + noise

    # 3000 esantioane pentru a avea 201 ferestre temporale suprapuse in care se calculeaza STFT
    # numar_ferestre_temporale ~ (size_x - nperseg)/(nperseg-overlap), overlap = nperseg // 2
    # STFT -- partea reala; partea imaginara
    if transform_type == 'STFT':
        f, t, signal_transform = scipy_signal.stft(signal, fs=Fs, nperseg=nperseg, nfft=nfft,
                                                   boundary='zeros')  # signal_shape = [31, 201]
        f, t, noisy_signal_transform = scipy_signal.stft(noisy_signal, fs=Fs, nperseg=nperseg, nfft=nfft,
                                                         boundary='zeros')  # noise_shape = signal_shape

    elif transform_type == 'S':
        signal_transform = st(signal)
        noisy_signal_transform = st(noisy_signal)

    else:
        mother = pycwt.Morlet(6)
        signal_transform, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(signal, dt=Ts, dj=1 / 12, s0=2 * Ts,
                                                                        wavelet=mother)
        noisy_signal_transform, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(noisy_signal, dt=Ts, dj=1 / 12, s0=2 * Ts,
                                                                        wavelet=mother)

    noise_transform = noisy_signal_transform - signal_transform

    return signal, noise, noisy_signal_transform, signal_transform, noise_transform, noisy_signal, scales


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

