import numpy as np
from cv2 import INTER_CUBIC
from scipy import signal as scipy_signal
from ssqueezepy import ssq_cwt
from stockwell import st
import cv2


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
    if transform_type == 0:
        f, t, signal_transform = scipy_signal.stft(signal, fs=Fs, nperseg=nperseg, nfft=nfft,
                                                   boundary='zeros')  # signal_shape = [31, 201]
        f, t, noisy_signal_transform = scipy_signal.stft(noisy_signal, fs=Fs, nperseg=nperseg, nfft=nfft,
                                                         boundary='zeros')  # noise_shape = signal_shape

    elif transform_type == 1:
        signal_transform, *_ = ssq_cwt(signal)
        noisy_signal_transform, *_ = ssq_cwt(noisy_signal)

        signal_transform = cv2.resize(np.float32(signal_transform), (201, 31), interpolation=INTER_CUBIC)
        noisy_signal_transform = cv2.resize(np.float32(noisy_signal_transform), (201, 31), interpolation=INTER_CUBIC)

    elif transform_type == 2:
        signal_transform = st(signal)
        noisy_signal_transform = st(noisy_signal)

    elif transform_type == 3:
        widths = np.linspace(1, 32)
        signal_transform = scipy_signal.cwt(signal, scipy_signal.ricker, widths)
        noisy_signal_transform = scipy_signal.cwt(noisy_signal, scipy_signal.ricker, widths)

    noise_transform = noisy_signal_transform - signal_transform

    return signal, noise, noisy_signal_transform, signal_transform, noise_transform


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

# class ToTensor(object):
#     """Converts ndarrays in sample to Tensors"""
#
#     def __call__(self, sample):
#         if len(sample) == 5:
#             stft_dict = sample
#
#             stft_dict['Zxx_signal'] = torch.from_numpy(stft_dict['Zxx_signal'])
#             stft_dict['Zxx_processed'] = torch.from_numpy(stft_dict['Zxx_processed'])
#             return {'f': sample['f'], 't': sample['t'], 'Zxx_signal': stft_dict['Zxx_signal'],
#                     'Zxx_processed': stft_dict['Zxx_processed']}
#         else:
#             signal, noise, processed = sample['signal'], sample['noise'], sample['processed']
#             return {'signal': torch.from_numpy(signal),
#                     'noise': torch.from_numpy(noise),
#                     'processed': torch.from_numpy(processed)}


# class Rescale(object):
#     """Rescales Tensor values to given min and max values"""
#
#     def __call__(self, sample):
#         if len(sample) == 4:
#             stft_dict = sample
#
#             stft_dict['Zxx_signal'] = stft_dict['Zxx_signal'].numpy()
#             stft_dict['Zxx_processed'] = stft_dict['Zxx_processed'].numpy()
#
#             min_value_proc = np.amin(stft_dict['Zxx_processed'])
#             lower_value_proc = np.amax(stft_dict['Zxx_processed']) - min_value_proc
#
#             min_value_sig = np.amin(stft_dict['Zxx_signal'])
#             lower_value_sig = np.amax(stft_dict['Zxx_signal']) - min_value_sig
#
#             stft_dict['Zxx_processed'] = np.array(
#                 [(x - min_value_proc) / lower_value_proc for x in stft_dict['Zxx_processed']])
#             stft_dict['Zxx_signal'] = np.array([(x - min_value_sig) / lower_value_sig for x in stft_dict['Zxx_signal']])
#
#             return {'f': sample['f'], 't': sample['t'], 'Zxx_signal': torch.from_numpy(stft_dict['Zxx_signal']),
#                     'Zxx_processed': torch.from_numpy(stft_dict['Zxx_processed'])}
#         else:
#             signal, noise, processed = sample['signal'].numpy(), sample['noise'].numpy(), sample['processed'].numpy()
#
#             min_value_proc = np.amin(processed)
#             lower_value_proc = np.amax(processed) - min_value_proc
#
#             min_value_sig = np.amin(signal)
#             lower_value_sig = np.amax(signal) - min_value_sig
#
#             processed_new = np.array(
#                 [20 * (x - min_value_proc) / lower_value_proc - 10 for x in processed])
#             signal_new = np.array([20 * (x - min_value_sig) / lower_value_sig - 10 for x in signal])
#
#             return {'signal': torch.from_numpy(signal_new),
#                     'noise': torch.from_numpy(noise),
#                     'processed': torch.from_numpy(processed_new)}
#
#
# class Normalize(object):
#     '''Normalizes Tensor values to given mean and standard deviation'''
#
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#
#     def __call__(self, sample):
#         if len(sample) == 4:
#             stft_dict = sample
#
#             signal_real_mean = stft_dict['Zxx_signal'].real.mean()
#             signal_real_std = stft_dict['Zxx_signal'].real.std()
#             signal_imag_mean = stft_dict['Zxx_signal'].imag.mean()
#             signal_imag_std = stft_dict['Zxx_signal'].imag.std()
#             processed_real_mean = stft_dict['Zxx_processed'].real.mean()
#             processed_real_std = stft_dict['Zxx_processed'].real.std()
#             processed_imag_mean = stft_dict['Zxx_processed'].imag.mean()
#             processed_imag_std = stft_dict['Zxx_processed'].imag.std()
#
#             stft_dict['Zxx_signal'].real = self.mean + (stft_dict['Zxx_signal'].real - signal_real_mean) * (
#                     self.std / signal_real_std)
#             stft_dict['Zxx_signal'].imag = self.mean + (stft_dict['Zxx_signal'].imag - signal_imag_mean) * (
#                     self.std / signal_imag_std)
#             stft_dict['Zxx_processed'].real = self.mean + (stft_dict['Zxx_processed'].real - processed_real_mean) * (
#                     self.std / processed_real_std)
#             stft_dict['Zxx_processed'].imag = self.mean + (stft_dict['Zxx_processed'].imag - processed_imag_mean) * (
#                     self.std / processed_imag_std)
#
#             return {'f': sample['f'], 't': sample['t'], 'Zxx_signal': stft_dict['Zxx_signal'],
#                     'Zxx_processed': stft_dict['Zxx_processed']}
#         else:
#
#             signal, noise, processed = sample['signal'], sample['noise'], sample['processed']
#
#             signal = self.mean + (signal - signal.mean()) * (self.std / signal.std())
#             noise = self.mean + (noise - noise.mean()) * (self.std / noise.std())
#             processed = self.mean + (processed - processed.mean()) * (self.std / processed.std())
#
#             return {'signal': signal,
#                     'noise': noise,
#                     'processed': processed}
