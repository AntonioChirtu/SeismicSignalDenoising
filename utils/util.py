import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
from scipy import signal
import math


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


# CHANGE DIMMENSIONS TO 31X201 (31 FREQUENCY, 201 TIME SAMPLES)
def prepare_dataset(noise, data):
    """ Function for adding noise over signal to provide data for training """

    semnal_e = data

    T = 30  # secunde
    Fs = 100  # frecventa esantionare
    Ts = 1 / Fs
    freqs = [0, Fs / 2]  # Hz
    times = [0, T]  # secunde
    nperseg = 30
    nfft = 60
    nt = int(T / Ts)  # numar esantioane = 30 / (1/100) = 3000 esantioane
    if len(semnal_e) == 3000:
        x = semnal_e
    else:
        x = semnal_e[
            nt:2 * nt]  # selectam 3000 esantioane pentru a avea 201 ferestre temporale suprapuse in care se calculeaza STFT
    # numar_ferestre_temporale ~ (size_x - nperseg)/(nperseg-overlap), overlap = nperseg // 2

    # SNR[dB] = 10 * log_10 (P_signal / P_noise) = 10 * log_10 (P_signal / P_noise), std_noise = sqrt(P_noise)
    # SNR[dB] = 20 * log_10 (std_signal / std_noise) --> e gresita formula (5) din articol
    snr = 10
    std_signal = np.std(x)
    std_noise = std_signal / (10 ** (snr / 20))

    # daca noise are aceeasi lungime
    if len(noise) == len(data):
        noise = std_noise * np.random.randn(x.shape[0])  # noise ~ N(medie = 0, dispersie = sigma_noise)

    # daca noise are lungime mai mica, se insereaza zgomotul la o pozitie aleatoare din semnal
    else:
        noise_length = len(noise)
        position = np.random.randint(0, x.shape[0] - noise_length)
        noise = np.zeros(x.shape[0])
        noise[position:position + noise_length] = np.random.randn(noise_length)
        noise = noise / np.std(noise)
        noise = std_noise * noise

    # Adaugare zgomot
    noisy_x = x + noise
    snr_calculat = 20 * np.log10(np.std(x) / np.std(noise))

    # STFT -- partea reala; partea imaginara
    f, t, x_fft = signal.stft(x, fs=Fs, nperseg=nperseg, nfft=nfft, boundary='zeros')  # signal_shape = [31, 201]
    f, t, noise_fft = signal.stft(noise, fs=Fs, nperseg=nperseg, nfft=nfft, boundary='zeros')
    f, t, noisy_fft = signal.stft(noisy_x, fs=Fs, nperseg=nperseg, nfft=nfft,
                                  boundary='zeros')  # noise_shape = signal_shape

    x_fft = x_fft / np.std(x_fft)
    noisy_fft = noisy_fft / np.std(noisy_fft)

    return {'f': f, 't': t, 'Zxx_processed': noisy_fft, 'Zxx_signal': x_fft, 'Zxx_noise': noise_fft}, noisy_x, noise


class ToTensor(object):
    """Converts ndarrays in sample to Tensors"""

    def __call__(self, sample):
        if len(sample) == 5:
            stft_dict = sample

            stft_dict['Zxx_signal'] = torch.from_numpy(stft_dict['Zxx_signal'])
            stft_dict['Zxx_processed'] = torch.from_numpy(stft_dict['Zxx_processed'])
            return {'f': sample['f'], 't': sample['t'], 'Zxx_signal': stft_dict['Zxx_signal'],
                    'Zxx_processed': stft_dict['Zxx_processed']}
        else:
            signal, noise, processed = sample['signal'], sample['noise'], sample['processed']
            return {'signal': torch.from_numpy(signal),
                    'noise': torch.from_numpy(noise),
                    'processed': torch.from_numpy(processed)}


class Rescale(object):
    """Rescales Tensor values to given min and max values"""

    def __call__(self, sample):
        if len(sample) == 4:
            stft_dict = sample

            stft_dict['Zxx_signal'] = stft_dict['Zxx_signal'].numpy()
            stft_dict['Zxx_processed'] = stft_dict['Zxx_processed'].numpy()

            min_value_proc = np.amin(stft_dict['Zxx_processed'])
            lower_value_proc = np.amax(stft_dict['Zxx_processed']) - min_value_proc

            min_value_sig = np.amin(stft_dict['Zxx_signal'])
            lower_value_sig = np.amax(stft_dict['Zxx_signal']) - min_value_sig

            stft_dict['Zxx_processed'] = np.array(
                [(x - min_value_proc) / lower_value_proc for x in stft_dict['Zxx_processed']])
            stft_dict['Zxx_signal'] = np.array([(x - min_value_sig) / lower_value_sig for x in stft_dict['Zxx_signal']])

            return {'f': sample['f'], 't': sample['t'], 'Zxx_signal': torch.from_numpy(stft_dict['Zxx_signal']),
                    'Zxx_processed': torch.from_numpy(stft_dict['Zxx_processed'])}
        else:
            signal, noise, processed = sample['signal'].numpy(), sample['noise'].numpy(), sample['processed'].numpy()

            min_value_proc = np.amin(processed)
            lower_value_proc = np.amax(processed) - min_value_proc

            min_value_sig = np.amin(signal)
            lower_value_sig = np.amax(signal) - min_value_sig

            processed = np.array(
                [(x - min_value_proc) / lower_value_proc for x in processed])
            signal = np.array([(x - min_value_sig) / lower_value_sig for x in signal])

            return {'signal': torch.from_numpy(signal),
                    'noise': torch.from_numpy(noise),
                    'processed': torch.from_numpy(processed)}


class Normalize(object):
    '''Normalizes Tensor values to given mean and standard deviation'''

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        if len(sample) == 4:
            stft_dict = sample

            signal_real_mean = stft_dict['Zxx_signal'].real.mean()
            signal_real_std = stft_dict['Zxx_signal'].real.std()
            signal_imag_mean = stft_dict['Zxx_signal'].imag.mean()
            signal_imag_std = stft_dict['Zxx_signal'].imag.std()
            processed_real_mean = stft_dict['Zxx_processed'].real.mean()
            processed_real_std = stft_dict['Zxx_processed'].real.std()
            processed_imag_mean = stft_dict['Zxx_processed'].imag.mean()
            processed_imag_std = stft_dict['Zxx_processed'].imag.std()

            stft_dict['Zxx_signal'].real = self.mean + (stft_dict['Zxx_signal'].real - signal_real_mean) * (
                        self.std / signal_real_std)
            stft_dict['Zxx_signal'].imag = self.mean + (stft_dict['Zxx_signal'].imag - signal_imag_mean) * (
                    self.std / signal_imag_std)
            stft_dict['Zxx_processed'].real = self.mean + (stft_dict['Zxx_processed'].real - processed_real_mean) * (
                    self.std / processed_real_std)
            stft_dict['Zxx_processed'].imag = self.mean + (stft_dict['Zxx_processed'].imag - processed_imag_mean) * (
                    self.std / processed_imag_std)

            return {'f': sample['f'], 't': sample['t'], 'Zxx_signal': stft_dict['Zxx_signal'],
                    'Zxx_processed': stft_dict['Zxx_processed']}
        else:

            signal, noise, processed = sample['signal'], sample['noise'], sample['processed']

            signal = self.mean + (signal - signal.mean()) * (self.std / signal.std())
            noise = self.mean + (noise - noise.mean()) * (self.std / noise.std())
            processed = self.mean + (processed - processed.mean()) * (self.std / processed.std())

            return {'signal': signal,
                    'noise': noise,
                    'processed': processed}
