# from torch.utils.data import Dataset
from base.base_data_loader import BaseDataLoader
import numpy as np
import torch
import os
from random import randint
from utils.util import prepare_dataset, pol2cart
from scipy.signal import stft, resample
from matplotlib import pyplot as plt

NOISE_SAMPLES = 100


class SeismicDatasetLoader(BaseDataLoader):
    def __init__(self, root_dir, signal_dir, noise_dir, transform=None):
        self.root_dir = root_dir
        self.signal_dir = signal_dir
        self.noise_dir = noise_dir
        self.transform = transform

        assert os.path.exists(os.path.join(self.root_dir, self.signal_dir)), 'Path to signal images cannot be found'
        assert os.path.exists(os.path.join(self.root_dir, self.noise_dir)), 'Path to noise images cannot be found'

        self.signal = sorted([os.path.join(self.root_dir, signal_dir, file) for file in
                              os.listdir(os.path.join(self.root_dir, self.signal_dir))
                              if file.endswith('.npz')])  # and np.isin(int(file[0:4]), self.idx_list)])
        self.noise = sorted([os.path.join(self.root_dir, noise_dir, file) for file in
                             os.listdir(os.path.join(self.root_dir, self.noise_dir))
                             if file.endswith('.npz')])  # and np.isin(int(file[0:4]), self.idx_list)])

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        signal = np.load(self.signal[item], allow_pickle=True)['arr_0']
        noise = np.load(self.noise[randint(0, item)], allow_pickle=True)['arr_0']

        signal = resample(signal, 25500)
        processed = prepare_dataset(noise, signal)
        processed = resample(processed, len(signal))
        noise = resample(noise, len(signal))

        sample = {'signal': signal, 'noise': noise, 'processed': processed}

        f, t, Zxx_processed = stft(processed)
        _, _, Zxx_signal = stft(signal)
        _, _, Zxx_noise = stft(noise)

        Zxx_processed = pol2cart(np.abs(Zxx_processed), np.angle(Zxx_processed))
        Zxx_signal = pol2cart(np.abs(Zxx_signal), np.angle(Zxx_signal))
        Zxx_noise = pol2cart(np.abs(Zxx_noise), np.angle(Zxx_noise))

        Zxx_signal[0] += 1e-10
        Zxx_signal[1] += 1e-10

        Zxx_processed = resample(Zxx_processed, 31, axis=1)
        Zxx_signal = resample(Zxx_signal, 31, axis=1)
        Zxx_noise = resample(Zxx_noise, 31, axis=1)
        f = resample(f, 31)

        # Ms
        signal_mask = 1 / (1 + np.abs(np.sqrt(Zxx_noise[0] ** 2 + Zxx_noise[1] ** 2)) / np.abs(
            np.sqrt(Zxx_signal[0] ** 2 + Zxx_signal[1] ** 2)))

        # Mn
        noise_mask = (np.abs(np.sqrt(Zxx_noise[0] ** 2 + Zxx_noise[1] ** 2)) / np.abs(
            np.sqrt(Zxx_signal[0] ** 2 + Zxx_signal[1] ** 2))) / (
                                 1 + np.abs(np.sqrt(Zxx_noise[0] ** 2 + Zxx_noise[1] ** 2)) / np.abs(
                             np.sqrt(Zxx_signal[0] ** 2 + Zxx_signal[1] ** 2)))

        stft_dict = {'f': f, 't': t, 'Zxx_signal': Zxx_signal, 'Zxx_processed': Zxx_processed}

        if self.transform:
            sample = self.transform(sample)

        return sample, stft_dict, noise_mask, signal_mask
