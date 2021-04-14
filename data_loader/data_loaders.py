# from torch.utils.data import Dataset
from base.base_data_loader import BaseDataLoader
import numpy as np
import torch
import os
from random import randint
from utils.util import prepare_dataset
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

        # data = [np.load(f, mmap_mode='r')) for f in os.listdir(signal_dir)]

        # lengths =[d.shape[0] for d in data]
        self.signal = sorted([os.path.join(self.root_dir, signal_dir, file) for file in
                              os.listdir(os.path.join(self.root_dir, self.signal_dir))
                              if file.endswith('.npz')])  # and np.isin(int(file[0:4]), self.idx_list)])
        self.noise = sorted([os.path.join(self.root_dir, noise_dir, file) for file in
                             os.listdir(os.path.join(self.root_dir, self.noise_dir))
                             if file.endswith('.npz')])  # and np.isin(int(file[0:4]), self.idx_list)])

    def __len__(self):
        # return len(self.signal)
        return 1000

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        # np_Array = np.load(self.signal[item])
        # print(np_Array['data'])

        noise = np.load(self.noise[randint(0, 99)], allow_pickle=True)['arr_0']
        item = np.random.randint(0, 99)
        signal_dict = np.load(self.signal[item], allow_pickle=True)['data']
        # print(signal_dict.shape)
        if len(signal_dict.shape) > 1:
            signal = signal_dict[:, 0]
        else:
            signal = signal_dict

        stft_dict, processed, noise = prepare_dataset(noise, signal)
        # print(stft_dict['Zxx_processed'].shape)
        sample = {'signal': signal, 'noise': noise, 'processed': processed}

        # Ms
        signal_mask = 1 / (
                1 + np.abs(np.sqrt(stft_dict['Zxx_noise'].real ** 2 + stft_dict['Zxx_noise'].imag ** 2)) / np.abs(
            np.sqrt(stft_dict['Zxx_signal'].real ** 2 + stft_dict['Zxx_signal'].imag ** 2)))

        # Mn
        noise_mask = (np.abs(np.sqrt(stft_dict['Zxx_noise'].real ** 2 + stft_dict['Zxx_noise'].imag ** 2)) / np.abs(
            np.sqrt(stft_dict['Zxx_signal'].real ** 2 + stft_dict['Zxx_signal'].imag ** 2))) / (
                             1 + np.abs(
                         np.sqrt(stft_dict['Zxx_noise'].real ** 2 + stft_dict['Zxx_noise'].imag ** 2)) / np.abs(
                         np.sqrt(stft_dict['Zxx_signal'].real ** 2 + stft_dict['Zxx_signal'].imag ** 2)))

        if self.transform:
            sample = self.transform(sample)
            stft_dict = self.transform(stft_dict)

        return sample, stft_dict, signal_mask, noise_mask, noise
