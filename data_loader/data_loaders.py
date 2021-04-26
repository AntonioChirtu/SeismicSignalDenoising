import random

from torch.utils.data import Dataset
import numpy as np
import torch
import os
from random import randint, choice
from utils.util import prepare_dataset
import matplotlib.pyplot as plt

NOISE_SAMPLES = 100


class SeismicDatasetLoader(Dataset):
    def __init__(self, root_dir, signal_dir, noise_dir, type, transform=None):
        self.root_dir = root_dir
        self.signal_dir = signal_dir
        self.noise_dir = noise_dir
        self.type = type
        self.transform = transform

        assert os.path.exists(os.path.join(self.root_dir, self.signal_dir)), 'Path to signal images cannot be found'
        assert os.path.exists(os.path.join(self.root_dir, self.noise_dir)), 'Path to noise images cannot be found'

        self.signal_files = sorted([os.path.join(self.root_dir, signal_dir, file) for file in
                                    os.listdir(os.path.join(self.root_dir, self.signal_dir))
                                    if file.endswith('.npz')])
        self.noise_files = sorted([os.path.join(self.root_dir, noise_dir, file) for file in
                                   os.listdir(os.path.join(self.root_dir, self.noise_dir))
                                   if file.endswith('.npz')])

    def __len__(self):
        if self.type == 'train':
            l = 500
        if self.type == 'test':
            l = len(self.signal_files)
        return l

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        item = np.mod(item, len(self.signal_files))

        signal_dict = np.load(self.signal_files[item], allow_pickle=True)['data']
        if len(signal_dict.shape) > 1:
            signal = signal_dict[:, 0]
        else:
            signal = signal_dict

        noise = np.load(self.noise_files[randint(0, len(self.noise_files) - 1)], allow_pickle=True)['arr_0']
        while np.sum(noise) == 0:
            noise = np.load(self.noise_files[randint(0, len(self.noise_files) - 1)], allow_pickle=True)['arr_0']

        A_noise = random.randint(1, 4)
        snr = random.randint(1, 12)
        if self.type == 'train':
            signal, noise, noisy_signal_fft, signal_fft, noise_fft = prepare_dataset(signal, noise, A_noise, snr,
                                                                                     itp=3000)
        if self.type == 'test':
            signal, noise, noisy_signal_fft, signal_fft, noise_fft = prepare_dataset(signal, noise, A_noise, snr, itp=0)

        # Masks
        r = np.abs(noise_fft) / (np.abs(signal_fft) + 1e-5)
        targets = np.zeros(shape=(r.shape[0], r.shape[1], 2))
        targets[:, :, 0] = 1 / (1 + r)  # Ms = signal mask
        targets[:, :, 1] = r / (1 + r)  # Mn = noise mask

        inputs = np.zeros(shape=(noisy_signal_fft.shape[0], noisy_signal_fft.shape[1], 2))
        inputs[:, :, 0] = self.transform(noisy_signal_fft.real)
        inputs[:, :, 1] = self.transform(noisy_signal_fft.imag)

        return torch.from_numpy(signal), inputs, torch.from_numpy(noisy_signal_fft), torch.from_numpy(np.array(snr)), \
               torch.from_numpy(targets)
