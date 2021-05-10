import random

from torch.utils.data import Dataset
import numpy as np
import torch
import os
from random import randint, choice
from utils.util import prepare_dataset
import pandas as pd
import h5py

signal_hdf = "chunk2.hdf5"
noise_hdf = "chunk1.hdf5"


class SeismicDatasetLoader(Dataset):
    def __init__(self, root_dir, signal_dir, noise_dir, type, transform=None):
        self.root_dir = root_dir
        self.signal_dir = signal_dir
        self.noise_dir = noise_dir
        self.type = type
        self.transform = transform

        assert os.path.exists(os.path.join(self.root_dir, self.signal_dir)), 'Path to signal images cannot be found'
        assert os.path.exists(os.path.join(self.root_dir, self.noise_dir)), 'Path to noise images cannot be found'

        self.signal_files = []
        self.noise_files = []

        noise_df = pd.read_csv(os.path.join(self.root_dir, self.noise_dir, 'chunk1.csv'))
        noise_list = noise_df['trace_name'].to_list()
        noise_file = h5py.File(os.path.join(self.root_dir, self.noise_dir, noise_hdf), 'r')
        for _, evi in enumerate(noise_list):
            noise_dataset = noise_file.get('data/' + str(evi))
            noise_dataset = np.array(noise_dataset)
            self.noise_files.append(noise_dataset)

        if self.type == 'train':
            signal_df = pd.read_csv(os.path.join(self.root_dir, self.signal_dir, 'chunk2.csv'))
            signal_list = signal_df['trace_name'].to_list()
            signal_file = h5py.File(os.path.join(self.root_dir, self.signal_dir, signal_hdf), 'r')
            for _, evi in enumerate(signal_list):
                signal_dataset = signal_file.get('data/' + str(evi))
                signal_dataset = np.array(signal_dataset)
                self.signal_files.append(signal_dataset)

        if self.type == 'test':
            self.signal_files = sorted([os.path.join(self.root_dir, signal_dir, file) for file in
                                        os.listdir(os.path.join(self.root_dir, self.signal_dir))
                                        if file.endswith('.npz')])

        self.signal_files = np.array(self.signal_files)
        self.noise_files = np.array(self.noise_files)

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

        noise = self.noise_files[item]
        noise = noise[:, 0]

        if self.type == 'test':
            signal_dict = np.load(self.signal_files[item], allow_pickle=True)['data']
            if len(signal_dict.shape) > 1:
                signal = signal_dict[:, 0]
            else:
                signal = signal_dict

        else:
            signal = self.signal_files[item]
            signal = signal[:, 0]
        snr = random.randint(1, 12)
        if self.type == 'train':
            signal, noise, noisy_signal_fft, signal_fft, noise_fft = prepare_dataset(signal, noise, snr,
                                                                                     itp=3000)
        if self.type == 'test':
            signal, noise, noisy_signal_fft, signal_fft, noise_fft = prepare_dataset(signal, noise, snr, itp=0)

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
