import random

from torch.utils.data import Dataset
import numpy as np
import torch
import os
from utils.util import prepare_dataset
import pandas as pd
import h5py
import cv2

signal_hdf = "chunk2.hdf5"
noise_hdf = "chunk1.hdf5"
transform_type = 2  # 0 - STFT;  1 - SSQ-CWT;  2 - Stockwell transform;  3 - CWT


class SeismicDatasetLoader(Dataset):
    def __init__(self, root_dir, signal_dir, noise_dir, type, transform=None):
        self.root_dir = root_dir
        self.signal_dir = signal_dir
        self.noise_dir = noise_dir
        self.type = type
        self.transform = transform

        assert os.path.exists(os.path.join(self.root_dir, self.signal_dir)), 'Path to signal images cannot be found'
        assert os.path.exists(os.path.join(self.root_dir, self.noise_dir)), 'Path to noise images cannot be found'

        self.signal_files_train = []
        self.signal_files_test = []
        self.noise_files = []

        noise_df = pd.read_csv(os.path.join(self.root_dir, self.noise_dir, 'chunk1.csv'))
        noise_list = noise_df['trace_name'].to_list()
        noise_file = h5py.File(os.path.join(self.root_dir, self.noise_dir, noise_hdf), 'r')
        for _, evi in enumerate(noise_list):
            noise_dataset = noise_file.get('data/' + str(evi))
            noise_dataset = np.array(noise_dataset)
            self.noise_files.append(noise_dataset)

        signal_df = pd.read_csv(os.path.join(self.root_dir, self.signal_dir, 'chunk2.csv'))
        signal_list = signal_df['trace_name'].to_list()
        signal_file = h5py.File(os.path.join(self.root_dir, self.signal_dir, signal_hdf), 'r')
        if self.type == 'train':
            for _, evi in enumerate(signal_list[:160000]):
                signal_dataset_train = signal_file.get('data/' + str(evi))
                signal_dataset_train = np.array(signal_dataset_train)
                self.signal_files_train.append(signal_dataset_train)

        if self.type == 'test':
            for _, evi in enumerate(signal_list[160000:]):
                signal_dataset_test = signal_file.get('data/' + str(evi))
                signal_dataset_test = np.array(signal_dataset_test)
                self.signal_files_test.append(signal_dataset_test)

        self.signal_files_train = np.array(self.signal_files_train)
        self.signal_files_test = np.array(self.signal_files_test)
        self.noise_files = np.array(self.noise_files)
        self.range_choice = list(range(7359)) + list(range(7428, 12219)) + list(range(14673, 18060)) + list(
            range(18061, 19463)) + list(range(19497, 19769)) + list(range(19881, 25284)) + list(
            range(25482, 49550)) + list(range(49670, 68172)) + list(range(68674, 72740)) + list(
            range(73497, 97207)) + list(range(97883, len(self.noise_files)))

    def __len__(self):
        if self.type == 'train':
            l = 25000
        if self.type == 'test':
            l = len(self.signal_files_test)
        return l

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        if self.type == 'train':
            item = np.mod(item, len(self.signal_files_train))
            signal = self.signal_files_train[item]
        else:
            item = np.mod(item, len(self.signal_files_test))
            signal = self.signal_files_test[item]
        signal = signal[:, 0]

        noise = self.noise_files[random.choice(self.range_choice)]
        noise = noise[:, 0]

        snr = random.randint(1, 12)
        if self.type == 'train':
            signal, noise, noisy_signal_transform, signal_transform, noise_transform = prepare_dataset(signal, noise,
                                                                                                       snr,
                                                                                                       itp=0,
                                                                                                       transform_type=transform_type)
        if self.type == 'test':
            signal, noise, noisy_signal_transform, signal_transform, noise_transform = prepare_dataset(signal, noise,
                                                                                                       snr,
                                                                                                       itp=0,
                                                                                                       transform_type=transform_type)

        noise_resized_re = cv2.resize(noise_transform.real, (201, 31), interpolation=cv2.INTER_CUBIC)
        noise_resized_im = cv2.resize(noise_transform.imag, (201, 31), interpolation=cv2.INTER_CUBIC)
        noise_resized = noise_resized_re + 1j * noise_resized_im

        signal_resized_re = cv2.resize(signal_transform.real, (201, 31), interpolation=cv2.INTER_CUBIC)
        signal_resized_im = cv2.resize(signal_transform.imag, (201, 31), interpolation=cv2.INTER_CUBIC)
        signal_resized = signal_resized_re + 1j * signal_resized_im

        # Masks
        r = np.abs(noise_resized) / (np.abs(signal_resized) + 1e-5)  # signal_transform, noise_transform
        targets = np.zeros(shape=(r.shape[0], r.shape[1], 2))
        targets[:, :, 0] = 1 / (1 + r)  # Ms = signal mask
        targets[:, :, 1] = r / (1 + r)  # Mn = noise mask

        transform_resized_re = cv2.resize(noisy_signal_transform.real, (201, 31), interpolation=cv2.INTER_CUBIC)
        transform_resized_im = cv2.resize(noisy_signal_transform.imag, (201, 31), interpolation=cv2.INTER_CUBIC)
        transform_resized = transform_resized_re + 1j * transform_resized_im

        inputs = np.zeros(shape=(transform_resized.shape[0], transform_resized.shape[1], 2))
        inputs[:, :, 0] = self.transform(transform_resized.real)
        inputs[:, :, 1] = self.transform(transform_resized.imag)

        return torch.from_numpy(signal), inputs, torch.from_numpy(noisy_signal_transform), torch.from_numpy(
            np.array(snr)), \
               torch.from_numpy(targets), transform_type
