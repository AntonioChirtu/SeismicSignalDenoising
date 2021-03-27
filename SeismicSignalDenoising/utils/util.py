import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
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


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y])


# CHANGE DIMMENSIONS TO 31X201 (31 FREQUENCY, 201 TIME SAMPLES)
def prepare_dataset(noise, signal):
    ''' Function for adding noise over signal to provide data for training '''

    signal_cp = np.zeros(len(signal))

    if (len(signal) * 0.2 <= len(noise)):
        noise_cp = noise[:int(0.2 * len(signal))]
    else:
        noise_cp = noise

    '''
    #Duplicate noise untill so it's the same length as signal
    #noise_cp = np.zeros(len(signal) * 0.4)
    if (len(signal) > len(noise)):
        for i in range(len(signal) // len(noise)):
            noise_cp[i * len(noise):(i + 1) * len(noise)] = noise
    elif (len(signal) < len(noise)):
        noise_cp = noise[:len(signal)]
    else:
        noise_cp = noise
    '''

    # calculate constant for required RMS
    signal_RMS = math.sqrt(np.mean(signal ** 2))
    required_RMS = math.sqrt(signal_RMS ** 2 / 10 ** 2)
    noise_RMS = math.sqrt(max(np.mean(noise_cp ** 2), 0))
    if (noise_RMS != 0):
        constant = required_RMS / noise_RMS
    else:
        constant = 1

    for idx in range(len(noise_cp)):
        noise_cp[idx] = noise_cp[idx] * constant

    if (len(signal_cp) * 0.2 <= len(noise_cp)):
        signal_cp[int(0.2 * len(signal_cp)):int(0.4 * len(signal_cp))] += noise_cp
        signal_cp[int(0.2 * len(signal_cp)):int(0.4 * len(signal_cp))] += signal[
                                                                          int(0.2 * len(signal)):int(0.4 * len(signal))]
        signal_cp[:int(0.2 * len(signal_cp))] = signal[:int(0.2 * len(signal))]
        signal_cp[int(0.4 * len(signal_cp)) + 1:] = signal[int(0.4 * len(signal)) + 1:]
    else:
        signal_cp[int(0.2 * len(signal_cp)):int(0.2 * len(signal_cp)) + len(noise_cp)] += noise_cp
        signal_cp[int(0.2 * len(signal_cp)):int(0.2 * len(signal_cp)) + len(noise_cp)] += signal[
                                                                                          int(0.2 * len(signal)):int(
                                                                                              0.2 * len(signal)) + len(
                                                                                              noise_cp)]
        signal_cp[:int(0.2 * len(signal_cp))] = signal[:int(0.2 * len(signal))]
        signal_cp[int(0.2 * len(signal_cp)) + len(noise_cp) + 1:] = signal[int(0.2 * len(signal)) + len(noise_cp) + 1:]

    return signal_cp


class ToTensor(object):
    '''Converts ndarrays in sample to Tensors'''

    def __call__(self, sample):
        signal, noise, processed = sample['signal'], sample['noise'], sample['processed']

        return {'signal': torch.from_numpy(signal, requires_grad=True),
                'noise': torch.from_numpy(noise, required_grad=True),
                'processed': torch.from_numpy(processed, requires_grad=True)}


class Normalize(object):
    '''Normalizes Tensor values to given mean and standard deviation'''

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        signal, noise, processed = sample['signal'], sample['noise'], sample['processed']

        transform = transforms.Normalize(self.mean, self.std)
        return {'signal': transform(signal),
                'noise': transform(noise),
                'processed': transform(processed)}
