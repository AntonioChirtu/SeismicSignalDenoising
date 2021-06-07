import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy.signal import istft
import pycwt
from stockwell import ist
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader.data_loaders import SeismicDatasetLoader
from model.loss import softCrossEntropy
from model.model import Net
from utils.util import UnNormalize

TRAIN_DIR = 'chunk2'
PRED_DIR = 'chunk2'
NOISE_DIR = 'chunk1'
path = './/data'
save_path = 'denoising_net.pth'

T = 30  # secunde
Fs = 100  # frecventa esantionare
Ts = 1 / Fs
nt = int(T / Ts)
nperseg = 30
nfft = 60
dj = 1/12
dt = Ts
mother = pycwt.Morlet(6)


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main():
    if not os.path.exists(save_path):
        train_dataset = SeismicDatasetLoader(root_dir=path, signal_dir=TRAIN_DIR, noise_dir=NOISE_DIR,
                                             type='train', transform=None)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    test_dataset = SeismicDatasetLoader(root_dir=path, signal_dir=PRED_DIR, noise_dir=NOISE_DIR,
                                        type='test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net().double()
    net.apply(init_weights)
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

    SNR_before_denoising = []
    SNR_after_denoising = []
    loss_per_epoch = []

    if not os.path.exists(save_path):
        net.train()
        for epoch in range(50):
            print('TRAIN epoch #', epoch)
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                with torch.enable_grad():
                    _, inputs, _, _, targets, *_ = data

                    inputs = inputs.permute(0, 3, 1, 2).to(device)
                    targets = targets.permute(0, 3, 1, 2).to(device)

                    optimizer.zero_grad()

                    outputs = net(inputs)

                    outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
                    targets = targets.view(targets.size(0), targets.size(1), -1)
                    loss = softCrossEntropy(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * data[0].size(0)
                    loss_per_epoch.append(running_loss)

            print('Epoch %d loss: %.3f' % (epoch, running_loss / train_loader.__len__()))

        print('Finished Training')

        torch.save(net.state_dict(), save_path)

    npy_path = r'/home/antonio/SeismicSignalDenoising'
    np.save(npy_path + '/loss.npy', loss_per_epoch)

    model = Net()
    model.load_state_dict(torch.load(save_path))
    model.eval()
    cnt = 0

    with torch.no_grad():
        for data in test_loader:
            signal, inputs, noisy_signal_transform, snr, _, transform_type, signal_transform, noisy_signal, scales = data

            signal = signal.cpu().detach().numpy()
            signal_transform = signal_transform.cpu().detach().numpy()
            noisy_signal = noisy_signal.cpu().detach().numpy()
            noisy_signal_transform = noisy_signal_transform.cpu().detach().numpy()
            signal_max = signal_max.cpu().detach().numpy()
            signal_min = signal_min.cpu().detach().numpy()
            scales = scales.cpu().detach().numpy()

            SNR_before_denoising.append(snr.cpu().detach().numpy())

            inputs = inputs.permute(0, 3, 1, 2).to(device)
            outputs = net(inputs)

            outputs = outputs.cpu().detach().numpy()

            if transform_type[0] == 'STFT':
                denoised_transform = noisy_signal_transform[0, :, :] * outputs[0, 0, :, :]
                _, denoised_signal = istft(denoised_transform, fs=Fs, nperseg=nperseg, nfft=nfft, boundary='zeros')
            elif transform_type[0] == 'S':
                outputs = cv2.resize(outputs[0, 0, :, :], (3000, 1501), interpolation=cv2.INTER_CUBIC)
                denoised_transform = noisy_signal_transform[0, :, :] * outputs
                denoised_signal = ist(denoised_transform)
            elif transform_type[0] == 'CWT':
                outputs = cv2.resize(outputs[0, 0, :, :], (3000, 128), interpolation=cv2.INTER_CUBIC)
                denoised_transform = noisy_signal_transform[0, :, :] * outputs
                sj = scales
                a, b = noisy_signal_transform[0, :, :].shape
                c = sj.size
                if a == c:
                    sj = (np.ones([b, 1]) * sj).transpose()
                elif b == c:
                    sj = np.ones([a, 1]) * sj
                else:
                    raise Warning('Input array dimensions do not match.')
                denoised_signal = (dj * np.sqrt(dt) / (mother.cdelta * mother.psi(0)) * (np.real(denoised_transform) / np.sqrt(sj)).sum(axis=0))

            denoised_signal = 10 * (2 * (denoised_signal - np.amin(denoised_signal)) / (np.amax(denoised_signal) - np.amin(denoised_signal)) - 1)

            plt.figure()
            plt.plot(denoised_signal.flatten())
            plt.title('Output Signal')
            # plt.figure()
            # plt.imshow(denoised_transform[0, :, :].real)
            # plt.title('Output Signal Transform Magnitude')
            # plt.figure()
            # plt.imshow(denoised_transform[0, :, :].imag)
            # plt.title('Output Signal Transform Phase')
            plt.figure()
            plt.plot(signal.flatten())
            plt.title('Original Signal')
            # plt.figure()
            # plt.imshow(signal_transform[0, :, :].real)
            # plt.title('Original Signal Transform Magnitude')
            # plt.figure()
            # plt.imshow(signal_transform[0, :, :].imag)
            # plt.title('Original Signal Transform Phase')
            # plt.figure()
            # plt.plot(noisy_signal.flatten())
            # plt.title('Noisy Signal')
            # plt.figure()
            # plt.imshow(noisy_signal_transform[0, :, :].real)
            # plt.title('Noisy Signal Transform Magnitude')
            # plt.figure()
            # plt.imshow(noisy_signal_transform[0, :, :].imag)
            # plt.title('Noisy Signal Transform Phase')
            plt.show()
            while cnt < 3:
                np.save(npy_path + '/npy_files/' + transform_type[0] + '/signal_' + str(cnt) + '.npy', signal)
                np.save(npy_path + '/npy_files/' + transform_type[0] + '/signal_transform_' + str(cnt) + '.npy',
                        signal_transform)
                np.save(npy_path + '/npy_files/' + transform_type[0] + '/noisy_signal_' + str(cnt) + '.npy',
                        noisy_signal)
                np.save(npy_path + '/npy_files/' + transform_type[0] + '/noisy_signal_transform_' + str(cnt) + '.npy',
                        noisy_signal_transform)
                np.save(npy_path + '/npy_files/' + transform_type[0] + '/denoised_signal_' + str(cnt) + '.npy',
                        denoised_signal)
                np.save(npy_path + '/npy_files/' + transform_type[0] + '/denoised_transform_' + str(cnt) + '.npy',
                        denoised_transform)
                cnt += 1

            snr_calculat = 20 * np.log10(np.std(signal) / np.std(denoised_signal - signal))
            SNR_after_denoising.append(snr_calculat)

    plt.figure()
    plt.plot(np.array(SNR_before_denoising), np.array(SNR_after_denoising), '*')
    plt.xlabel("SNR before denoising")
    plt.ylabel("SNR after denoising")
    plt.show()

    np.save(npy_path + '/SNR_after.npy', SNR_after_denoising)
    np.save(npy_path + '/SNR_before.npy', SNR_before_denoising)


if __name__ == '__main__':
    main()
