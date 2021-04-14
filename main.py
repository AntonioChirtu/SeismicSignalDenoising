import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader.data_loaders import SeismicDatasetLoader
from model.model import Net
from utils.util import ToTensor, Rescale, Normalize
from scipy.signal import istft, resample

TRAIN_DIR = 'train'
PRED_DIR = 'pred'
NOISE_DIR = 'Noise_waveforms'
path = r'/home/antonio/SeismicSignalDenoising/data'
save_path = './denoising_net.pth'

T = 30  # secunde
Fs = 100  # frecventa esantionare
Ts = 1 / Fs
nt = int(T / Ts)  #
nperseg = 30
nfft = 60

train_size = 0.8
test_size = 0.2

tensor = ToTensor()
rescale = Rescale()
normalize = Normalize(0.5, 0.5)

transform = transforms.Compose([
    tensor,
    normalize
    # rescale
    # transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5])
])


def main():
    train_dataset = SeismicDatasetLoader(root_dir=path, signal_dir=TRAIN_DIR, noise_dir=NOISE_DIR, transform=transform)
    test_dataset = SeismicDatasetLoader(root_dir=path, signal_dir=PRED_DIR, noise_dir=NOISE_DIR, transform=transform)

    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(round(train_size * dataset_size)),
    #                                                                     int(round(test_size * dataset_size))])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    cnt = 0
    amp = 2 * np.sqrt(2)
    # for i in range(len(dataset)):
    #   if (i != 45):
    #      sample, _, _, _ = dataset[i]
    # print('Sample', i, ':')
    # if len(sample['signal']) == 15000:
    #    cnt += 1
    # plt.plot(sample['signal'])
    # plt.show()
    # print(cnt)
    sample, stft_dict_tmp, mask, _, _ = train_dataset[8]
    # plt.hist(np.array(stft_dict_tmp['Zxx_processed']).ravel(), bins=50, density=True);
    # plt.xlabel("pixel values")
    # plt.ylabel("relative frequency")
    # plt.title("distribution of pixels");
    # print(stft_dict_tmp['Zxx_processed'].shape)
    # print(mask.shape)
    # plt.show()
    # plt.figure()
    # plt.plot(sample['signal'], 'r')
    # plt.figure()
    # plt.plot(sample['noise'], 'b')
    # plt.show()
    # plt.figure()
    # plt.plot(sample['processed'], 'm', sample['noise'], 'b')
    # plt.figure()
    # plt.plot(sample['processed'], 'm', sample['signal'], 'r')
    # plt.figure()
    # plt.plot(sample['signal'], 'r')
    '''
    plt.figure()
    plt.pcolormesh(stft_dict_tmp['t'], stft_dict_tmp['f'], stft_dict_tmp['Zxx_processed'][0], vmin=0, vmax=amp,
                   shading='gouraud')
    plt.title('STFT Real')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.figure()
    plt.pcolormesh(stft_dict_tmp['t'], stft_dict_tmp['f'], stft_dict_tmp['Zxx_signal'][0], vmin=0, vmax=amp,
                   shading='gouraud')
    plt.title('STFT Real')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.figure()
    plt.pcolormesh(stft_dict_tmp['t'], stft_dict_tmp['f'], stft_dict_tmp['Zxx_processed'][1])
    plt.title('STFT Imaginary')
    plt.ylabel('Angle [degrees]')
    plt.xlabel('Time [sec]')

    plt.figure()
    plt.pcolormesh(stft_dict_tmp['t'], stft_dict_tmp['f'], stft_dict_tmp['Zxx_signal'][1])
    plt.title('STFT Imaginary')
    plt.ylabel('Angle [degrees]')
    plt.xlabel('Time [sec]')
    # plt.show()
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net().double()
    net.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    MSE = []
    RMS_sig = 0
    RMS_noise = 0
    SNR_orig = []
    SNR = []

    for epoch in range(20):
        error_list = []
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            with torch.enable_grad():

                sample, stft_dict, signal_mask, noise_mask, _ = data

                signal_mask = signal_mask.to(device)
                noise_mask = noise_mask.to(device)
                sample = sample['signal'].to(device)
                inputs = stft_dict['Zxx_processed'].to(device)

                real_inputs = inputs.real
                imag_inputs = inputs.imag
                composed_inputs = torch.stack((real_inputs, imag_inputs), 1)
                composed_inputs = composed_inputs.to(device)

                labels = torch.stack([signal_mask, noise_mask])
                labels = signal_mask.view(signal_mask.size(0), -1)
                labels = labels.squeeze(0)
                labels = labels.long()

                optimizer.zero_grad()

                outputs = net(composed_inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 20 == 19:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
                    running_loss = 0.0
    print('Finished Training')

    # plt.figure()
    # plt.plot(MSE, 'x')
    # plt.figure()
    # plt.plot(SNR_orig, SNR, 'x')
    # plt.show()

    torch.save(net.state_dict(), save_path)

    correct = 0
    total = 0
    MSE = []
    with torch.no_grad():
        for data in test_loader:
            sample, images, signal_labels, noise_labels, noise = data
            noise_labels = noise_labels.to(device)
            signal_labels = signal_labels.to(device)
            sample = sample['signal'].to(device)
            images = images['Zxx_processed'].to(device)

            composed_images = torch.stack((images.real, images.imag), 1)
            composed_images = composed_images.to(device)

            sample = sample.squeeze(0)
            sample = sample.cpu().detach().numpy()

            # plt.figure()
            # plt.plot(sample)
            # plt.title('Original signal')

            signal_labels = signal_labels.view(signal_labels.size(0), -1)
            signal_labels = signal_labels.squeeze(0)

            outputs = net(composed_images)
            outputs = outputs.view(outputs.size(0), outputs.size(1), 31, -1)

            signal_approx = composed_images * outputs

            new_signal_approx = signal_approx[:, 0, :, :] + 1j * signal_approx[:, 1, :, :]

            _, new_signal_approx = istft(new_signal_approx.cpu().detach().numpy(), fs=Fs, nperseg=nperseg, nfft=nfft,
                                         boundary='zeros')

            new_signal_approx = new_signal_approx.squeeze(0)
            rescaled_signal = 0.5 + (new_signal_approx - new_signal_approx.mean()) * (0.5 / new_signal_approx.std())

            # plt.figure()
            # plt.plot(rescaled_signal)
            # plt.title('Output Signal')
            # plt.show()

            snr_calculat = 20 * np.log10(np.std(rescaled_signal) / np.std(sample - rescaled_signal))
            SNR.append(snr_calculat)
            SNR_orig.append(10)

    plt.figure()
    # plt.plot(MSE, 'x')
    plt.plot(SNR, '*')
    plt.plot(SNR_orig, '*')
    plt.show()


if __name__ == '__main__':
    main()
