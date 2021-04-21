import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader.data_loaders import SeismicDatasetLoader
from model.model import Net
from model.loss import softCrossEntropy
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
    # normalize
    rescale
])


def main():
    train_dataset = SeismicDatasetLoader(root_dir=path, signal_dir=TRAIN_DIR, noise_dir=NOISE_DIR, snr=10, type='train',
                                         transform=transform)
    test_dataset = SeismicDatasetLoader(root_dir=path, signal_dir=PRED_DIR, noise_dir=NOISE_DIR, snr=10, type='test',
                                        transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    sample, stft_dict_tmp, mask, _, _, _ = train_dataset[8]
    # plt.figure()
    # plt.plot(sample['processed'])
    # plt.figure()
    # plt.plot(sample['signal'])
    # plt.show()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net().double()
    net.to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = softCrossEntropy()

    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    MSE = []
    SNR_orig = []
    SNR = []

    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            train_dataset.snr = np.random.randint(0, 13)
            with torch.enable_grad():

                sample, stft_dict, signal_mask, noise_mask, _, _ = data

                signal_mask = signal_mask.to(device)
                noise_mask = noise_mask.to(device)
                sample = sample['signal']
                inputs = stft_dict['Zxx_processed']

                composed_inputs = torch.stack((inputs.real, inputs.imag), 1)
                composed_inputs = composed_inputs.to(device)

                labels = torch.stack([signal_mask, noise_mask])
                # labels = signal_mask.view(signal_mask.size(0), -1)
                labels = labels.view(labels.size(1), labels.size(0), -1)
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

    plt.figure()
    plt.plot(MSE, 'x')
    plt.figure()
    plt.plot(SNR_orig, SNR, 'x')
    plt.show()

    torch.save(net.state_dict(), save_path)

    model = Net()
    model.load_state_dict(torch.load(save_path))
    model.eval()

    SNR_mean = []
    with torch.no_grad():
        for i in range(13):
            test_dataset.snr = i
            for data in test_loader:
                sample, images, signal_labels, noise_labels, noise, noisy_snr = data
                noise_labels = noise_labels
                signal_labels = signal_labels
                sample = sample['signal']
                images = images['Zxx_processed']

                composed_images = torch.stack((images.real, images.imag), 1)
                composed_images = composed_images.to(device)

                sample = sample.squeeze(0)
                sample = sample.numpy()

                signal_labels = signal_labels.view(signal_labels.size(0), -1)
                signal_labels = signal_labels.squeeze(0)

                outputs = net(composed_images)
                outputs = outputs.view(outputs.size(0), outputs.size(1), 31, -1)

                signal_approx = composed_images * outputs[:, 0, :, :]

                new_signal_approx = signal_approx[:, 0, :, :] + 1j * signal_approx[:, 1, :, :]

                _, new_signal_approx = istft(new_signal_approx.cpu().detach().numpy(), fs=Fs, nperseg=nperseg,
                                             nfft=nfft,
                                             boundary='zeros')

                new_signal_approx = new_signal_approx.squeeze(0)
                rescaled_signal = 0.5 + (new_signal_approx - new_signal_approx.mean()) * (0.5 / new_signal_approx.std())

                plt.figure()
                plt.plot(sample)
                plt.title('Original signal')
                plt.figure()
                plt.plot(rescaled_signal)
                plt.title('Output Signal')
                plt.show()

                snr_calculat = 20 * np.log10(np.std(rescaled_signal) / np.std(sample - rescaled_signal))
                SNR.append(snr_calculat)
            print(i)
            SNR_mean.append(sum(SNR) / len(SNR))
            SNR_orig.append(i)

    plt.figure()
    plt.plot((SNR_mean,), '*')
    plt.legend("Blue - after denoising")
    plt.plot((SNR_orig,), '*')
    plt.legend("Orange - original")
    plt.show()


if __name__ == '__main__':
    main()
