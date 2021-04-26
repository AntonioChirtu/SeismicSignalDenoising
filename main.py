import os
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader.data_loaders import SeismicDatasetLoader
from model.model import Net
from model.loss import softCrossEntropy
# from util import ToTensor, Rescale, Normalize
from scipy.signal import istft

TRAIN_DIR = 'train'
PRED_DIR = 'pred'
NOISE_DIR = 'Noise_waveforms'
path = './/data'
save_path = 'denoising_net.pth'

T = 30  # secunde
Fs = 100  # frecventa esantionare
Ts = 1 / Fs
nt = int(T / Ts)  #
nperseg = 30
nfft = 60

train_size = 0.8
test_size = 0.2

# tensor = ToTensor()
# rescale = Rescale()
# normalize = Normalize(0.5, 0.5)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def main():
    train_dataset = SeismicDatasetLoader(root_dir=path, signal_dir=TRAIN_DIR, noise_dir=NOISE_DIR,
                                         type='train', transform=transform)
    test_dataset = SeismicDatasetLoader(root_dir=path, signal_dir=PRED_DIR, noise_dir=NOISE_DIR,
                                        type='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net().double()
    net.to(device)
    # criterion = nn.CrossEntropyLoss()
    # criterion = softCrossEntropy()

    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

    SNR_before_denoising = []
    SNR_after_denoising = []

    if not os.path.exists(save_path):
        net.train()
        for epoch in range(50):
            print('TRAIN epoch #', epoch)
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                with torch.enable_grad():
                    _, inputs, _, _, targets = data

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

            print('Epoch %d loss: %.3f' % (epoch, running_loss / train_loader.__len__()))

        print('Finished Training')

        torch.save(net.state_dict(), save_path)

    model = Net()
    model.load_state_dict(torch.load(save_path))
    model.eval()

    with torch.no_grad():
        for data in test_loader:
            signal, inputs, noisy_signal_fft, snr, targets = data
            signal = signal.cpu().detach().numpy()
            noisy_signal_fft = noisy_signal_fft.cpu().detach().numpy()
            SNR_before_denoising.append(snr.cpu().detach().numpy())

            inputs = inputs.permute(0, 3, 1, 2).to(device)
            outputs = net(inputs)

            outputs = outputs.cpu().detach().numpy()

            denoised_fft = noisy_signal_fft * outputs[:, 0, :, :]
            _, denoised_signal = istft(denoised_fft, fs=Fs, nperseg=nperseg, nfft=nfft, boundary='zeros')

            # plt.figure()
            # plt.plot(signal.flatten())
            # plt.title('Original signal')
            # plt.figure()
            # plt.plot(denoised_signal.flatten())
            # plt.title('Output Signal')
            # plt.show()
            snr_calculat = 20 * np.log10(np.std(signal) / np.std(denoised_signal - signal))
            SNR_after_denoising.append(snr_calculat)

    plt.figure()
    plt.plot(np.array(SNR_before_denoising), np.array(SNR_after_denoising), '*')
    plt.xlabel("SNR before denoising")
    plt.ylabel("SNR after denoising")
    plt.show()


if __name__ == '__main__':
    main()
