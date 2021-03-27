import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader.data_loaders import SeismicDatasetLoader
from model.model import Net
from utils.util import ToTensor, Normalize
from scipy.signal import istft, resample

SIGNAL_DIR = 'Signal_waveforms'
NOISE_DIR = 'Noise_waveforms'
path = r'/home/antonio/SeismicSignalDenoising/SeismicSignalDenoising/data'
save_path = './denoising_net.pth'

train_size = 0.8
test_size = 0.2

tensor = ToTensor()
normalize = Normalize(0, 1)


# transform = transforms.Compose([
# tensor,
# normalize
#     #transforms.ToTensor(),
#    #transforms.Normalize([0], [1])
# ])

def main():
    dataset = SeismicDatasetLoader(root_dir=path, signal_dir=SIGNAL_DIR, noise_dir=NOISE_DIR, transform=None)
    dataset_size = len(dataset)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(round(train_size * dataset_size)),
                                                                          int(round(test_size * dataset_size))])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    sample, stft_dict_tmp, _, _ = dataset[8]
    amp = 2 * np.sqrt(2)

    # plt.figure()
    # plt.plot(sample['signal'], 'r')
    # plt.figure()
    # plt.plot(sample['noise'], 'b')
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
    # net.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    MSE = []
    for epoch in range(50):
        error_list = []
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            with torch.enable_grad():

                sample, stft_dict, noise_mask, signal_mask = data
                sample = sample['signal']
                inputs = stft_dict['Zxx_processed']  # .to(device)

                signal_approx = signal_mask * inputs[:, 0, :, :]
                signal_approx = signal_approx.squeeze(0)
                sample = sample.squeeze(0)

                _, signal_approx = istft(signal_approx)

                signal_approx = resample(signal_approx, 25500)

                sample = sample.detach().numpy()

                error = np.abs(signal_approx - sample) ** 2
                error = error.sum() / len(error)
                error_list.append(error)

                labels = torch.stack([signal_mask, noise_mask])
                labels = signal_mask.view(signal_mask.size(0), -1)
                labels = labels.squeeze(0)
                labels = labels.long()

                optimizer.zero_grad()

                outputs = net(inputs)
                output_plots = outputs[:, 0].detach().numpy()
                output_plots = output_plots.reshape(31, 201)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 20 == 19:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
                    running_loss = 0.0
        MSE.append(sum(error_list) / len(error_list))
    print('Finished Training')

    plt.figure()
    plt.plot(MSE, 'x')
    plt.show()

    torch.save(net.state_dict(), save_path)

    correct = 0
    total = 0
    MSE = []
    with torch.no_grad():
        for data in test_loader:
            sample, images, noise_labels, signal_labels = data
            sample = sample['signal']
            images = images['Zxx_processed']

            signal_approx = signal_labels * images[:, 0, :, :]
            signal_approx = signal_approx.squeeze(0)
            sample = sample.squeeze(0)

            _, signal_approx = istft(signal_approx)

            signal_approx = resample(signal_approx, 25500)

            sample = sample.detach().numpy()
            plt.figure()
            plt.plot(signal_approx)
            plt.figure()
            plt.plot(sample)
            plt.show()
            error = np.abs(signal_approx - sample) ** 2
            MSE.append(error.sum() / len(error))
            # print("MSE for current test image is:", MSE)
            # print(signal_approx[:, 0, 0])

            signal_labels = signal_labels.view(signal_labels.size(0), -1)
            signal_labels = signal_labels.squeeze(0)

            outputs = net(images)

            plot_outputs = outputs[:, 0].detach().numpy()
            plot_outputs = plot_outputs.reshape(31, 201)

            plot_signal_labels = signal_labels.detach().numpy()
            plot_signal_labels = plot_signal_labels.reshape(31, 201)

            # plt.figure()
            # plt.imshow(plot_outputs)
            # plt.title('Network output')
            # plt.figure()
            # plt.imshow(plot_signal_labels)
            # plt.title('Signal_labels')
            # plt.show()

            # _, predicted = torch.max(outputs.data, 1)
            predicted = outputs[:, 0]
            total += signal_labels.size(0)

            correct += (predicted == signal_labels).sum().item()

    # print('Accuracy of the network on the test images: %d %%' % (
    #       100 * correct / total))
    plt.figure()
    plt.plot(MSE, 'x')
    plt.show()


if __name__ == '__main__':
    main()
