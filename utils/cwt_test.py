import numpy as np
import matplotlib.pyplot as plt

import pycwt

if __name__ == '__main__':

    data = np.load('AZ_KNW_2012080816512091.npz', allow_pickle=True)
    a = data['data']
    semnal_e = a[:, 0]


    T = 30 # secunde
    Fs = 100 # frecventa esantionare
    Ts = 1/Fs
    freqs = [0, Fs/2] # Hz
    times = [0, T] # secunde
    nperseg = 30
    nfft = 60
    nt = int(T/Ts) # numar esantioane = 30 / (1/100) = 3000 esantioane
    x = semnal_e[nt:2*nt] # selectam 3000 esantioane pentru a avea 201 ferestre temporale suprapuse in care se calculeaza STFT
    # numar_ferestre_temporale ~ (size_x - nperseg)/(nperseg-overlap), overlap = nperseg // 2


    mother = pycwt.Morlet(6)
    W, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(x, dt=Ts, dj=1/12, s0=2*Ts, wavelet=mother)
    # ix = pycwt.icwt(W, sj=scales, dt=Ts, dj=1/12, wavelet=mother)

    dj = 1/12
    dt = Ts
    sj = scales
    # iW = (dj * np.sqrt(dt) / wavelet.cdelta * wavelet.psi(0) *
    #       (np.real(W) / sj).sum(axis=0)) should be
    # iW = (dj * np.sqrt(dt) / (wavelet.cdelta * wavelet.psi(0)) * (np.real(W) / np.sqrt(sj)).sum(axis=0))
    a, b = W.shape
    c = sj.size
    if a == c:
        sj = (np.ones([b, 1]) * sj).transpose()
    elif b == c:
        sj = np.ones([a, 1]) * sj
    else:
        raise Warning('Input array dimensions do not match.')
    iW = (dj * np.sqrt(dt) / (mother.cdelta * mother.psi(0)) * (np.real(W) / np.sqrt(sj)).sum(axis=0))

    plt.figure()
    plt.plot(x)
    plt.figure()
    plt.imshow(np.abs(W))
    plt.figure()
    plt.plot(iW.real)
    plt.show()