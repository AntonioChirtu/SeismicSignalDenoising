import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft


def viz(x, Tx, Wx):
    plt.figure()
    plt.imshow(np.abs(Wx), aspect='auto', cmap='jet')
    plt.figure()
    plt.imshow(np.abs(Tx), aspect='auto', vmin=0, vmax=.2, cmap='jet')
    plt.show()


# %%# Define signal ####################################
N = 3000
t = np.linspace(0, 1, N, endpoint=False)
xo = np.random.randint(10) * np.cos(2 * np.pi * 2 * (np.exp(t / 2.2) - 1))
xo += xo[::-1]  # add self reflected
x = xo + np.sqrt(2) * np.random.randn(N)  # add noise

# plt.plot(xo);
# plt.show()
# plt.plot(x);
# plt.show()

# %%# CWT + SSQ CWT ####################################
print(type(xo))
print(xo.shape)
print(xo)
Twxo, Wxo, *_ = ssq_cwt(xo, fs=100, nv=64)
viz(xo, Twxo, Wxo)

Twx, Wx, *_ = ssq_cwt(x)
viz(x, Twx, Wx)

# %%# STFT + SSQ STFT ##################################
Tsxo, Sxo, *_ = ssq_stft(xo)
viz(xo, np.flipud(Tsxo), np.flipud(Sxo))

Tsx, Sx, *_ = ssq_stft(x)
viz(x, np.flipud(Tsx), np.flipud(Sx))
