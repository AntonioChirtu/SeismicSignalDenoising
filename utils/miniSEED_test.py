from obspy import read
from matplotlib import pyplot as plt
import numpy as np
import os

'''
st = read('file1')
st1 = read('file2')

for k, v in sorted(st[0].stats.mseed.items()):
    print("'%s': %s" % (k, str(v))) 

plt.plot(st[0].data)

np.savez('file1.npz', st[0].data)

np.savez('file2.npz', st1[0].data)
'''

path = r'/home/ant/Python_projects/grad_thesis/SeismicSignalDenoising/data/miniSEED'
noise_path = r'/home/ant/Python_projects/grad_thesis/SeismicSignalDenoising/data/Noise_waveforms'
signal_path = r'/home/ant/Python_projects/grad_thesis/SeismicSignalDenoising/data/Signal_waveforms'

for count, filename in enumerate(os.listdir(path)):
    src = filename 
    dst = 'file' + str(count)
    # rename() function will 
    # rename all the files 
    #if (src[:4] != 'file'):
        #os.rename(path + '/' + src, path + '/' + dst)
    print(filename)
    st = read(path + '/' + filename)
    np.savez(noise_path + '/' + "noise_" + filename + '.npz', st[0].data)
    os.remove(path + '/' + filename)



#data = np.asarray(st1[0].data)
#np.savetxt('out1.csv', data, delimiter=',')