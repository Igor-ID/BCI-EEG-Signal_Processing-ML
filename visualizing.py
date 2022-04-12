import numpy as np
import os
import matplotlib.pyplot as plt
import pywt
import mne

plt.style.use('seaborn')
sampling_rate = 125      # EEG sample rate

starting_dir = "data/Hades/"  # or "os.path.dirname(os.path.realpath(__file__))/acquisition/data"
archetype_dir = 'Sun'
data = np.load(os.path.join(starting_dir, "1648771418.npy"))
data_reshaped = np.reshape(data, (1600, 16))

eeg_ch = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3', 'F4', 'T7', 'T8', 'P3', 'P4']
sfreq = 125
info = mne.create_info(eeg_ch, sfreq, ch_types='eeg')
info.set_montage('standard_1020')

data_mne = (data_reshaped / 1000000).T
data_mne_V = data_reshaped.T
info['description'] = f'Archetype_{archetype_dir}'
raw = mne.io.RawArray(data_mne, info)
raw_V = mne.io.RawArray(data_mne_V, info)
raw.notch_filter(freqs=60)  # creating band-stop filter to eliminate power line noise (US - 60Hz, EU - 50Hz)
raw_V.notch_filter(freqs=60)
data2 = raw_V.get_data()
raw.filter(l_freq=0.2, h_freq=60)  # creating band-pass filter, to eliminate non-physiological frequencies
raw_V.filter(l_freq=0.2, h_freq=60)
data3 = raw_V.get_data()
# print(raw.info)
# print(raw_V.info)

events = mne.make_fixed_length_events(raw, start=0, stop=12, duration=12)
epochs = mne.Epochs(raw, events, tmin=0, tmax=12, baseline=(0, 12), reject=None)
freqs = np.arange(0.2, 60)
n_cycles = freqs / 2
power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False)
power.plot([0])
power.plot([0], mode='logratio', title=power.ch_names[0])
power.plot_topo(baseline=(0, 12), mode='logratio', title='Average power')
power.plot([8], baseline=(0, 6), mode='logratio', title=power.ch_names[8])
print(power.info)
# power.plot_topo(baseline=(0, 0), mode='logratio', title='Average power')
# power.plot([0], baseline=(0, 0), mode='logratio', title=power.ch_names[0])
# print(epochs)
# print(events)
# fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'])
# fig.subplots_adjust(right=0.7)  # make room for legend

# df = raw_V.to_data_frame()
# print(df.head())
# df = raw_V.to_data_frame()
# print(df.head())
# print(raw.info)
# print('-------')
# print(raw_V.info)
# data1, times1 = raw[:]  # get data (numpy.ndarray) and time (array)
# data2 = raw.get_data()  # get data (numpy.ndarray)
# print(data1.shape)
# print('----')
# print(data2)
# print('----')
# print(data3)
# print('----')
# print(data3.shape)
# print('----')
# print(data3.T.shape)
# print('----')
# print(type(data2))
# raw.plot()
# print(len(raw))
# data, times = raw[:]
# print(times)
# print(len(times))
'''
data_mne_processed = data3.T

t = np.linspace(0, data_mne_processed.shape[0] / sampling_rate, data_mne_processed.shape[0])  # 12.8 seconds long each numpy file
print(t)

ch_C3_res = data_mne_processed[:, 0]
ch_C4_res = data_mne_processed[:, 8]

# fig = plt.figure(figsize=(14, 8))
# ax1 = fig.add_subplot(1, 2, 1)
# plt.plot(t, ch_C3_res, color='red', label='C3')
# ax2 = fig.add_subplot(1, 2, 2)
# plt.plot(t, ch_C4_res, color='blue', label='C4')
# plt.xlabel('time /s')
# plt.ylabel('amplitude')
# plt.legend(loc='upper right', fontsize='large', frameon=True, edgecolor='blue')

# print(data.shape[0])
# print(data_reshaped.shape[0])
# print(len(t))
# print(t.shape)
# print(data.shape[0]/sampleRate)
# print(data_reshaped.shape[0]/sampleRate)
# print(t)

# python wavelet transfrom
# cA, cD = pywt.dwt(ch_C3_res, 'db4')
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# ax1.plot(ch_C3_res)
# ax2.plot(cA, '-g')
# ax3.plot(cD, '-r')

print(plt.style.available)
plt.style.use('classic')
wavlist = pywt.wavelist(kind='continuous')
print("Class of continuous wavelet functions:")
print(wavlist)
wavename = 'mexh'    # "cmorB-C" where B is the bandwidth and C is the center frequency.
frequencies = pywt.scale2frequency('cmor1.5-0.5', [1, 2, 3, 4]) / (1/sampling_rate)
print('frequencies')
print(frequencies)
totalscal = 62.5    # scale
fc = pywt.central_frequency(wavename)  # central frequency
print('central frequency')
print(fc)
cparam = 2 * fc * totalscal
print('central param')
print(cparam)
scales = cparam/np.arange(1, totalscal+1)
print('scales')
print(scales)
# C3 channel
[cwtmatr, frequencies] = pywt.cwt(ch_C3_res, scales, wavename, 1.0/sampling_rate)  # continuous wavelet transform
fig = plt.figure(1)
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.colorbar()
fig.savefig('C3.png')
# C4 channel
fig = plt.figure(2)
[cwtmatr, frequencies] = pywt.cwt(ch_C4_res, scales, wavename, 1.0/sampling_rate)
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.colorbar()
fig.savefig('C4.png')
'''
plt.show()
