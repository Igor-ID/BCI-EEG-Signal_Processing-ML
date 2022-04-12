import numpy as np
import os
import random
import time
import mne
from mne.time_frequency import tfr_morlet
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Try to get data using brainflow.
# Match data of lsl stream with txt file from OpenBCI GUI to insure in correct chanel sequence.
# Add info to the lsl stream.
# convert to MNE raw data and preprocess.
# Use wavelet transforms.

eeg_ch = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3', 'F4', 'T7', 'T8', 'P3', 'P4']
sfreq = 125
info = mne.create_info(eeg_ch, sfreq, ch_types='eeg')
info.set_montage('standard_1020')

# if using brainflow acquisition approach uncomment the following code and use the following function
def create_mne_data_from_csv(csv_brainflow_dir='../acquisition/test.csv'):
    npcsv = np.loadtxt(csv_brainflow_dir)  # or use equivalent np.genfromtxt('../acquisition/test.csv')
    # Slice data to 17 columns one time and 16 channel columns and create the DataFrame from csv
    # df = pd.DataFrame(np.transpose(npcsv[:, :17]))
    # df = df.T
    # print('Data From the csv File')
    # print(df.head(10))
    # Get data from csv as of shape (n_channels, n_samples) and convert to Volts
    # eeg_data = eeg_data / 1000000 # BrainFlow returns uV, convert to V for MNE
    data_mne = (npcsv[:, 1:17] / 1000000).T
    # Create the raw MNE object from csv. A NumPy array must be of shape (n_channels, n_samples)
    raw = mne.io.RawArray(data_mne, info)
    return raw
    # print(raw.info)
    # df_mne = raw.to_data_frame()
    # print('Data From the MNE approach')
    # print(df_mne.head(10))
    # raw.plot(show_scrollbars=True, show_scalebars=True)
    # plt.show()


def create_mne_data_from_numpy(starting_dir="../data", archetype_dir='Sun'):
    # starting_dir = "../data"  # or "os.path.dirname(os.path.realpath(__file__))/acquisition/data"
    # archetype_dir = 'Sun'
    training_data = []
    data_dir = os.path.join(starting_dir, archetype_dir)
    for item in os.listdir(data_dir):
        data = np.load(os.path.join(data_dir, item))
        for i in data:
            training_data.append(i)

    # training_data = np.asarray(training_data)
    reshaped = np.reshape(training_data, (16000, 16))
    data_mne = (reshaped / 1000000).T
    info['description'] = f'Archetype_{archetype_dir}'
    raw = mne.io.RawArray(data_mne, info)
    # df_mne = raw.to_data_frame()
    return raw


mne_raw_sun = create_mne_data_from_numpy()
# mne_raw_sun.plot()
freq = 125
# mne_raw_sun.notch_filter(freqs=60)
# print(len(mne_raw_sun))
# data, times = mne_raw_sun[:]
# print(times)
# print(len(times))
# data1, times1 = mne_raw_sun.get_data(['Fp1', 'Fp2'])
# print(times1)
# print(mne_raw_sun.info)
df = mne_raw_sun.to_data_frame()
freqs = range(1, 125)

# power = tfr_morlet(mne_raw_sun, freqs=freqs, n_cycles=3, return_itc=False)
# power.plot()
# print(df.head())
numpy_ar = df.drop(columns='time').to_numpy()
print(numpy_ar.shape)
cwtcoef, cwtfreq = pywt.cwt(numpy_ar, freqs, wavelet='gaus1', sampling_period=(1 / freq))
print(cwtcoef.itemsize)
print('--------')
print(cwtcoef.ndim)
print('--------')
print(cwtcoef.shape)
print('--------')
print(cwtcoef.size)
print('--------')
print(cwtfreq.shape)

# plt.figure(figsize=(15, 10))
# plt.imshow(cwtcoef, )

# plot spectrogram using contourf
fig, ax = plt.subplots(figsize=(15, 10))
period = 1. / cwtfreq
t = np.arange(0, 128, 1 / freq)  # time between 0 and 128 seconds
im = ax.contourf(t, cwtfreq, cwtcoef[:, :, 1], extend='both')

# fig = plt.figure(figsize=(14, 8))
# ax1 = fig.add_subplot(1, 4, 1)
# plt.plot(cwtcoef[:, :, 1], freqs, color='red')
# ax2 = fig.add_subplot(1, 4, 2)
# plt.plot(cwtcoef[:, :, 10], freqs, color='blue')
# ax3 = fig.add_subplot(1, 4, 3)
# plt.plot(cwtcoef[:, :, 1], cwtfreq, color='green')
# ax4 = fig.add_subplot(1, 4, 4)
# plt.plot(cwtcoef[:, :, 10], cwtfreq, color='black')
# print(numpy_ar.shape)

# mne_raw_sun.plot()
# mne_raw_sun.plot_sensors(show_names=True, kind='3d')
# print(mne_raw_sun.info)
# starting_dir = "../data"  # or "os.path.dirname(os.path.realpath(__file__))/acquisition/data"
# archetype_dir = ['Sun', 'Hades']
# training_data = {}
# for arch in archetype_dir:
#     if arch not in training_data:
#         training_data[arch] = []
#     data_dir = os.path.join(starting_dir, arch)
#     for item in os.listdir(data_dir):
#         data = np.load(os.path.join(data_dir, item))
#         for i in data:
#             training_data[arch].append(i)
#
# combined_data = []
# for arch in archetype_dir:
#     for data in training_data[arch]:
#         if arch == "Sun":
#             combined_data.append([data, [1, 0]])
#         elif arch == "Hades":
#             combined_data.append([data, [0, 1]])
#
# print(combined_data[0])

# training_data = np.asarray(training_data)
# reshaped = np.reshape(training_data, (16000, 16))
# print('-----------')
# print(reshaped[0])
# print('-----------')
# data_mne = (reshaped / 1000000).T
# print(len(data_mne[0]))
# raw = mne.io.RawArray(data_mne, info)
# print(raw.info)
# df_mne = raw.to_data_frame()
# print('Data From the MNE approach')
# print(df_mne.head(10))
plt.show()
# os.mkdir("../data")

