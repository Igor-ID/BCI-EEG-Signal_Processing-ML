import numpy as np
import os
import random
import time
import mne
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import pandas as pd
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
    df = pd.DataFrame(np.transpose(npcsv[:, :17]))
    df = df.T
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


def create_mne_data_from_numpy(starting_dir = "../data"):
    # starting_dir = "../data"  # or "os.path.dirname(os.path.realpath(__file__))/acquisition/data"
    archetype_dir = 'Sun'
    training_data = []
    data_dir = os.path.join(starting_dir, archetype_dir)
    for item in os.listdir(data_dir):
        data = np.load(os.path.join(data_dir, item))
        print(data[0])
        for i in data:
            training_data.append(i)

    # training_data = np.asarray(training_data)
    reshaped = np.reshape(training_data, (16000, 16))
    data_mne = (reshaped / 1000000).T
    raw = mne.io.RawArray(data_mne, info)
    df_mne = raw.to_data_frame()
    return raw

starting_dir = "../data"  # or "os.path.dirname(os.path.realpath(__file__))/acquisition/data"
archetype_dir = 'Sun'
training_data = []
data_dir = os.path.join(starting_dir, archetype_dir)
for item in os.listdir(data_dir):
    data = np.load(os.path.join(data_dir, item))
    print(data[0])
    for i in data:
        training_data.append(i)

# training_data = np.asarray(training_data)
reshaped = np.reshape(training_data, (16000, 16))
print('-----------')
print(reshaped[0])
print('-----------')
data_mne = (reshaped / 1000000).T
print(len(data_mne[0]))
raw = mne.io.RawArray(data_mne, info)
print(raw.info)
df_mne = raw.to_data_frame()
print('Data From the MNE approach')
print(df_mne.head(10))

# os.mkdir("../data")

