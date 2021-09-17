import time
import numpy as np
import pandas as pd
import os
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import mne
from mne.channels import read_layout
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

params = BrainFlowInputParams()
params.serial_port = "COM3"
board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
board.prepare_session()
board.start_stream()
time.sleep(10)
data = board.get_board_data()
board.stop_stream()
board.release_session()

# eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
# df = pd.DataFrame(np.transpose(data))
# print('Data From the Board')
# print(df.head(10))

DataFilter.write_file(data, 'test.csv', 'w')  # use 'a' for append mode
restored_data = DataFilter.read_file('test.csv')
restored_df = pd.DataFrame(np.transpose(restored_data))
print('Data From the File')
print(restored_df.head(10))

# print(data[:, 0])
# print(np.shape(data))
# print(np.shape(data[1]))
# eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
# eeg_data = data[eeg_channels, :]
# print(eeg_channels)
# print(np.shape(eeg_data))
# for i in eeg_data[:, 0]:
#     print(i)
#     break


# eeg_data = eeg_data / 1000000 # BrainFlow returns uV, convert to V for MNE
#
# # Creating MNE objects from brainflow data arrays
# ch_types = ['eeg'] * len(eeg_channels)
# ch_names = BoardShim.get_eeg_names(BoardIds.CYTON_DAISY_BOARD.value)
# sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
# info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
# raw = mne.io.RawArray(eeg_data, info)
# # its time to plot something!
# raw.plot_psd(average=False)
# # raw.plot(duration=20, n_channels=16, block=True)
#
# plt.show()
