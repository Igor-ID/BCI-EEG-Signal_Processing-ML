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

'''
We get a data in the each numpy file of shape (num_timepoint, n_channels, num_point) (100, 16, 16)
We do not really need number of point, so we need to reshape the data to 2D array of shape (1600, 16) for each file.
Interestingly, that data acquired through OpenBCI GUI LSL and pyLSL differ from data created by OpenBCI GUI and 
written to txt file, but this difference is insignificant 
(e.g. (-9661.29589844, -9801.44140625, -9661.07226562) vs (-9661.29567161747, -9801.441109352245, -9661.072154172916))
'''

# a = np.arange(27).reshape(10, 5, 5)
# x = np.array([[[1,1,2], [2,3,3]], [[1,1,1], [1,1,1]], [[5,1,1], [5,1,1]], [[7,1,1], [7,1,1]]])
# x_r = x.reshape(8, 3)
# print(x.shape)
# print(x)
# print(x_r.shape)
# print(x_r)
# print('------')
# print(x[0:3, 1, 0:1])
# print(x_r[0:3, 0:1])


starting_dir = "data/Hades/"  # or "os.path.dirname(os.path.realpath(__file__))/acquisition/data"
archetype_dir = 'Sun'
data = np.load(os.path.join(starting_dir, "1648771418.npy"))
# data_reshaped = np.reshape(data, (1600, 16))
data_reshaped = data.reshape(1600, 16)
# df = pd.DataFrame(np.transpose(data))
print(data_reshaped.shape)
print(data_reshaped.ndim)
print(data_reshaped[0:20, 0:1])
print('-------')
print(data_reshaped[0:20, 0])
print('-------')
print(data_reshaped[0:20, 1])
print(data.shape)
print(data.ndim)
print(data[0:20, 0, 0])
print('-------')
print(data[0:20, 0, 1])
print('-------')
print(data[0:20, :, 0])
print('-------')
print(data[0:20, :, 1])


# print(df.head())
# training_data = []
#     data_dir = os.path.join(starting_dir, archetype_dir)
#     for item in os.listdir(data_dir):
#         data = np.load(os.path.join(data_dir, item))
#         for i in data:
#             training_data.append(i)
#
#     # training_data = np.asarray(training_data)
#     reshaped = np.reshape(training_data, (16000, 16))

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

ACTIONS = ["feet", "none", "hands"]


def split_data(starting_dir="personal_dataset", splitting_percentage=(70, 20, 10), shuffle=True, coupling=False,
               division_factor=0):
    """
        This function splits the dataset in three folders, training, validation, test
        Has to be run just everytime the dataset is changed
    :param starting_dir: string, the directory of the dataset
    :param splitting_percentage:  tuple, (training_percentage, validation_percentage, test_percentage)
    :param shuffle: bool, decides if the personal_dataset will be shuffled
    :param coupling: bool, decides if samples are shuffled singularly or by couples
    :param division_factor: int, if the personal_dataset used is made of FFTs which are taken from multiple sittings
                            one sample might be very similar to an adjacent one, so not all the samples
                            should be considered because some very similar samples could fall both in
                            validation and training, thus the division_factor divides the personal_dataset.
                            if division_factor == 0 the function will maintain all the personal_dataset
    """
    training_per, validation_per, test_per = splitting_percentage

    if not os.path.exists("training_data") and not os.path.exists("validation_data") \
            and not os.path.exists("test_data"):

        # creating directories

        os.mkdir("training_data")
        os.mkdir("validation_data")
        os.mkdir("test_data")

        for action in ACTIONS:

            action_data = []
            all_action_data = []
            # this will contain all the samples relative to the action

            data_dir = os.path.join(starting_dir, action)
            # sorted will make sure that the personal_dataset is appended in the order of acquisition
            # since each sample file is saved as "timestamp".npy
            for file in sorted(os.listdir(data_dir)):
                # each item is a ndarray of shape (8, 90) that represents ≈1sec of acquisition
                all_action_data.append(np.load(os.path.join(data_dir, file)))

            # TODO: make this coupling part readable
            # coupling was used when overlapping FFTs were used
            # is now deprecated with EEG models and very time-distant acquisitions
            if coupling:
                # coupling near time acquired samples to reduce the probability of having
                # similar samples in both train and validation sets
                coupled_actions = []
                first = True
                for i in range(len(all_action_data)):
                    if division_factor != 0:
                        if i % division_factor == 0:
                            if first:
                                tmp_act = all_action_data[i]
                                first = False
                            else:
                                coupled_actions.append([tmp_act, all_action_data[i]])
                                first = True
                    else:
                        if first:
                            tmp_act = all_action_data[i]
                            first = False
                        else:
                            coupled_actions.append([tmp_act, all_action_data[i]])
                            first = True

                if shuffle:
                    np.random.shuffle(coupled_actions)

                # reformatting all the samples in a single list
                for i in range(len(coupled_actions)):
                    for j in range(len(coupled_actions[i])):
                        action_data.append(coupled_actions[i][j])

            else:
                for i in range(len(all_action_data)):
                    if division_factor != 0:
                        if i % division_factor == 0:
                            action_data.append(all_action_data[i])
                    else:
                        action_data = all_action_data

                if shuffle:
                    np.random.shuffle(action_data)

            num_training_samples = int(len(action_data) * training_per / 100)
            num_validation_samples = int(len(action_data) * validation_per / 100)
            num_test_samples = int(len(action_data) * test_per / 100)

            # creating subdirectories for each action
            tmp_dir = os.path.join("training_data", action)
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            for sample in range(num_training_samples):
                np.save(file=os.path.join(tmp_dir, str(sample)), arr=action_data[sample])

            tmp_dir = os.path.join("validation_data", action)
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            for sample in range(num_training_samples, num_training_samples + num_validation_samples):
                np.save(file=os.path.join(tmp_dir, str(sample)), arr=action_data[sample])

            if test_per != 0:
                tmp_dir = os.path.join("test_data", action)
                if not os.path.exists(tmp_dir):
                    os.mkdir(tmp_dir)
                for sample in range(num_training_samples + num_validation_samples,
                                    num_training_samples + num_validation_samples + num_test_samples):
                    np.save(file=os.path.join(tmp_dir, str(sample)), arr=action_data[sample])