from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
import csv
import os
import random

ARCH_IM = 'Hades' # description of the evaluated image

# MAX_HZ = 60  # not relevant working with time series
last_print = time.time()
# fps_counter = deque(maxlen=150)
channel_data = []
fps_counter = deque(maxlen=150)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

channel_datas = []
# print(f'Input Info name: {inlet.info().name()}')
# print(f'Input Info type: {inlet.info().type()}')
# print(f'Input Info Channels: {inlet.info().channel_count()}')
# print(f'Input Info Channel format: {inlet.info().channel_format()}')
print(f'Input Info xml description: {inlet.info().as_xml()}')

for i in range(100):  # how many iterations. Eventually this would be a while True
    channel_data = []
    for i in range(16):  # each of the 16 channels here
        # get the sample - for the case of raw time series of shape (100, 16, 16)
        # To remap time stamp to the local clock, add the value returned by .time_correction() to it.
        sample, timestamp = inlet.pull_sample()
        channel_data.append(sample)
        # channel_data.append(sample[:MAX_HZ])
        # print(f'Inner timestamp {timestamp}')

    # fps_counter.append(time.time() - last_print)
    # last_print = time.time()
    # cur_raw_hz = 1 / (sum(fps_counter) / len(fps_counter))
    # print(f"HZ of raw data: {cur_raw_hz}")

    channel_datas.append(channel_data)
# print(f'post print {time.time()}')

datadir = "../data"
if not os.path.exists(datadir):
    os.mkdir(datadir)

actiondir = f"{datadir}/{ARCH_IM}"
if not os.path.exists(actiondir):
    os.mkdir(actiondir)

print(len(channel_datas))

print(f"saving {ARCH_IM} data...")
np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))
print("done.")

# with open("output.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(channel_data)

