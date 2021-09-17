from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
import csv

last_print = time.time()
# fps_counter = deque(maxlen=150)
channel_data = []

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

for i in range(100):  # how many iterations. Eventually this would be a while True
    # ch_data = []
    sample, timestamp = inlet.pull_sample()
    channel_data.append(sample)
    # if we're going to work with FFT data it's necessary to work with each bin (channel) to obtain 125 points from
    # each bin, in this case we need one more nested for loop. In my case I only need raw time series.
    # for j in range(16):  # each of the 16 channels here
    #     sample, timestamp = inlet.pull_sample()
    #     .....
    # fps_counter.append(time.time() - last_print)
    # last_print = time.time()
    # cur_raw = 1/(sum(fps_counter)/len(fps_counter))
    # print(cur_raw)

print(np.shape(channel_data))

with open("output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(channel_data)

