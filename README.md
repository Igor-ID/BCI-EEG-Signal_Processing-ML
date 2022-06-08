## BCI-EEG-Signal_Processing-ML

The purpose of the repository is to implement the best practices in the BCI field. The ultimate goal is to integrate them into my long-term research interest, namely the notion of how Jungian archetypal symbols can be related to cognitive processes, as well as the correlation between archetypal motifs and mental patterns, including emotions, memory, semantic cognition, consciousness, recognition, and so on.
Headset used to create the BCI is the OpenBCI 16-channel Ultracortex Mark IV.

### 1. Data acquisition
Acquiring data from an OpenBCI stream. 
In the acquisition directory, you can use stream_lsl.py or stream_brainflow.py to acquire EEG data. stream_lsl.py acquires the data using the OpenBCI GUI via the LSL protocol, in this case, you get the EEG data as a bunch of NumPy arrays.
stream_brainflow.py is the recommended way to acquire data from OpenBCI. In this case, you can get much more information, such as sampling rate, channel names, AUX channel data, and much more. [see OpenBCI documentation](https://docs.google.com/document/d/e/2PACX-1vR_4DXPTh1nuiOwWKwIZN3NkGP3kRwpP4Hu6fQmy3jRAOaydOuEI1jket6V4V6PG4yIG15H1N7oFfdV/pub)

