# BCI-EEG-Signal_Processing-ML
These files contemplate my scientific initiation project in digital signal processing. In this project I developed a BCI in Python using  Cyton + Daisy Biosensing Boards from OpenBCI. I'm going to implement ML algorythms for different purposes (e.g. emotion/image recognition)

## 1. Data acquisition
Acquiring data from an OpenBCI stream.
In the acquisition directory, you can use stream_lsl.py or stream_brainflow.py to get the data. stream_lsl.py gets the data using the OpenBCI GUI, in this case you get the data as a bunch of arrays. This is useful because you can choose from many of the options available in the OpenBCI GUI, you can get preprocessed signal (FFT) and other useful data [see OpenBCI documentation](https://docs.google.com/document/d/e/2PACX-1vR_4DXPTh1nuiOwWKwIZN3NkGP3kRwpP4Hu6fQmy3jRAOaydOuEI1jket6V4V6PG4yIG15H1N7oFfdV/pub)
stream_brainflow.py is the recommended way to get data from openBCI. In this case, you can get much more information, such as channel names, AUX channel data for evoked potentials, and much more.
For now we need only raw data, so both options work.
