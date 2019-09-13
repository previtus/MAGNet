import sys
import tensorflow as tf
import tflearn
import pywt
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.core import dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from utils.audio_dataset_generator import AudioDatasetGenerator
from utils.audio_dataset_generator import AudioWaveletDatasetGenerator
import numpy as np

# Dataset
sequence_length      = 40
audio_data_path      = "data/"
force_new_dataset    = True
use_wavelets         = False
predict_coeff        = False
wavelet              = 'db10'

# Feature Extraction and Audio Genreation
sample_rate          = 44100
fft_settings         = [2048, 1024, 512]
fft_size             = fft_settings[0]
window_size          = fft_settings[1]
hop_size             = fft_settings[2]

# General Network
learning_rate        = 1e-3
keep_prob            = 0.2
loss_type            = "mean_square"
activation           = 'tanh'
optimiser            = 'adam'
fully_connected_dim  = 1024

# Recurrent Neural Network
rnn_type             = "lstm"
number_rnn_layers    = 3
rnn_number_units     = 128

# Convolutional Neural Network
use_cnn              = False
number_filters       = [32]
filter_sizes         = [3]



if use_wavelets:
    dataset = AudioWaveletDatasetGenerator(1024, sequence_length,
                                    sample_rate)
else:
    dataset = AudioDatasetGenerator(fft_size, window_size, hop_size,
                                sequence_length, sample_rate)

dataset.load(audio_data_path, force_new_dataset)#, mp3=True)

if use_cnn:
    dataset.x_frames = dataset.x_frames.reshape(dataset.x_frames.shape[0],
                                                dataset.x_frames.shape[1],
                                                dataset.x_frames.shape[2], 1)
print(dataset.x_frames.shape, dataset.y_frames.shape)



def conv_net(net, filters, kernels, non_linearity):
    """
    A quick function to build a conv net.
    At the end it reshapes the network to be 3d to work with recurrent units.
    """
    assert len(filters) == len(kernels)

    for i in range(len(filters)):
        net = conv_2d(net, filters[i], kernels[i], activation=non_linearity)
        net = max_pool_2d(net, 2)

    dim1 = net.get_shape().as_list()[1]
    dim2 = net.get_shape().as_list()[2]
    dim3 = net.get_shape().as_list()[3]
    return tf.reshape(net, [-1, dim1 * dim3, dim2])


def recurrent_net(net, rec_type, rec_size, return_sequence):
    """
    A quick if else block to build a recurrent layer, based on the type specified
    by the user.
    """
    if rec_type == 'lstm':
        net = tflearn.layers.recurrent.lstm(net, rec_size, return_seq=return_sequence)
    elif rec_type == 'gru':
        net = tflearn.layers.recurrent.gru(net, rec_size, return_seq=return_sequence)
    elif rec_type == 'bi_lstm':
        net = bidirectional_rnn(net,
                                BasicLSTMCell(rec_size),
                                BasicLSTMCell(rec_size),
                                return_seq=return_sequence)
    elif rec_type == 'bi_gru':
        net = bidirectional_rnn(net,
                                GRUCell(rec_size),
                                GRUCell(rec_size),
                                return_seq=return_sequence)
    else:
        raise ValueError('Incorrect rnn type passed. Try lstm, gru, bi_lstm or bi_gru.')
    return net


# Input
if use_cnn:
    net = tflearn.input_data([None,
                              dataset.x_frames.shape[1],
                              dataset.x_frames.shape[2],
                              dataset.x_frames.shape[3]], name="input_data0")
    net = conv_net(net, number_filters, filter_sizes, activation)
else:
    net = tflearn.input_data([None,
                              dataset.x_frames.shape[1],
                              dataset.x_frames.shape[2]], name="input_data0")

# Batch Norm
net = tflearn.batch_normalization(net, name="batch_norm0")

# Recurrent
for layer in range(number_rnn_layers):
    return_sequence = False if layer == (number_rnn_layers - 1) else True
    net = recurrent_net(net, rnn_type, rnn_number_units, return_sequence)
    net = dropout(net, 1 - keep_prob) if keep_prob < 1.0 else net

# Dense + MLP Out
net = tflearn.fully_connected(net, dataset.y_frames.shape[1],
                              activation=activation,
                              regularizer='L2',
                              weight_decay=0.001)

net = tflearn.fully_connected(net, dataset.y_frames.shape[1],
                              activation='linear')

net = tflearn.regression(net, optimizer=optimiser, learning_rate=learning_rate,
                         loss=loss_type)

model = tflearn.DNN(net, tensorboard_verbose=1)


####################################


model.load("trained_model_last.tfl")

amount_samples = 5
sequence_length_max = 1000
impulse_scale = 1.0
griffin_iterations = 60
random_chance = 0.0
random_strength = 0.0

dimension1 = dataset.x_frames.shape[1]
dimension2 = dataset.x_frames.shape[2]
shape = (1, dimension1, dimension2, 1) if use_cnn else (1, dimension1, dimension2)

audio = []

if use_wavelets:
    temp_audio = np.array(0)
for i in range(amount_samples):

    random_index = np.random.randint(0, (len(dataset.x_frames) - 1))
    impulse = np.array(dataset.x_frames[random_index]) * impulse_scale
    predicted_magnitudes = impulse

    if use_wavelets:
        for seq in range(impulse.shape[0]):
            coeffs = pywt.array_to_coeffs(impulse[seq], dataset.coeff_slices)
            recon = (pywt.waverecn(coeffs, wavelet=wavelet))
            temp_audio = np.append(temp_audio, recon)
    for j in range(sequence_length_max):
        prediction = model.predict(impulse.reshape(shape))
        # Wavelet audio
        if use_wavelets:
            coeffs = pywt.array_to_coeffs(prediction[0], dataset.coeff_slices)
            recon = (pywt.waverecn(coeffs, wavelet=wavelet))
            temp_audio = np.append(temp_audio, recon)

        if use_cnn:
            prediction = prediction.reshape(1, dataset.y_frames.shape[1], 1)

        predicted_magnitudes = np.vstack((predicted_magnitudes, prediction))
        impulse = predicted_magnitudes[-sequence_length:]

        if (np.random.random_sample() < random_chance):
            idx = np.random.randint(0, dataset.sequence_length)
            impulse[idx] = impulse[idx] + np.random.random_sample(impulse[idx].shape) * random_strength

        done = int(float(i * sequence_length_max + j) / float(amount_samples * sequence_length_max) * 100.0) + 1
        sys.stdout.write('{}% audio generation complete.   \r'.format(done))
        sys.stdout.flush()

    if use_wavelets:
        audio += [temp_audio]
    else:
        predicted_magnitudes = np.array(predicted_magnitudes).reshape(-1, window_size + 1)
        audio += [dataset.griffin_lim(predicted_magnitudes.T, griffin_iterations)]
audio = np.array(audio)

###

import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

plt.specgram(audio[i], NFFT=2048, Fs=sample_rate, noverlap=512)

# Plot a spectrogram
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
