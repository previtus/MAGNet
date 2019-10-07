import sys
import tensorflow as tf
import pywt
from utils.audio_dataset_generator import AudioDatasetGenerator
import numpy as np
#import tensorflowjs as tfjs


#audio_data_path      = "/media/vitek/Data/Vitek/Projects/2019_LONDON/music generation/small_file/"
audio_data_path = "/cluster/work/igp_psr/ruzickav/MAGnet_large_files/data_dnb1/"
#sample_rate          = 44100
#fft_settings         = [2048, 1024, 512] # < orig

sample_rate          = 22050
fft_settings         = [2048, 2048, 512] # good setting from others

amount_epochs        = 300
amount_epochs        = 3
model_name = "KerasModel.h5"

# or 2 * 256
number_rnn_layers    = 3  # < I guess test both of these versions
rnn_number_units     = 128


# Model
load_model           = False

# Dataset
sequence_length      = 40


force_new_dataset    = True
# Feature Extraction and Audio Genreation
fft_size             = fft_settings[0]
window_size          = fft_settings[1]
hop_size             = fft_settings[2]

# General Network
learning_rate        = 0.001
batch_size           = 64
loss_type            = "mse"
weight_decay         = 0.0001

# Recurrent Neural Network
rnn_type             = "lstm"

# Is also ADAM, however this model doesn't use Dropouts
#    the other lstm's end with dropout(net, 1 - keep_prob) #1 - 0.2

# Make your dataset

dataset = AudioDatasetGenerator(fft_size, window_size, hop_size,
                                sequence_length, sample_rate)

dataset.load(audio_data_path, force_new_dataset)

# Set up the model

model = tf.keras.Sequential()

model.add(tf.keras.layers.BatchNormalization(input_shape=[dataset.x_frames.shape[1], dataset.x_frames.shape[2]]))

for layer in range(number_rnn_layers):
    return_sequence = False if layer == (number_rnn_layers - 1) else True
    model.add(tf.keras.layers.LSTM(rnn_number_units, return_sequences=return_sequence))

model.add(tf.keras.layers.Dense(dataset.y_frames.shape[1]))

model.add(tf.keras.layers.Activation('linear'))
opt = tf.keras.optimizers.Adam(learning_rate)
model.compile(optimizer=opt, loss=loss_type)

# Train

model.fit(dataset.x_frames, dataset.y_frames, batch_size=batch_size, epochs=amount_epochs)

# Save your model

model.save(model_name)

### JS conversion
#tfjs.converters.save_keras_model(model, "KerasModel.json")


# (2) Generate samples:

# Or generate samples in python

amount_samples = 1
sequence_length_max = 500
impulse_scale = 1.0
griffin_iterations = 60
random_chance = 0.05
random_strength = 0.0

use_cnn = False
use_wavelets = False
wavelet              = 'db10'

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
        predicted_magnitudes = np.array(predicted_magnitudes).reshape(-1, int(window_size) + 1)
        audio += [dataset.griffin_lim(predicted_magnitudes.T, griffin_iterations)]
audio = np.array(audio)


# Plot:

import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

plt.specgram(audio[i], NFFT=2048, Fs=sample_rate, noverlap=512)

# Plot a spectrogram
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
