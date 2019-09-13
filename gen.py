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
