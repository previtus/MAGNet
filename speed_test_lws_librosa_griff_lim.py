### We have two implementations:
###  - librosa and griff. lim algo
###  - lws library and it's own stft
from timeit import default_timer as timer

import librosa
fft_size=2048
window_size=1024 # window size
hop_size=512     # window shift size - 1024 by 512 means 50% overlap
sample_rate=44100

#sample_rate=22050 # whoa, freaky aliasing (right?)

# settings sensitive - Louis was using lws.setup(512, 128); => awin_or_fsize, fshift
#window_size = 512
#hop_size = 128

lws_L = 5 # approximation order, default
#lws_L = 20 # slower and not better rly
#lws_L = 1 # faster and (a bit?) badder

griffin_iterations = 60 # was default
#griffin_iterations = 200 # much slower,
#griffin_iterations = 20 # faster obv

def griffin_lim(stftm_matrix, max_iter=100):
    """"Iterative method to 'build' phases for magnitudes."""
    stft_matrix = np.random.random(stftm_matrix.shape)
    y = librosa.core.istft(stft_matrix, hop_size, window_size)
    for i in range(max_iter):
        stft_matrix = librosa.core.stft(y, fft_size, hop_size, window_size)
        stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
        y = librosa.core.istft(stft_matrix, hop_size, window_size)
    return y

def griffin_lim_HACKYROBUST(stftm_matrix, max_iter=100):
    """"Iterative method to 'build' phases for magnitudes."""
    stft_matrix = np.random.random(stftm_matrix.shape)
    y = librosa.core.istft(stft_matrix, hop_size, window_size)

    if not np.isfinite(y).all():
        print("Problem with the signal - it's not finite (contains inf or NaN)")
        print("Signal = ", y)
        y = np.nan_to_num(y)
        print("Attempted hacky fix")

    for i in range(max_iter):
        if not np.isfinite(y).all():
            print("Problem with the signal - it's not finite (contains inf or NaN), in iteration", i)
            print("Signal = ", y)
            y = np.nan_to_num(y)
            print("Attempted hacky fix inside the iterative method")

        stft_matrix = librosa.core.stft(y, fft_size, hop_size, window_size)
        stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
        y = librosa.core.istft(stft_matrix, hop_size, window_size)
    return y

# Concern about stft implementations:
#   (librosa) predicted_magnitudes: (1040, 1025)
#   (lws) audio X0: (1042, 1025)


def griff_lim_librosa_reconstruct(audio, griffin_iterations = 60):
    ## This is to showcase that even this simple and immediate reconstruction brings large loss in quality!
    print("\tinput shape", np.asarray(audio).shape)

    fft_frames = []

    data = audio
    mags_phases = librosa.stft(data, n_fft=fft_size,win_length=window_size,hop_length=hop_size)
    magnitudes, phases = librosa.magphase(mags_phases)
    for magnitude_bins in magnitudes.T:
        fft_frames += [magnitude_bins]

    fft_frames = np.asarray(fft_frames)
    represented = fft_frames
    # now reconstruct from:
    print("\trepresented shape", represented.shape)

    predicted_magnitudes = represented
    #audio_reconstruction = griffin_lim(predicted_magnitudes.T, max_iter=griffin_iterations)
    audio_reconstruction = griffin_lim_HACKYROBUST(predicted_magnitudes.T, max_iter=griffin_iterations)
    audio_reconstruction = np.asarray(audio_reconstruction)
    print("\taudio_reconstruction shape", audio_reconstruction.shape)
    print("\taudio_reconstruction data type:", audio_reconstruction.dtype)

    return audio_reconstruction

import lws
import numpy as np
def lws_deconstruct_reconstruct(audio):
    print("\tinput shape", np.asarray(audio).shape) # desired shape is 1 channel data = (1, M)
    x = audio

    lws_processor=lws.lws(window_size,hop_size, L=lws_L, fftsize=fft_size, mode="music")
    #lws_processor=lws.lws(window_size,hop_size, fftsize=fft_size,
    #        mode = None,nofuture_iterations = 1,online_iterations = 1,batch_iterations = 50)

    X = lws_processor.stft(x) # where x is a single-channel waveform
    #print("audio X:", np.asarray(X).shape) #()

    X0 = np.abs(X) # Magnitude spectrogram

    # added ... griff lim was working with 32 bits, nn's will as well
    # X0 = np.float32(X0) # ... maybe shouldnt - https://github.com/Jonathan-LeRoux/lws/issues/11


    """
    # low pass on spectrogram? # HAX
    for ai in range(len(X0)):
        for bi in range(int(len(X0[0])/2),len(X0[0])):
            X0[ai,bi] = 0
    """

    #print("audio X0:", np.asarray(X0).shape) #(4159, 257)
    #print('{:6}: {:5.2f} dB'.format('Abs(X)', lws_processor.get_consistency(X0)))

    X1 = lws_processor.run_lws(X0) # reconstruction from magnitude (in general, one can reconstruct from an initial complex spectrogram)

    #print('{:6}: {:5.2f} dB'.format('LWS', lws_processor.get_consistency(X1)))
    #print("audio X1:", np.asarray(X1).shape) #(4159, 257)

    represented = X1
    # now reconstruct from:
    print("\trepresented shape", represented.shape)


    reconstruction = lws_processor.istft(represented) # where x is a single-channel waveform
    reconstruction = np.asarray(reconstruction)
    print("\toutput reconstruction:", reconstruction.shape) # (531968,)
    print("\treconstruction data type:", reconstruction.dtype)

    return reconstruction

""" maybe shouldnt be doing this on the signal, but on the spectro
def low_pass_filter_scipy_2(x, cutoff_hz = 16000): #Butter
    from scipy.signal import butter, lfilter

    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    y = butter_lowpass_filter(x, cutoff_hz, sample_rate)
    return y

def low_pass_filter_scipy(x, cutoff_hz = 16000): #FIR
    from scipy.signal import kaiserord, lfilter, firwin, freqz

    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0 / nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = 10.0

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

    # Use lfilter to filter x with the FIR filter.
    filtered_x = lfilter(taps, 1.0, x)
    return filtered_x
"""

def plot_audio(audio, title="Audio signal", save=""):
    import matplotlib.pyplot as plt
    plt.specgram(audio, NFFT=fft_size, Fs=sample_rate, noverlap=512)

    # Plot a spectrogram
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.title(title)

    if len(save) > 0:
        plt.savefig(save)
    plt.show()

""" # Ok maybe this makes no sense, LWS seems to have to be initialized too
def hybrid(audio):
    # Previous data processing:

    fft_frames = []
    data = audio
    mags_phases = librosa.stft(data, n_fft=fft_size,win_length=window_size,hop_length=hop_size)
    magnitudes, phases = librosa.magphase(mags_phases)
    for magnitude_bins in magnitudes.T:
        fft_frames += [magnitude_bins]
    fft_frames = np.asarray(fft_frames)
    represented = fft_frames

    # with LWS

    lws_processor = lws.lws(window_size, hop_size, fftsize=fft_size, mode="music")
    # lws_processor=lws.lws(window_size,hop_size, fftsize=fft_size,
    #        mode = None,nofuture_iterations = 1,online_iterations = 1,batch_iterations = 50)

    X1 = represented

    print('{:6}: {:5.2f} dB'.format('LWS', lws_processor.get_consistency(X1)))
    print("audio X1:", np.asarray(X1).shape)  # (4159, 257)

    reconstruction = lws_processor.istft(X1)  # where x is a single-channel waveform
    reconstruction = np.asarray(reconstruction)
    print("output reconstruction:", np.asarray(reconstruction).shape)  # (531968,)

    return reconstruction
"""

import numpy as np
#audio = np.load("tmp_audio_1stConvUsingGrifLim.npy")
#print("Input audio:", np.asarray(audio).shape)

file = "/home/vitek/Downloads/_music_samples/SwordSworceryLP.wav"
file = "/home/vitek/Downloads/_music_samples/brad-mehldau.wav"
data, sample_rate = librosa.load(file, sr=sample_rate, mono=True)
#data = np.append(np.zeros(window_size * 200), data)

skip = window_size * 200
audio = data[skip:531968+skip]
print("Input audio:", np.asarray(audio).shape)


print("Griff lim test:")
start = timer()
reconstruction_griff = griff_lim_librosa_reconstruct(audio, griffin_iterations)
end = timer()
time = (end - start)
print("Griff took " + str(time) + "s (" + str(time / 60.0) + "min)")

print("LWS test:")
start = timer()
reconstruction_lws = lws_deconstruct_reconstruct(audio)
end = timer()
time = (end - start)
print("LWS took " + str(time) + "s (" + str(time / 60.0) + "min)")

"""
print("Hybrid (librosa data + lws wilderness) test:")
start = timer()
reconstruction_hybrid = hybrid(audio)
end = timer()
time = (end - start)
print("Hybrid (librosa data + lws wilderness) took " + str(time) + "s (" + str(time / 60.0) + "min)")
"""

audio = audio[skip:]
reconstruction_griff = reconstruction_griff[skip:]
reconstruction_lws = reconstruction_lws[skip:]
#reconstruction_hybrid = reconstruction_hybrid[skip:]


##reconstruction_lws_lowpassed = low_pass_filter_scipy_2(reconstruction_lws)

plot_audio(audio, title="Original audio", save="original.png")
plot_audio(reconstruction_griff, title="Reconstructed Griff.lim. "+str(griffin_iterations)+" it", save="griff.png")
plot_audio(reconstruction_lws, title="Reconstructed LWS", save="lws.png")
##plot_audio(reconstruction_lws_lowpassed, title="Reconstructed LWS + lowpass (15kHz)", save="lws_lowpass.png")
#plot_audio(reconstruction_hybrid, title="Reconstructed librosa+LWS", save="hybridlws.png")

import scipy.io.wavfile
scipy.io.wavfile.write("original.wav", sample_rate, audio)
scipy.io.wavfile.write("reconstruction_griff_"+str(griffin_iterations)+".wav", sample_rate, reconstruction_griff)
scipy.io.wavfile.write("reconstruction_lws.wav", sample_rate, reconstruction_lws)
##scipy.io.wavfile.write("reconstruction_lws_lowpass.wav", sample_rate, reconstruction_lws_lowpassed)
#scipy.io.wavfile.write("reconstruction_hybridlws.wav", sample_rate, reconstruction_hybrid)
