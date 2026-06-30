import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import waveform


# Get the coefficients associated with a polynomial of degree deg
def get_weights(x, y, deg):
    x = np.array(x)
    y = np.array(y)
    M = np.matrix([[sum(x ** (j - i)) for j in range(deg * 2, deg - 1, -1)] for i in range(deg + 1)])
    s = np.matrix([sum(x ** j * y) for j in range(deg, -1, -1)]).T
    #sp.Matrix(M)*sp.Matrix(s)
    return np.array(M.I * s).flatten()

# polynomial approximation function
def poly_f(x, w):
    x = np.array(x)
    w = np.array(w)
    return sum(x ** j * w[-j - 1] for j in range(len(w) - 1, -1, -1))

A = 55
D = A * 2 ** (5/12)
E = A * 2 ** (7/12)
freq_kernel = [A, D, E]
drifts = waveform.generate_frequency_drift(freq_kernel, .3, 3)
#t, s, wave = waveform.combine_random_smoothed(1, 3, drifts, waveform.SAMPLE_RATE, n=17, M=13)
#ef = waveform.play(wave)
#wave = wave[:, 0]
wave, sr = sf.read("audio/acoustic_guitar.mp3")
wave = wave[:, 0]

sf.write("huh.wav", wave, waveform.SAMPLE_RATE)


new_sample_rate = 2 ** (2/12) * waveform.SAMPLE_RATE
new_sample_count = int(new_sample_rate * 3)
new_sample_index = 0
sample_delta = waveform.SAMPLE_RATE / new_sample_rate
resampled = []
window_start = 0
window_len = 4
interpolating_samples = np.array([i for i in range(window_len)])
weights = get_weights(interpolating_samples, wave[window_start:window_start+window_len], 3)
while new_sample_index < new_sample_count:
    if new_sample_index >= window_start + window_len:
        window_start += window_len
        weights = get_weights(interpolating_samples, wave[window_start:window_start+window_len], 3)
    x = new_sample_index - window_start
    resampled.append(poly_f(x, weights))
    new_sample_index += sample_delta

what = np.array(resampled)
sf.write("huh_semitone_up.wav", what, waveform.SAMPLE_RATE)
