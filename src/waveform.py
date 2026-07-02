from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QSoundEffect
import copy
import glob
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.fft as fft
import shutil
import soundfile as sf
import string
import sys
import time

np.random.seed(0)

AUDIO_DIR = "audio"

F = 43.653528929125486

SAMPLE_RATE = 44100

NOTE_LETTERS = ["F", "F#", "G", "G#", "A", "A#", "B", "C", "C#", "D", "D#", "E"]
NOTE_FREQUENCY_MAP = dict(zip(NOTE_LETTERS, [F * 2 ** (i / 12) for i in range(len(NOTE_LETTERS))]))

def _build_note_map():
    note_letters = NOTE_LETTERS
    base_indices = list(map(lambda index: (index + 1) * 100, range(0, len(note_letters))))
    note_map = {}
    for i in range(7):
        numbered_letters = list(map(lambda note: note + str(i + 1), note_letters))
        indices = list(map(lambda index: index + 1200 * i, base_indices))
        note_map |= dict(zip(numbered_letters, indices))
    return note_map

def seeded_waveform(vol, duration, hz, seed, sr, sine_count=100):
    t = np.linspace(0, duration, int(sr * duration))
    wavelength = 1.0 / hz
    coefficients = np.zeros(sine_count)
    n_points = len(seed) - 1
    ms = (seed[1:] - seed[:-1]) * n_points / wavelength
    js = np.array([i for i in range(1, n_points + 1)])
    lowers = (js - 1) * wavelength / n_points
    uppers = js * wavelength / n_points
    wave = np.zeros_like(t)
    for k in range(1, sine_count + 1):
        alpha_k = 2 * np.pi * k / wavelength
        segments = -(ms * wavelength / (alpha_k * n_points) + seed[:-1] / alpha_k) * np.cos(alpha_k * uppers) + seed[:-1] / alpha_k * np.cos(alpha_k * lowers) + ms / alpha_k ** 2 * np.sin(alpha_k * uppers) - ms / alpha_k ** 2 * np.sin(alpha_k * lowers)
        coefficients[k - 1] = 2 / wavelength * np.sum(segments)
        wave += coefficients[k - 1] * np.sin(alpha_k * t)
    wave = vol * wave / np.max(np.abs(wave))
    return vol * np.array([wave, wave]).T

def resample(wave, cents:int, poly_length:int=4, sr:int=44100):
    new_sample_rate = 2 ** (-cents / 1200) * sr
    sample_delta = sr / new_sample_rate
    print("sample delta: {}".format(sample_delta))
    n = len(wave)
    print("old sample count: {}".format(n))
    new_sample_count = int(n / sample_delta)
    print('new sample count: {}'.format(new_sample_count))
    new_sample_index = 0
    resampled = []
    interp_x = np.array([i for i in range(poly_length)])
    window_start = 0
    window_len = poly_length
    y = wave[window_start:window_start+window_len]
    weights = np.polyfit(interp_x, y, 3)
    while new_sample_index < new_sample_count:
        if new_sample_index >= window_start + window_len:
            window_start += window_len
            y = wave[window_start:window_start+window_len]
            if len(y) < len(interp_x):
                if len(y) < 2:
                    break
                print("y len: {}".format(len(y)))
                print("x len: {}".format(len(interp_x)))
                print("Current new sample count: {}".format(new_sample_index))
                # Edge case for end of signal
                interp_x = np.array([i for i in range(len(y))])
            weights = np.polyfit(interp_x, y, 3)
        x = new_sample_index - window_start
        resampled.append(np.polyval(weights, x))
        new_sample_index += sample_delta
    new_wave = np.array(resampled)
    return new_wave

def sigmoid_env(A, t_i, y_i, t_f, y_f, duration, sr):
    t = np.linspace(0, duration, int(sr * duration))
    k = np.log((A / y_f - 1) / (A / y_i - 1)) / (t_i - t_f)
    s = 1 / k * np.log(A / y_i - 1) + t_i
    return A / (np.exp(-k * (t - s)) + 1)

def hyperbolic_tangent_env(A: float, t_f:float, percent_A:float, duration:float, sr:int) -> np.ndarray:
    """
    A: hyperbolic tangent scaling factor
    t_f: time value at which function become percent_A percent of A
    percent_A: some percentage of A reached at a point in time. Strictly less than 1
    duration: length of envelope
    sr: sample rate of envelope
    """
    if t_f <= 0:
        raise ValueError("t_f must be strictly greater than 0")
    if percent_A == A:
        raise ValueError(f"percent_A must be strictly less than 1")
    t = np.linspace(0, duration, int(duration * sr))
    # k: Hyperbolic tangent input scaling coefficient
    k = 1 / (2 * t_f) * np.log((1 + percent_A) / (1-percent_A))
    return A * np.tanh(k * t)

def random_smoothed_test(vol, duration, hz, sr, shift=0, n=15, M=13):
    assert duration > 0, "Duration must be greater than 0"
    wavelengths = 1.0 / hz
    t = np.linspace(0, duration, int(duration * sr))
    s = np.random.uniform(-1, 1, n)
    s = np.concatenate([s, [s[0]]])
    js = np.linspace(1, n, n)
    lowers = (js - 1) * wavelengths / n
    uppers = js * wavelength / n
    ms = n / wavelength * (s[1:] - s[:-1])
    alpha_ks = (2 * np.pi * np.linspace(1, M, M) / wavelength).reshape(-1, 1)
    segments = -1 / alpha_ks * (ms * wavelength / n + s[:-1]) * np.cos(alpha_ks * uppers) + s[:-1] / alpha_ks * np.cos(alpha_ks * lowers) + ms / alpha_ks ** 2 * np.sin(alpha_ks * uppers) - ms / alpha_ks ** 2 * np.sin(alpha_ks * lowers)
    coefficients = segments.sum(axis=1).reshape(-1, 1)
    wave = np.sum(coefficients * np.sin(alpha_ks * t), axis=0)
    wave *= vol
    wave /= np.max(np.abs(wave), axis=0)
    return np.array([wave, wave]).T


def random_smoothed(vol, duration, hz, sr, shift=0, n=5, M=17):
    assert duration > 0, "Duration must be greater than 0"
    wavelength = 1.0 / hz
    t = np.linspace(0, duration, int(duration*sr))
    s = np.random.uniform(-1, 1, n)
    s = np.concatenate((s, [s[0]]))
    js = np.array([i for i in range(1, n + 1)])
    lowers = (js - 1) * wavelength / n
    uppers = js * wavelength / n
    ms = n / wavelength * (s[1:] - s[:-1])
    coefficients = np.zeros(M)
    wave = np.zeros_like(t)
    for k in range(1, M + 1):
        alpha_k = 2 * np.pi * k / wavelength
        segments = -1 / alpha_k * (ms * wavelength / n + s[:-1]) * np.cos(alpha_k * uppers) + s[:-1] / alpha_k * np.cos(alpha_k * lowers) + ms / alpha_k ** 2 * np.sin(alpha_k * uppers) - ms / alpha_k ** 2 * np.sin(alpha_k * lowers)
        coefficients[k - 1] = 2 / wavelength * np.sum(segments)
        wave += coefficients[k - 1] * np.sin(alpha_k * t)
    #plt.plot(t, np.interp(t, np.linspace(0, 1.0/wavelength, len(s) - 1), s[:-1], period=1/wavelength))
    wave /= np.max(np.abs(wave))
    wave = np.array([wave, wave]).T
    return t, s, wave 


def _build_wavetables(sr=SAMPLE_RATE):
    note_map = _build_note_map()
    wavetable_path = os.path.join(AUDIO_DIR, "wavetable", "wavetable.json")
    if not os.path.isfile(wavetable_path):
        E = F * 2 ** (-1/12)
        frequency_spectrum = [E * 2 ** (i / 1200) for i in range(1200 * 7 + 200)]
        left_seeds, right_seeds = np.random.uniform(-1, 1, (2, 100, 15))
        indices = np.array([i for i in range(1200 * 7 + 200)]) % 100
        left_seeds = left_seeds[indices]
        right_seeds = right_seeds[indices]
        left_waves = [seeded_waveform(1, 1 / hz, hz, seed, sr, sine_count=13)[:, 0] for (seed, hz) in zip(left_seeds, frequency_spectrum)]
        right_waves = [seeded_waveform(1, 1 / hz, hz, seed, sr, sine_count=13)[:, 0] for (seed, hz) in zip(right_seeds, frequency_spectrum)]
        waves = [left_waves, right_waves]
    else:
        with open(wavetable_path, "r") as f:
            waves = json.load(f)
    return note_map, waves

if __name__ == "__main__":
    NOTE_MAP, WAVETABLES = _build_wavetables()

def integrate(f, a, b, n):
    dx = (b - a) / n
    total = 0
    x_i = a + dx / 2
    while x_i < b:
        total += f(x_i) * dx
        x_i += dx
    return total

def sine(vol, duration, hz, sr, shift=0):
    assert duration > 0, "Must have positive duration"
    assert sr > 0, "Must have positive sample rate"
    t = np.arange(0, duration, 1.0 / sr)
    return vol * np.sin(2 * np.pi * hz * t - 2 * np.pi * hz * shift)


# Sawtooth sine series
def sawtooth_fs(vol, duration, hz, sr, N=5):
    t = np.linspace(0, duration, int(duration * sr))
    ns = np.linspace(1, N, N).reshape(-1, 1)
    ns_matrix = np.outer(2 * np.pi * hz * ns, t)
    return vol * np.sum(-2 / np.pi * np.sin(ns_matrix) / ns, axis=0)

def square(vol, duration, hz, sr, shift=0):
    diff = sawtooth(vol, duration, hz, sr, shift=shift)
    geq = diff >= 0
    pos = 1.0 * geq
    neg = -1.0 * (~geq)
    return vol * (pos + neg)

def square_fs(vol, duration, hz, sr, N=5):
    t = np.linspace(0, duration, int(sr * duration))
    ns = np.linspace(1, N, N).reshape(-1, 1)
    ns_alt = 2 * ns - 1
    ns_matrix = np.outer(ns_alt, t)
    arg = 2 * np.pi * hz * ns_matrix
    wave = np.sum(4 * np.sin(arg) / np.pi / ns_alt, axis=0)
    wave /= np.max(np.abs(wave), axis=0)
    return vol * wave

def sawtooth(vol, duration, hz, sr, shift=0, form="positive"):
    assert duration > 0
    assert sr > 0, "Must have positive sample rate"
    t = np.arange(0, duration, 1.0 / sr)
    scaled = (t - shift) * hz
    rounded = np.round(scaled)
    diff = scaled - rounded
    if form == "negative":
        diff = -diff
    return 2 * vol * diff


def triangular(vol, duration, hz, sr, shift=0):
    assert duration > 0
    t = np.arange(0, duration, 1.0 / sr)
    return vol * 2 / np.pi * np.arcsin(np.sin(2 * np.pi * hz * t - 2 * np.pi * hz * shift))

def combined_random_waveforms(vol, duration, frequencies, sr, shift=0, n_points=30, sine_count=20):
    s = np.random.uniform(-1, 1, n_points)
    s = np.concatenate((s, [s[0]]))
    t = np.linspace(0, duration, int(sr * duration))
    wave = np.zeros_like(t)
    for freq in frequencies:
        wave += seeded_waveform(vol, duration, freq, s, sr, sine_count)
    return vol * wave / np.max(np.abs(wave))

def classic_envelope(audio, attack_ms, decay_ms, release_ms, sustain_fraction, sr):
    """
    TODO: Should the audio be required to be at least as long as the 
    attack_ms + decay_ms + release_ms
    """
    copied_audio = copy.copy(audio)
    attack_samples = int(sr * attack_ms)
    decay_samples = int(sr * decay_ms)
    release_samples = int(sr * release_ms)
    attack = np.linspace(0, 1, attack_samples).reshape(-1, 1)
    decay = np.linspace(1, sustain_fraction, decay_samples).reshape(-1, 1)
    release = np.linspace(sustain_fraction, 0, release_samples).reshape(-1, 1)
    copied_audio[:attack_samples] *= attack
    copied_audio[attack_samples:attack_samples+decay_samples] *= decay
    copied_audio[attack_samples+decay_samples:-release_samples] *= sustain_fraction
    copied_audio[-release_samples:] *= release
    return copied_audio

def gen_envelope(duration, amplitudes, sr):
    x = np.linspace(0, duration, int(duration * sr))
    xp = np.linspace(0, duration, len(amplitudes))
    return np.interp(x, xp, amplitudes).reshape(-1, 1)


def get_drift(vol, duration, notes: list[str], drifts_tenths: int=3, drift_granularity="hundredths"):
    if drift_granularity.lower() not in ["tenths", "hundredths"]:
        raise ValueError("Granularity must either be tenths or hundredths")
    if type(drifts_tenths) is not int:
        raise ValueError("Drift tenths must be an integer")
    if drift_granularity == "tenths":
        drift_factor = 10
    elif drift_granularity == "hundredths":
        drift_factor = 1
    note_map = _build_note_map()
    drifts_tenths = np.abs(drifts_tenths)
    n_samples = int(sr * duration)
    t_indices = np.linspace(0, n_samples, n_samples, dtype=int)
    left = np.zeros_like(t_indices)
    right = np.zeros_like(t_indices)
    wave_index_offsets = np.array([i for i in range(-drifts_tenths*drift_factor, drifts_tenths*drift_factor+1, drift_factor)])
    left_wavetable, right_wavetable = WAVETABLES
    wavelength_samples = list(map(len, left_wavetable))
    for note in notes:
        # Needs to be integer not float
        wave_base_index = note_map[note]
        for i, wave_index_offset in enumerate(wave_index_offsets):
            wave_current_index = wave_base_index + wave_index_offset
            wave_samples = wavelength_samples[wave_current_index]
            indices_modulo = t_indices % wave_samples
            left = left + left_wavetable[wave_current_index][indices_modulo]
            right = right + right_wavetable[wave_current_index][indices_modulo]
    wave = np.array([left, right]).T
    wave = threshold_filter(wave, threshold_amplitude_percentage=1)
    wave = classic_envelope(wave, .01, .02, .02, .8, sr)
    wave = vol * wave
    wave /= np.max(np.abs(wave), axis=0)
    return wave

def threshold_filter(audio, threshold_amplitude_percentage=.1):
    freq = np.fft.fft(audio, axis=0)
    amps = np.abs(freq)
    amps /= np.max(amps, axis=0)
    where_rows, where_cols = np.where(amps < threshold_amplitude_percentage * .01)
    freq[where_rows, where_cols] *= 0
    back = np.fft.ifft(freq, axis=0).real
    back /= np.max(np.abs(back), axis=0)
    return back
    
def delay(audio, samples_ahead, num_delays=4, decay_fraction=.8, decay_base=2, sr=44100):
    """
    Here's the idea: take a signal and add some number of delays some multiples
    of the `samples_ahead`. At each multiple exponentiate the decay base by the 
    index distance from the start.
    """
    indexes = [(i + 1) * int(samples_ahead) for i in range(num_delays)]
    n = len(audio)
    delayed = np.concatenate((audio, np.zeros((num_delays * int(samples_ahead), audio.shape[1]))))
    # For testing purposes try to filter out all frequencies between 20 Hz and 200 Hz
    for i, index in enumerate(indexes):
        decay_amount = decay_fraction / decay_base ** i
        delayed[index:index+n] += decay_amount * audio
    delayed /= np.max(np.abs(delayed))
    return delayed

def filtered_delay(
    audio: np.ndarray,
    samples_ahead: int,
    low_freq,
    high_freq,
    filter_fraction,
    num_delays: int=4,
    decay_fraction: float=0.8,
    decay_base: float=2,
    sr: int=44100
):
    indexes = [(i + 1) * int(samples_ahead) for i in range(num_delays)]
    n = len(audio)
    delayed = np.concatenate((audio, np.zeros((num_delays * int(samples_ahead), audio.shape[1]))))
    filtered = hard_filter(audio, low_freq, high_freq, filter_fraction, sr)
    for i, index in enumerate(indexes):
        decay_amount = decay_fraction / decay_base ** i
        delayed[index:index+n] += decay_amount * filtered
        filtered = hard_filter(filtered, low_freq, high_freq, filter_fraction, sr)
    delayed /= np.max(np.abs(delayed))
    return delayed


def hard_filter(audio: np.ndarray, lower_freq, upper_freq, fraction, sr, normalize=True):
    """
    Use FFT to adjust specified frequency amplitudes
    """
    frequencies = fft.fft(audio, axis=0)
    n = len(audio)
    lower_index = int(lower_freq * n / sr)
    upper_index = int(upper_freq * n / sr)
    freq_filter = np.ones((n // 2, 2))
    freq_filter[lower_index:upper_index+1] = fraction
    frequencies[:n/2] *= freq_filter
    frequencies[n//2:] *= freq_filter[::-1]
    filtered = fft.ifft(frequencies, axis=0).real
    if normalize:
        filtered /= np.max(np.abs(filtered))
    return filtered

def pitch_shift_freq(wave: np.ndarray, amount_cents: float) -> np.ndarray:
    """
    Just shift the first half of the FFT array
    """
    n = len(wave)
    freq = np.fft.fft(wave, axis=0)
    new_freq = np.zeros_like(freq)
    shift_factor = 2 ** (amount_cents / 1200)
    max_index = min(int(n / shift_factor), n)
    new_indices = np.array([round(i * shift_factor) for i in range(n // 2)])
    low_index = np.min(np.where(new_indices >= n // 2)[0])
    orig_indices = np.array([i for i in range(low_index)])
    new_freq[new_indices[:low_index]] = freq[orig_indices]
    new_wave = np.fft.ifft(new_freq, axis=0).real
    new_wave /= np.max(np.abs(new_wave), axis=0)
    return new_wave

def phaser(
    audio: np.ndarray,
    bandwidth: float,
    min_freq: float,
    max_freq: float,
    depth: float, 
    cycle_seconds: float,
    phase_window_seconds: int,
    sr: int
):
    if cycle_seconds <= 0:
        raise ValueError("Cycle duration must exceed 0")
    if min_freq > max_freq:
        min_freq, max_freq = max_freq, min_freq
    bandwidth = np.abs(bandwidth)
    if max_freq - min_freq < bandwidth:
        raise ValueError("Bandwidth exceeds frequency range")
    phased = copy.copy(audio)
    cycle_hz = 1.0 / cycle_seconds
    phase_window_samples = int(phase_window_seconds * sr)
    for index_start in range(0, len(audio) - phase_window_samples, phase_window_samples):
        low_freq = (max_freq - min_freq) * np.sin(2 * np.pi * cycle_hz * index_start / sr) + min_freq
        audio_chunk = phased[index_start:index_start + phase_window_samples] 
        phased[index_start:index_start + phase_window_samples] = hard_filter(
            audio_chunk,
            low_freq, 
            low_freq + bandwidth,
            depth,
            sr,
            normalize=False
        )
    return phased / np.max(np.abs(phased))



def fade_out(audio, seconds_from_end, sr):
    """
    
    """
    n = len(audio)
    if seconds_from_end * sr >= n:
        return audio
    start_index = int(n - seconds_from_end * sr)
    fade = np.concatenate((audio[:start_index], audio[start_index:] - audio[start_index:] * np.arange(0, 1, 1 / (n - start_index)).reshape(-1, 1)))
    return fade

def fade_in(audio, seconds_from_start, sr):
    n = len(audio)
    if seconds_from_start * sr >= n:
        return audio
    start_index = int(seconds_from_start * sr)
    faded_portion = audio[:start_index] * np.linspace(0, 1, start_index).reshape(-1, 1)
    fade = np.concatenate((faded_portion, audio[start_index:]))
    return fade


def sine_sweep(duration, low, high, sr):
    t_ir = np.linspace(0, duration, int(duration * sr))
    return np.sin(2 * np.pi * ((high - low) / duration * t_ir + low) * t_ir)

def eff_sweep(duration, low, high, sr):
    return play(sine_sweep(duration, low, high, sr), sr=sr)

def random_linear(vol, duration, hz, sr, shift=0, n=50):
    assert duration > 0, "Duration must be greater than 0"
    t = np.arange(0, duration, 1.0 / sr)
    s = np.random.uniform(-1, 1, n)
    s = np.concatenate((s, [s[0]]))
    wavelength = 1.0 / hz
    t_norm = t - wavelength * np.floor(t / wavelength)
    j = np.ceil(t_norm * n / wavelength).astype(int)
    y = n / wavelength * (s[j] - s[j - 1]) * (t_norm - (j - 1) * wavelength / n) + s[j - 1]
    y_norm = y / np.max(np.abs(y))
    return s, vol * np.array([y_norm, y_norm]).T

def combined_random_linear(vol, duration, freqs, sr, shift=0, n=17):
    assert duration > 0, "Duration must be greater than 0"
    seeds = []
    wave = np.zeros((int(sr * duration), 2))
    for freq in freqs:
        s, w = random_linear(vol, duration, freq, sr, shift, n)
        seeds.append(s)
        wave += w
    return seeds, wave / np.max(np.abs(wave), axis=0)

def sine_coefficients_of_points(points_seed: np.ndarray, hz: float, sine_count=17):
    n = len(points_seed)
    wavelength = 1.0 / hz
    js = np.array([i for i in range(1, n + 1)])
    lowers = (js - 1) * wavelength / n
    uppers = js * wavelength / n
    s = np.concatenate((points_seed, [points_seed[0]]))
    ms = n / wavelength * (s[1:] - s[:-1])
    coefficients = np.zeros(sine_count)
    for k in range(1, sine_count + 1):
        alpha_k = 2 * np.pi * k / wavelength
        segments = -1 / alpha_k * (ms * wavelength / n + s[:-1]) * np.cos(alpha_k * uppers) + s[:-1] / alpha_k * np.cos(alpha_k * lowers) + ms / alpha_k ** 2 * np.sin(alpha_k * uppers) - ms / alpha_k ** 2 * np.sin(alpha_k * lowers)
        coefficients[k - 1] = 2 / wavelength * np.sum(segments)
    return coefficients


def random_smoothed_with_time(vol, tim, hz, sr, shift=0, n=5, M=17):
    assert len(tim) > 0, "Duration must be greater than zero"
    wavelength = 1.0 / hz
    seed = np.random.uniform(-1, 1, n)

def random_smoothed_with_bend(vol, duration, hz, sr, shift=0, n=13, M=17, max_bend=.3, max_osc=10):
    t = np.linspace(0, duration, int(duration * sr))
    wavelength = 1.0 / hz
    seed = np.random.uniform(-1, 1, n)
    coefficients = sine_coefficients_of_points(seed, hz, sine_count=M)
    wave = np.zeros_like(t)
    bend_depths = np.random.uniform(0, max_bend, M)
    bend_osc = np.random.uniform(0, max_osc, M)
    for k in range(1, M + 1):
        wave += coefficients[k - 1] * sine_bend_centered_with_time(t, k / wavelength, bend_depths[k-1], bend_osc[k-1], sr)
    wave /= np.max(np.abs(wave))
    wave = np.array([wave, wave]).T
    return t, seed, wave

def random_smoothed_with_bend_with_time(vol, tim, hz, sr, shift=0, n=13, M=17, max_bend=.3, max_osc=10):
    wavelength = 1.0 / hz
    seed = np.random.uniform(-1, 1, n)
    coefficients = sine_coefficients_of_points(seed, hz, sine_count=M)
    wave = np.zeros_like(tim)
    bend_depths = np.random.uniform(0, max_bend, M)
    bend_osc = np.random.uniform(0, max_osc, M)
    for k in range(1, M + 1):
        wave += coefficients[k - 1] * sine_bend_centered_with_time(tim, k / wavelength, bend_depths[k - 1], bend_osc[k - 1], sr)

    wave /= np.max(np.abs(wave))
    return tim, seed, wave

def random_smoothed_with_freq_env_with_time(vol, t, hz, freq_env, sr, n=13, M=15):
    """
    Assume the frequency envelope is unscaled
    """
    seed = np.random.uniform(-1, 1, n)
    coefficients = sine_coefficients_of_points(seed, hz, sine_count=M)
    wave = np.zeros_like(t)
    for k in range(1, M + 1):
        freq = k * hz
        theta_prime = 2 * np.pi * freq * 2 ** (freq_env / 12)
        theta = np.cumsum(theta_prime) / sr
        wave += coefficients[k - 1] * np.sin(theta)
    wave /= np.max(np.abs(wave))
    return t, seed, wave

    

def random_smoothed_with_flutter(vol, duration, hz, sr, shift=0, n=13, M=17, max_flutter_freq=10):
    t = np.linspace(0, duration, int(duration * sr))
    wavelength = 1.0 / hz
    seed = np.random.uniform(-1, 1, n)
    coefficients = sine_coefficients_of_points(seed, hz, sine_count=M)
    wave = np.zeros_like(t)
    flutter_freqs = np.random.uniform(0, max_flutter_freq, M)
    for k in range(1, M + 1):
        wave += np.sin(2 * np.pi * flutter_freqs[k - 1] * t) * coefficients[k - 1] * np.sin(2 * np.pi * k / wavelength * t)
    wave /= np.max(np.abs(wave), axis=0)
    return t, seed, wave

def random_smoothed_with_bend_and_flutter_with_time(vol, tim, hz, sr, shift=0, n=13, M=17, max_bend=.3, max_bend_osc=5, max_flutter_freq=10):
    wavelength = 1.0 / hz
    seed = np.random.uniform(-1, 1, n)
    coefficients = sine_coefficients_of_points(seed, hz, sine_count=M)
    wave = np.zeros_like(tim)
    bend_depths = np.random.uniform(0, max_bend, M)
    bend_osc = np.random.uniform(0, max_bend_osc, M)
    flutter_freqs = np.random.uniform(0, max_flutter_freq, M)
    for k in range(1, M + 1):
        wave += coefficients[k - 1] * np.sin(2 * np.pi * flutter_freqs[k - 1] * tim) * sine_bend_centered_with_time(tim, k / wavelength, bend_depths[k-1], bend_osc[k-1], sr)
    wave /= np.max(np.abs(wave), axis=0)
    return tim, seed, wave


def sine_bend_centered(duration, freq, bend_dist, osc_freq, sr):
    """
    @param duration length of sample in seconds
    @param freq is just the ordinary frequency of the sine wave (in hz)
    @param bend_dist maximum pitch bend in half steps
    @param osc_freq the maximum message ordinary frequency (pitch bend frequency)
    @param sr the sample rate
    """
    t = np.linspace(0, duration, int(duration * sr))
    freq_carrier = 2 * np.pi * freq
    freq_message = 2 * np.pi * osc_freq
    p = 2 * np.pi * freq * (2 ** (bend_dist / 12) - 1)
    theta = lambda tau: freq_carrier * tau - p / freq_message * np.cos(freq_message * tau)
    return np.sin(theta(t))

def sine_bend_centered_with_time(tim, freq, bend_dist, osc_freq, sr):
    start = time.time()
    freq_carrier = 2 * np.pi * freq
    freq_message = 2 * np.pi * osc_freq
    p = 2 * np.pi * freq * (2 ** (bend_dist / 12) - 1)
    theta = lambda tau: freq_carrier * tau - p / freq_message * np.cos(freq_message * tau)
    result = np.sin(theta(tim))
    print("sin(theta(t)): {}".format(time.time() - start))
    return result

def tri_bend_centered_with_time(tim, freq, bend_dist, osc_freq, sr):
    return np.arcsin(sine_bend_centered_with_time(tim, freq, bend_dist, osc_freq, sr))

def combine_random_smoothed(vol, duration, freqs, sr, shift=0, n=17, M=15):
    assert duration > 0, "Duration must be greater than 0"
    wave = np.zeros((int(sr * duration), 2))
    seeds = []
    for freq in freqs:
        t, s, w = random_smoothed(vol, duration, freq, sr, shift=shift, n=n, M=M)
        seeds.append(s)
        wave += w
    return t, seeds, wave / np.max(np.abs(wave))

def combine_bendy(vol, duration, freqs, sr, shift=0, n=13, M=17, max_bend=.3, max_osc=5):

    wave = np.zeros((int(sr * duration), 2))
    seeds = []
    for freq in freqs:
        t, s, w = random_smoothed_with_bend(vol, duration, freq, sr, shift=shift, n=n, M=M, max_bend=max_bend, max_osc=max_osc)
        seeds.append(s)
        wave += w
    return t, seeds, wave / np.max(np.abs(wave))

def combine_bendy_lr(vol, duration, freqs, sr, shift=0, n=13, M=15, max_bend=.3, max_osc=5):
    t = np.linspace(0, duration, int(sr * duration))
    left = np.zeros(int(sr * duration))
    right = np.zeros(int(sr * duration))
    seeds = []
    compute_durations = []
    for freq in freqs:
        start = time.time()
        t, s_l, w_l = random_smoothed_with_bend_with_time(vol, t, freq, sr, shift=shift, n=n, M=M, max_bend=max_bend, max_osc=max_osc)
        compute_durations.append(time.time() - start)
        start = time.time()
        t, s_r, w_r = random_smoothed_with_bend_with_time(vol, t, freq, sr, shift=shift, n=n, M=M, max_bend=max_bend, max_osc=max_osc)
        compute_durations.append(time.time() - start)
        seeds.extend([s_l, s_r])
        left += w_l
        right += w_r
    left /= np.max(np.abs(left), axis=0)
    right /= np.max(np.abs(right), axis=0)
    print(compute_durations)
    print(np.mean(compute_durations))
    print(np.std(compute_durations))
    return t, seeds, np.array([left, right]).T

def combine_freq_env_lr(vol, duration, freqs, freq_env, sr, shift=0, n=13, M=15):
    t = np.linspace(0, duration, int(sr * duration))
    left = np.zeros(int(sr * duration))
    right = np.zeros(int(sr * duration))
    seeds = []
    for freq in freqs:
        t, s_l, w_l = random_smoothed_with_freq_env_with_time(vol, t, freq, freq_env, sr, n=n, M=M)
        t, s_r, w_r = random_smoothed_with_freq_env_with_time(vol, t, freq, freq_env, sr, n=n, M=M)
        seeds.extend([s_l, s_r])
        left += w_l
        right += w_r
    left /= np.max(np.abs(left))
    right /= np.max(np.abs(right))
    return t, seeds, np.array([left, right]).T

def combine_bendy_fluttered_lr(vol, duration, freqs, sr, shift=0, n=13, M=15, max_bend=.3, max_osc=5, max_flutter_freq=10):
    t = np.linspace(0, duration, int(sr * duration))
    left = np.zeros(int(sr * duration))
    right = np.zeros(int(sr * duration))
    seeds = []
    flutter_freqs = np.random.uniform(0, max_flutter_freq, M)
    for freq in freqs:
        t, s_l, w_l = random_smoothed_with_bend_and_flutter_with_time(vol, t, freq, sr, shift=shift, n=n, M=M, max_bend=max_bend, max_bend_osc=max_osc, max_flutter_freq=max_flutter_freq)
        t, s_r, w_r = random_smoothed_with_bend_and_flutter_with_time(vol, t, freq, sr, shift=shift, n=n, M=M, max_bend=max_bend, max_bend_osc=max_osc, max_flutter_freq=max_flutter_freq)
        seeds.extend([s_l, s_r])
        left += w_l
        right += w_r
    left /= np.max(np.abs(left), axis=0)
    right /= np.max(np.abs(right), axis=0)
    return t, seeds, np.array([left, right]).T
        

def combine_random_smoothed_lr(vol, duration, freqs, sr, shift=0, n=17, M=15):
    """
    This one makes semi-ethereal organ sounds, depending on the drift width chosen
    """
    left = np.zeros((int(sr * duration)))
    right = np.zeros((int(sr * duration)))
    seeds = []
    for freq in freqs:
        t, s_l, w_l = random_smoothed(vol, duration, freq, sr, shift=shift, n=n, M=M)
        t, s_r, w_r = random_smoothed(vol, duration, freq, sr, shift=shift, n=n, M=M)
        seeds.extend([s_l, s_r])
        left += w_l[:, 0]
        right += w_r[:, 0]
    left /= np.max(np.abs(left))
    right /= np.max(np.abs(right))
    return t, seeds, np.array([left, right]).T

def combine_flutter(vol, duration, freqs, sr, shift=0, n=17, M=15, max_flutter_freq=10):
    left = np.zeros((int(sr * duration)))
    right = np.zeros_like(left)
    seeds = []
    for freq in freqs:
        t, s_l, w_l = random_smoothed_with_flutter(vol, duration, freq, sr, shift=shift, n=n, M=M, max_flutter_freq=max_flutter_freq)
        t, s_r, w_r = random_smoothed_with_flutter(vol, duration, freq, sr, shift=shift, n=n, M=M, max_flutter_freq=max_flutter_freq)
        seeds.extend([s_l, s_r])
        left += w_l
        right += w_r
    left /= np.max(np.abs(left))
    right /= np.max(np.abs(right))
    return t, seeds, np.array([left, right]).T

def generate_frequency_drift(freq_kernel: list[float], drift_max: float=.1, drifts: int=3):
    drift = np.abs(drift_max)
    drift_exponents = [drift / (drifts+1) * i / 12 for i in range(-drifts, drifts+1)]
    frequencies = []
    for freq in freq_kernel:
        frequencies.extend(map(lambda e: freq * 2 ** e, drift_exponents))
    return frequencies

def compress(audio, sr, method="peaks"):
    """
    """
    if method == "peaks":
        forward = audio[1:] > audio[:-1]
        backward = audio[:-1] > audio[1:]
        row_maxima, col_maxima = np.where((backward[1:] * forward[:-1]) == True)
        row_maxima += 1
        row_minima, col_minima = np.where((backward[:-1] * forward[1:]) == True)
        row_minima += 1

        left_row_minima = np.array([row for row, col in zip(row_minima, col_minima) if col == 0])
        left_row_maxima = np.array([row for row, col in zip(row_maxima, col_maxima) if col == 0])

        right_row_minima = np.array([row for row, col in zip(row_minima, col_minima) if col == 1])
        right_row_maxima = np.array([row for row, col in zip(row_maxima, col_maxima) if col == 1])
        fig, ax = plt.subplots(figsize=(16, 9))

def get_wavetable(filename) -> dict|None:
    with open(filename, "r") as f:
        data = json.load(f)
        return data
        

def all_pass(
    audio: np.ndarray
    , delay_seconds
):
    pass


def echo(signal: np.ndarray, dist: float, decay, sr: int):
    # Dist in meters
    if len(signal.shape) > 1 and signal.shape[1] > 2:
        # Permit panned sounds
        raise ValueError("Expected an ndarray with shape of (n,), (n, 1), or (n, 2)")
    v_sound = 343
    dt = 2 * dist / v_sound
    sample_window = int(sr * dt)
    extension = np.concatenate((signal, np.zeros(len(signal))))
    
    index = 0
    start = time.time()
    while index < len(signal):
        extension[index+sample_window:index+2*sample_window] += extension[index:index+sample_window] * decay
        index += 1
        print("%.2f" % (time.time() - start), end="\r")
        sys.stdout.flush()
    print()
    return extension

FORM_TO_STR_MAP = {
    sine: "sine",
    square: "square",
    sawtooth: "sawtooth",
    triangular: "triangular",
    seeded_waveform: "seeded",
}

STRING_TO_FORM_MAP = dict((FORM_TO_STR_MAP[form], form) for form in FORM_TO_STR_MAP)

def play(waveform, loop=False, sr=44100):
    chars = [char for char in string.ascii_letters]
    name = "".join(np.random.choice(chars, 8, replace=True)) + ".wav"
    path = os.path.join(AUDIO_DIR, "tmp", name)
    sf.write(path, waveform, sr)
    url = QUrl.fromLocalFile(path)
    effect = QSoundEffect()
    effect.setSource(url)
    if loop:
        effect.setLoopCount(QSoundEffect.Loop.Infinite.value)
    else:
        effect.setLoopCount(0)
    return effect

def segment_partition(low, high, parts: int):
    if low > high:
        low, high = high, low
    diff = high - low
    frac = diff / parts
    return [[low + frac*i , low+frac*(i+1)] for i in range(parts+1)]

def dry_wet_reverb(audio, channel_count, channel_bandwidth, diffusion_steps, wet, sr):
    if wet < 0 or wet > 1:
        raise ValueError("Wet factor should be between 0 and 1, inclusive.")
    reverbed = reverb(audio, channel_count, channel_bandwidth, diffusion_steps, sr)
    combined = np.concatenate((audio, np.zeros((len(reverbed) - len(audio), 2)))) * (1 - wet) + reverbed * wet
    combined /= np.max(np.abs(combined), axis=0)
    return combined

def reverb(audio, channel_count, channel_bandwidth: list[2], diffusion_steps, sr):
    """
    First attempt at reverb. Referencing the doc at 
    https://signalsmith-audio.co.uk/writing/2021/lets-write-a-reverb/
    """
    # Multi-channel feedback loop

    sub_bandwidths = segment_partition(*channel_bandwidth, diffusion_steps)
    combination = np.copy(audio)
    for i in range(diffusion_steps):
        channel_durations = np.random.uniform(*sub_bandwidths[i], channel_count)
        delayed_channels = [delay(combination, int(duration*sr), decay_base=1.5) for duration in channel_durations]
        max_len = max([len(channel) for channel in delayed_channels])
        delayed_channels = np.array(list(map(lambda l: np.concatenate((l, np.zeros((max_len - len(l), 2)))), delayed_channels)))
        combination = diffuse(delayed_channels, channel_count)
    combination /= np.max(np.abs(combination), axis=0)
    return combination

def diffuse(audio, channel_count):
    """
    """
    H = np.random.random((channel_count, channel_count))
    H /= H.sum(axis=1).reshape(-1, 1)
    #H = hadamard_matrix(channel_count)
    shuffled = shuffle_polarity(audio, channel_count)
    left_shuffled = shuffled.T[0].T
    right_shuffled = shuffled.T[1].T
    left_combination = np.sum(H.dot(left_shuffled), axis=0)
    right_combination = np.sum(H.dot(right_shuffled), axis=0)
    combination = np.array([left_combination, right_combination]).T
    return combination

def shuffle_polarity(audio: np.ndarray, channel_count) -> np.ndarray:
    left_channels = audio.T[0].T
    right_channels = audio.T[1].T
    indices = [i for i in range(len(audio))]
    np.random.shuffle(indices)
    polarities = np.random.choice([-1, 1], replace=True, size=channel_count)
    left_shuffled = left_channels[indices] * polarities.reshape(-1, 1)
    right_shuffled = right_channels[indices] * polarities.reshape(-1, 1)
    return np.array([left_shuffled.T, right_shuffled.T]).T

def hadamard_matrix(dim: int) -> np.ndarray:
    """
    """
    assert type(dim) is int, "Hadamard matrix dimension must be an integer"
    log2 = np.log2(dim)
    assert dim >= 1 and log2 - int(log2) < .0000000000001, "Hadamard matrix dimension must be of the form 2^k"
    H = np.array([[1]])
    curr_dim = 1
    while curr_dim < dim:
        H = np.concatenate((
                np.concatenate((H, H), axis=0)
              , np.concatenate((H, -H), axis=0)
            ), axis=1)
        curr_dim *= 2
    return H

def make_uniform_channels(channels) -> np.ndarray:
    max_len = max(map(lambda c: len(c), channels))
    new_channels = np.array(list(map(lambda c: np.concatenate((c, np.zeros((max_len - len(c), 2))), axis=0), channels)))
    return new_channels

class SoundPlayer:
    """
    A hacky solution to play sounds in PyQt5

    Parameters:
    -----------
    waveform : np.ndarray 
        This is simply the actual waveform that will be played in audio

    sr : int
        The given sample rate of the waveform.

    loopCount : QSoundEffect int
        If 0 or 1, it will play once. The default behavior is to play 
        infinitely, as specified by QSoundEffect.Loop.Infinite.

    Attributes:
    -----------
    chars : list 
        Randomly sample from the list of letters to form random name of wav file

    base_dir : str
        audio/tmp directory to store randomly named tracks

    
    """
    def __init__(self, waveform, sr=None, loopCount=QSoundEffect.Loop.Infinite.value):
        self.chars = list(string.ascii_letters)
        self.base_dir = os.path.join(AUDIO_DIR, "tmp")
        self.updateWaveform(waveform, sr=sr)

        self.loopCount = loopCount
        self.started = False
    def play(self) -> None:
        if not self.started:
            self.name = "".join(np.random.choice(self.chars, 8, replace=True)) + ".wav"
            self.path = os.path.join(self.base_dir, self.name)
            sf.write(self.path, self.array, self.sr)
            url = QUrl.fromLocalFile(self.path)
            self.effect = QSoundEffect()
            self.effect.setSource(url)
            self.effect.setLoopCount(self.loopCount)
            self.effect.play()
            self.started = True
    def stop(self) -> None:
        if self.started:
            self.effect.stop()
            self.started = False
        for file in glob.glob(self.base_dir + "/*"):
            os.remove(file)
    def updateWaveform(self, waveform, sr=None):
        if isinstance(waveform, Waveform):
            # save the inner array
            self.waveform = waveform
            self.array = waveform.getArray()
            self.sr = waveform.sr
        elif isinstance(waveform, np.ndarray) or isinstance(waveform, list):
            self.waveform = None
            self.array = waveform
            if sr is None:
                raise ValueError("Must supply sr or Waveform")
            self.sr = sr
        else:
            raise ValueError("'waveform' parameter should be an 1d-array, list, or Waveform")



class Waveform:
    """
    All of amp, hz, and shift can be functions. The form parameter is one of
    the four available waveforms (sine, sawtooth, square, and triangle)

    Parameters:
    -----------

    amp : float
        
    """
    def __init__(self, amp, duration, hz, shift, form=sine, sr=44100):
        self.index = 0

        if amp < 0: raise ValueError("invalid negative amplitude")
        self.amp = amp
        self.hz = hz
        self.shift = shift
        self.form = form
        self.duration = duration
        self.sr = sr
        self.buildArray()

    def buildArray(self):
        self.array = self.form(self.amp, self.duration, self.hz, self.sr, shift=self.shift)

    def getParams(self):
        return {
            "amp": self.amp,
            "duration": self.duration,
            "hz": self.hz,
            "sr": self.sr,
            "shift": self.shift,
            "form": FORM_TO_STR_MAP[self.form]
        }

    def getArray(self):
        return self.array

    def __iter__(self):
        for element in self.array:
            yield element

    def __next__(self):
        if self.index < len(self.array):
            self.index += 1
            return self.array(self.index - 1)
        raise StopIteration
    def __getitem__(self, i):
        assert type(i) is int
        return self.array[i]
    def __len__(self):
        return len(self.array)
    def __add__(self, item):
        return self.array + item
    def __radd__(self, item):
        return item + self.array
    def __sub__(self, item):
        return self.array - item
    def __rsub__(self, item):
        return item - self.array
    def __mul__(self, item):
        return self.array * item
    def __rmul__(self, item):
        return item * self.array
    def __div__(self, item):
        return self.array / item
    def __rdiv__(self, item):
        return item / self.array
    def __repr__(self):
        return repr(self.array)


def harmonics(amplitude, duration, tonal_center, dumb, idk, key):
    note_names = np.array(list(NOTE_MAP.keys()))
    offsets = np.array(key)
    note_index = 9
    letters = note_names[note_index + offsets]
    return get_drift(.8 * amplitude, duration, letters, 25)

        
def hemorrhage_test():
    bpm = 155
    bps = 155/60
    root = "D1"
    sample_rate = 44100
    n=1

    sequence = [
        harmonics(1, 1 / (bpm / 60), root, sample_rate, n, key=[4, 7])
        , harmonics(0, 2 / (bpm / 60), root, sample_rate, n, key=[0])
        , harmonics(1, .5 / (bpm / 60), root, sample_rate, n, key=[ 4, 7])
        , harmonics(1, .5 / (bpm / 60), root, sample_rate, n, key=[ 5, 8])
        , harmonics(1, 1 / (bpm / 60), root, sample_rate, n, key=[ 4, 7])
        , harmonics(0, 1 / (bpm / 60), root, sample_rate, n, key=[0])
        , harmonics(1, 1 / (bpm / 60), root, sample_rate, n, key=[ 5, 8])
        , harmonics(0, 1 / (bpm / 60), root, sample_rate, n, key=[0])
        , harmonics(1, 1 / (bpm / 60), root, sample_rate, n, key=[ 7, 11])
        , harmonics(0, 2 / (bpm / 60), root, sample_rate, n, key=[0])
        , harmonics(1, .5 / (bpm / 60), root, sample_rate, n, key=[ 7, 11])
        , harmonics(1, .5 / (bpm / 60), root, sample_rate, n, key=[ 8, 12])
        , harmonics(1, 1 / (bpm / 60), root, sample_rate, n, key=[ 7, 11])
        , harmonics(0, 1 / (bpm / 60), root, sample_rate, n, key=[0])
        , harmonics(1, 1 / (bpm / 60), root, sample_rate, n, key=[ 5, 8])
        , harmonics(0, 1 / (bpm / 60), root, sample_rate, n, key=[0])
        , harmonics(1, 1 / (bpm / 60), root, sample_rate, n, key=[ -2, 1])
        , harmonics(0, 2 / (bpm / 60), root, sample_rate, n, key=[0])
        , harmonics(1, 1 / (bpm / 60), root, sample_rate, n, key=[ -4, 0])
        , harmonics(1, 1 / (bpm / 60), root, sample_rate, n, key=[ 0, 4])
        , harmonics(0, 1 / (bpm / 60), root, sample_rate, n, key=[0])
        , harmonics(1, 1 / (bpm / 60), root, sample_rate, n, key=[ -1, 3])
        , harmonics(0, 1 / (bpm / 60), root, sample_rate, n, key=[0])

        , get_drift(1, 3/bps, ["D1", "F#2", "A2"], 20)
        , get_drift(1, .5/bps, ["D1", "F#2", "A2"], 30)
        , get_drift(1, .5/bps, ["D1", "G2", "A2"], 30)
        , get_drift(1, 2/bps, ["D1", "F#2", "A2"], 20)
        , get_drift(1, 2/bps, ["D1", "G2", "A2"], 20)
        , get_drift(1, 3/bps, ["C1", "A2"], 20)
        , get_drift(1, .5/bps, ["C1", "A2"], 30)
        , get_drift(1, .5/bps, ["C1", "A#2"], 30)
        , get_drift(1, 2/bps, ["C1", "A2"], 20)
        , get_drift(1, 2/bps, ["C1", "G2"], 20)
        , get_drift(1, 3/bps, ["D#1", "C2"], 20)
        , get_drift(1, 1/bps, ["D#1", "A#2"], 20)
        , get_drift(1, 2/bps, ["F2", "A#2", "D2"], 20)
        , get_drift(1, 2/bps, ["F#1", "C#1", "F#2", "C#2"], 20)

        , get_drift(1, 3/bps, ["D1", "F#2", "A2"], 20)
        , get_drift(1, .5/bps, ["D1", "F#2", "A2"], 30)
        , get_drift(1, .5/bps, ["D1", "G2", "A2"], 30)
        , get_drift(1, 2/bps, ["D1", "F#2", "A2"], 20)
        , get_drift(1, 2/bps, ["D1", "G2", "A2"], 20)
        , get_drift(1, 3/bps, ["C1", "A2"], 20)
        , get_drift(1, .5/bps, ["C1", "A2"], 30)
        , get_drift(1, .5/bps, ["C1", "A#2"], 30)
        , get_drift(1, 2/bps, ["C1", "A2"], 20)
        , get_drift(1, 2/bps, ["C1", "G2"], 20)
        , get_drift(1, 3/bps, ["D#1", "C2"], 20)
        , get_drift(1, 1/bps, ["D#1", "A#2"], 20)
        , get_drift(1, 2/bps, ["F2", "A#2", "D2"], 20)
        , get_drift(1, 2/bps, ["F#1", "C#1", "F#2", "C#2"], 20)

        , harmonics(1, 4/bps, root, sample_rate, n, key=[1, 8, 17])
        , harmonics(1, 4/bps, root, sample_rate, n, key=[0, 7, 16])
        , harmonics(1, 3/bps, root, sample_rate, n, key=[1, 8, 13])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[0, 7, 12])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, 4/bps, root, sample_rate, n, key=[0, 7, 12])
        , harmonics(1, 4/bps, root, sample_rate, n, key=[1, 8, 17])
        , harmonics(1, 4/bps, root, sample_rate, n, key=[5, 12, 20])
        , harmonics(1, 3/bps, root, sample_rate, n, key=[3, 10, 19])
        , harmonics(1, .25/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .25/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .25/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .25/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, 1/bps, root, sample_rate, n, key=[-2, 5, 12])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 12])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 13])
        , harmonics(1, 1/bps, root, sample_rate, n, key=[-2, 5, 12])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 12])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 13])
        , harmonics(1, 1/bps, root, sample_rate, n, key=[-2, 5, 12])
        , harmonics(1, 1/bps, root, sample_rate, n, key=[-2, 5, 10, 17])
        , harmonics(1, 3/bps, root, sample_rate, n, key=[0, 7, 12, 16])
        , harmonics(1, 1/bps, root, sample_rate, n, key=[1, 8, 13])
        , harmonics(1, 3/bps, root, sample_rate, n, key=[0, 7, 12])
        , harmonics(1, .25/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .25/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .25/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .25/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, 1/bps, root, sample_rate, n, key=[-2, 5, 12])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 12])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 13])
        , harmonics(1, 1/bps, root, sample_rate, n, key=[-2, 5, 12])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 10])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 12])
        , harmonics(1, .5/bps, root, sample_rate, n, key=[-2, 5, 13])
        , harmonics(1, 1/bps, root, sample_rate, n, key=[-2, 5, 12])
        , harmonics(1, 1/bps, root, sample_rate, n, key=[-2, 5, 10, 17])
        , harmonics(1, 3/bps, root, sample_rate, n, key=[0, 7, 12, 16])
        , harmonics(1, 1/bps, root, sample_rate, n, key=[1, 8, 13])
        , harmonics(1, 3/bps, root, sample_rate, n, key=[0, 7, 12])
    ]
    wave = np.concatenate(sequence, axis=0)
    return wave



if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(16, 9))
    sr = 44100
    #bergunde, _ = sf.read("audio/acoustic_guitar.mp3")
    #tap, _ = sf.read("audio/tap.wav")
    #berg = bergunde[:2*sr]
    #berg = bergunde
    #berg = np.array([berg, berg]).T
    #orig = play(berg)
    #reverbed = reverb(bergunde, 8, [.004, .07], 4, sr)
    #effect = play(reverbed)
    #effect.play()

