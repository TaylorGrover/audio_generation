from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QSoundEffect
import glob
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


def square(vol, duration, hz, sr, shift=0):
    diff = sawtooth(vol, duration, hz, sr, shift=shift)
    geq = diff >= 0
    pos = 1.0 * geq
    neg = -1.0 * (~geq)
    return vol * (pos + neg)


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
    return vol * wave



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


def hard_filter(audio: np.ndarray, lower_freq, upper_freq, fraction, sr):
    """
    Use FFT to adjust specified frequency amplitudes
    """
    frequencies = fft.fft(audio, axis=0)
    n = len(audio)
    lower_index = int(lower_freq * n / sr)
    upper_index = int(upper_freq * n / sr)
    freq_filter = np.ones((n // 2, 2))
    freq_filter[lower_index:upper_index+1] = fraction
    frequencies[:n//2] *= freq_filter
    frequencies[n//2:] *= freq_filter[::-1]
    filtered = fft.ifft(frequencies, axis=0).real
    filtered /= np.max(np.abs(filtered))
    return filtered


def fade_out(audio, seconds_from_end, sr):
    """
    
    """
    n = len(audio)
    if seconds_from_end * sr >= n:
        return audio
    start_index = int(n - seconds_from_end * sr)
    fade = np.concatenate((audio[:start_index], audio[start_index:] - audio[start_index:] * np.arange(0, 1, 1 / (n - start_index)).reshape(-1, 1)))
    return fade


def sine_sweep(duration, low, high, sr):
    t_ir = np.linspace(0, duration, int(duration * sr))
    return np.sin(2 * np.pi * ((high - low) / duration * t_ir + low) * t_ir)

def eff_sweep(duration, low, high, sr):
    return play(sine_sweep(duration, low, high, sr), sr=sr)

def random(vol, duration, hz, sr, shift=0, n=50):
    assert duration > 0, "Duration must be greater than 0"
    t = np.arange(0, duration, 1.0 / sr)
    s = np.random.uniform(-1, 1, n)
    s = np.concatenate((s, [s[0]]))
    wavelength = 1.0 / hz
    t_norm = t - wavelength * np.floor(t / wavelength)
    j = np.ceil(t_norm * n / wavelength).astype(int)
    y = n / wavelength * (s[j] - s[j - 1]) * (t_norm - (j - 1) * wavelength / n) + s[j - 1]
    return vol * y / np.max(np.abs(y))


def random_smoothed(vol, duration, hz, sr, shift=0, n=5, M=100):
    assert duration > 0, "Duration must be greater than 0"
    wavelength = 1.0 / hz
    t = np.arange(0, duration, 1.0 / sr)
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
    return t, s, wave 


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

def play(waveform, sr=44100):
    chars = [char for char in string.ascii_letters]
    name = "".join(np.random.choice(chars, 8, replace=True)) + ".wav"
    path = os.path.join(AUDIO_DIR, "tmp", name)
    sf.write(path, waveform, sr)
    url = QUrl.fromLocalFile(path)
    effect = QSoundEffect()
    effect.setSource(url)
    print(QSoundEffect.Loop.Infinite.value)
    effect.setLoopCount(0)
    return effect


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

        


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(16, 9))
    sr = 44100
    t = np.arange(0, 5, 1.0 / sr)
