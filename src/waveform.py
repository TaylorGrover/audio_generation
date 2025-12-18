from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QSoundEffect
import glob
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import os
import shutil
import soundfile as sf
import string
import sys
import time


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
    return 2 / np.pi * np.arcsin(np.sin(2 * np.pi * hz * t - 2 * np.pi * hz * shift))


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
}

STRING_TO_FORM_MAP = dict((FORM_TO_STR_MAP[form], form) for form in FORM_TO_STR_MAP)

def play(waveform, sr=44100):
    chars = [char for char in string.ascii_letters]
    name = "".join(np.random.choice(chars, 8, replace=True)) + ".wav"
    path = os.path.join(AUDIO_DIR, name)
    sf.write(path, waveform, sr)
    url = QUrl.fromLocalFile(path)
    effect = QSoundEffect()
    effect.setSource(url)
    print(QSoundEffect.Loop.Infinite.value)
    effect.setLoopCount(QSoundEffect.Loop.Infinite.value)
    effect.play()
    os.remove(path)
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
        print(dir(self.loopCount))
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
    sr = 44100
    t = np.arange(0, 5, 1.0 / sr)
    y_smooth = random_smoothed(1, 5, 440, sr, 0, 50, 100)
    y_smooth_low = random_smoothed(1, 5, 220, sr, 0, 11, 10)
    y = y_smooth_low + y_smooth
    echoed = echo(y, 63, .5, sr)
    echoed_norm = echoed / np.max(np.abs(echoed))
    sf.write("audio/tests/y_smooth.wav", y_smooth, sr)
    sf.write("audio/tests/echoed.wav", echoed, sr)

