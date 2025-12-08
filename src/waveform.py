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
    t = np.arange(0, wavelength, 1.0 / sr)
    s = np.random.uniform(-1, 1, n)
    s = np.concatenate((s, [s[0]]))
    def s_func(t):
        t_norm = t - wavelength * np.floor(t / wavelength)
        j = np.ceil(t_norm * n / wavelength).astype(int)
        return n / wavelength * (s[j] - s[j-1]) * (t_norm - (j - 1) * wavelength / n) + s[j-1]
    j = np.array([i for i in range(1, n)])
    coefficients = []
    coefficients_test = []
    for k in range(1, M + 1):
        left = 2 * np.pi * k * (j - 1) / n
        right = 2 * np.pi * k * j / n
        coeff = np.sum(-s[j] / (np.pi * k) * np.cos(right) + n / (2 * np.pi ** 2 * k ** 2) * (s[j] - s[j-1]) * (np.sin(right) - np.sin(left)))
        coefficients_test.append(coeff)
    for k in range(1, M+1):
        f = lambda t: s_func(t) * np.sin(2 * np.pi * k * t / wavelength)
        coefficients.append(integrate(f, 0, wavelength, 1000))
    print(coefficients_test)
    print(coefficients)
    t = np.arange(0, duration, 1.0 / sr)
    t_norm = t - wavelength * np.floor(t / wavelength)
    j = np.ceil(t_norm * n / wavelength).astype(int)
    y_sharp = s_func(t)
    y_sharp /= np.max(np.abs(y_sharp))
    y_smooth = np.sum(list(map(lambda k: coefficients[k] * np.sin(2 * np.pi * (k+1) * t / wavelength), [i for i in range(len(coefficients))])), axis=0)
    y_smooth_test = np.sum(list(map(lambda k: coefficients_test[k] * np.sin(2 * np.pi * (k + 1) * t / wavelength), [i for i in range(len(coefficients_test))])), axis=0)
    y_smooth /= np.max(np.abs(y_smooth))
    y_smooth_test /= np.max(np.abs(y_smooth_test))
    return y_sharp, y_smooth, y_smooth_test

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
    t = np.arange(0, .02, 1.0 / sr)
    y_sharp, y, y_test = random_smoothed(1, .02, 440, 44100, 0, 50, 100)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(t, y_sharp)
    ax.plot(t, y)
    ax.plot(t, y_test)
    sf.write("audio/tests/y_sharp.mp3", y_sharp, sr)
    sf.write("audio/tests/y_smooth.mp3", y, sr)
    sf.write("audio/tests/y_smooth_test.mp3", y_test, sr)

