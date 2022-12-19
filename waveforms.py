from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QSoundEffect
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import shutil
import soundfile as sf


AUDIO_DIR = "audio"

def sine(vol, duration, hz, sr, shift=0):
    assert duration > 0
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


def play(waveform, sr=44100):
    path = os.path.join(AUDIO_DIR, "tmp.wav")
    sf.write(path, waveform, sr)
    url = QUrl.fromLocalFile(path)
    effect = QSoundEffect()
    effect.setSource(url)
    effect.setLoopCount(QSoundEffect.Infinite)
    effect.play()
    return effect


class Waveform:
    """
    All of amp, hz, and shift can be functions. The form parameter is one of
    the four available waveforms (sine, sawtooth, square, and triangle)
    """
    def __init__(self, amp, duration, hz, shift, form=sine, sr=44100):
        self.index = 0
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
            "form": FORM_MAP[self.form]
        }

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

