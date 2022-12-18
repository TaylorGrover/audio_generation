import librosa
import librosa.display
import matplotlib.pyplot as plt
from note_structure import *
import numpy as np
import soundfile as sf
import time
from util import *
from waveforms import *


def aug_note(vol, duration, hz, sr):
    """
    Add timbre
    """
    assert duration > 0
    t = np.arange(0, duration, 1.0 / sr)

    raw = .4 * sine(vol, duration, hz, sr)
    add1 = .2 * vol * sine(vol, duration, hz, sr)
    #add1 = .3 * vol * square(vol, duration, hz * 2 ** (4 / 12), sr)
    add2 = .3 * vol * (1 - np.exp(-2 * t / duration)) * square(vol, duration, hz * 2 ** (7 / 12), sr)

    dist = softmax(np.random.random(10))
    seq = .1 * vol * np.array([dist[i] * sine(vol, duration, np.random.choice([hz / 4, hz / 2, hz]) + np.random.normal(), sr) for i in range(10)])
    add3 = np.sum(seq, axis=0)
    return raw + add1 + add2 + add3


def fade_out(audio, seconds_from_end, sr):
    """
    
    """
    n = len(audio)
    if seconds_from_end * sr >= n:
        return audio
    start_index = n - seconds_from_end * sr
    fade = np.concatenate((audio[:start_index], audio[start_index:] - audio[start_index:] * np.arange(0, 1, 1 / (n - start_index)).reshape(-1, 1)))
    return fade


def alien(vol, duration, hz, sr, n, key="minor"):
    dist = softmax(np.random.random(n))
    t = np.arange(0, duration, 1.0 / sr)
    frequencies = get_frequencies(key, hz)
    funcs = [sine, sawtooth, square]
    seq = vol * np.array([pan(sine(vol, duration, np.random.choice(frequencies) + t * np.exp(-np.random.normal(0, 10) * t), sr, shift=.4 * np.random.random() - .2), np.random.random()) for i in range(n)])
    combined = np.sum(seq, axis=0)
    normalized = combined / np.max(np.abs(combined))
    return normalized


def shifter(vol, duration, hz, sr, n, bpm, key="power"):
    t = np.arange(0, duration, 1.0 / sr)
    frequencies = get_frequencies(key, hz)
    sample1 = .2 * harmonics(vol, duration, hz, sr, n, key="power")
    sample2 = .2 * np.sin(2 * np.pi * hz * t - np.sin(bpm * t))
    sample3 = .2 * np.sin((2 * np.pi * hz + np.sin(bpm * t)) * t - np.sin(bpm * t))
    sample4 = .1 * sawtooth(vol, duration, 2 * hz, sr, shift=np.sin(bpm * t))
    sample5 = .3 * harmonics(vol, duration, hz * 2 ** (7 / 12), sr, n, key="power")

    return sample1 + pan(sample2) + pan(sample3) + pan(sample4)


def harmonics(vol, duration, hz, sr, n, key="minor"):
    """
    Spooky noises
    """
    funcs = [sine, sawtooth, square]
    t = np.arange(0, duration, 1.0 / sr)
    frequencies = get_frequencies(key, hz)
    output = np.sum(np.array([pan(np.random.choice(funcs)(vol, duration, np.random.choice(frequencies) + np.sin(np.random.random() * t), sr, shift=.4 * np.random.random() - .2), direction=np.random.random()) for i in range(n)]), axis=0)
    rescaled = output / np.max(np.abs(output))
    return rescaled


def duh_nee_duh_noo(sample_rate, key="minor", bpm=100):
    """
    Generate random sequence of notes in either major/minor key. Assume 4/4 time
    """
    notes = []
    audio = np.concatenate(notes)
    fig, ax = plt.subplots()
    ax.plot(audio)
    fig.savefig(f"images/{sample_rate}.png")
    sf.write(f"audio/{sample_rate}.wav", audio, sample_rate)
    return audio


def pan(audio, direction=0.5):
    assert 0 <= direction <= 1
    panned = np.array([(1 - direction) * audio, direction * audio]).T
    return panned



def stereo_test(sample_rate):
    sample1 = aug_note(1, .5, 293.66, sample_rate)
    sample2 = aug_note(1, .5, 293.66 * 2 ** (3 / 12), sample_rate)
    return np.array([sample1, sample2]).T


def soul_hemorrhage(sample_rate, bpm):
    n = 100
    sample1 = harmonics(1, 4 / (bpm / 60), 293.66 * 2 ** (1 / 12), sample_rate, n, key="major")
    sample2 = harmonics(1, 4 / (bpm / 60), 293.66, sample_rate, n, key="major")
    sample3 = harmonics(1, 3 / (bpm / 60), 293.66 * 2 ** (1 / 12), sample_rate, n, key="power")
    sample4 = harmonics(1, .5 / (bpm / 60), 293.66 * 2 ** (-1 / 6), sample_rate, n, key="power")
    sample5 = harmonics(1, .5 / (bpm / 60), 293.66, sample_rate, n, key="power")
    sample6 = harmonics(1, 4 / (bpm / 60), 293.66, sample_rate, n, key="power")
    sample7 = harmonics(1, 4 / (bpm / 60), 293.66 * 2 ** (5 / 12), sample_rate, n, key="minor")
    sample8 = harmonics(1, 4 / (bpm / 60), 293.66 * 2 ** (1 / 4), sample_rate, n, key="major")
    audio = np.concatenate((sample1, sample2, sample3, sample4, sample5, sample4, sample6, sample1, sample7, sample8))
    faded = fade_out(audio, 1, sample_rate)
    sf.write("audio/hemorrhage.wav", faded, sample_rate)


if __name__ == "__main__":
    sample_rate = 44100
    bpm = 120
    soul_hemorrhage(sample_rate, 155)
    #audio = shifter(1, 4, 293.66, sample_rate, 10, bpm)
    #sf.write("audio/test.wav", audio, sample_rate)
    #spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    #librosa.display.specshow(spec)
