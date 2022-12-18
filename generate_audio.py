import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def sine(vol, duration, hz, sr):
    assert duration > 0
    t = np.arange(0, duration, 1.0 / sr)
    return vol * np.sin(2 * np.pi * hz * t)


def square(vol, duration, hz, sr):
    diff = sawtooth(vol, duration, hz, sr)
    geq = diff >= 0
    pos = 1.0 * geq
    neg = -1.0 * (~geq)
    return vol * (pos + neg)


def sawtooth(vol, duration, hz, sr, form="positive"):
    assert duration > 0
    t = np.arange(0, duration, 1.0 / sr)
    scaled = t * hz
    rounded = np.round(scaled)
    diff = scaled - rounded
    if form == "negative":
        diff = -diff
    return 2 * vol * diff


def triangular(vol, duration, hz, sr):
    assert duration > 0
    t = np.arange(0, duration, 1.0 / sr)
    scaled = t * hz
    


def aug_note(vol, duration, hz, sr):
    """
    Add timbre
    """
    assert duration > 0
    t = np.arange(0, duration, 1.0 / sr)

    raw = .4 * sine(vol, duration, hz, sr)
    add1 = .3 * vol * sawtooth(vol, duration, hz, sr)
    #add1 = .3 * vol * square(vol, duration, hz * 2 ** (4 / 12), sr)
    add2 = .3 * vol * (1 - np.exp(-2 * t / duration)) * square(vol, duration, hz * 2 ** (7 / 12), sr)
    return raw + add1 + add2


def duh_nee_duh_noo(sample_rate):
    sample1 = aug_note(1, .5, 440, sample_rate)
    sample2 = aug_note(1, .5, 440 * 2 ** (1 / 12), sample_rate)
    sample3 = aug_note(1, .5, 440 * 2 ** (-2 / 12), sample_rate)
    sample4 = aug_note(1, .5, 440 * 2 ** (6 / 12), sample_rate)
    audio = np.concatenate((sample1, sample2, sample1, sample3, sample4, sample1))
    fig, ax = plt.subplots()
    ax.plot(audio)
    fig.savefig(f"{sample_rate}.png")
    sf.write(f"{sample_rate}.wav", audio, sample_rate)
    return audio


def pan(audio, direction):
    assert -1 <= direction <= 1
    panned = np.zeros((2, len(audio)))


def stereo(sample_rate):
    sample1 = aug_note(1, .5, 440, sample_rate)
    sample2 = aug_note(1, .5, 440 * 2 ** (3 / 12), sample_rate)
    return np.array([sample1, sample2]).T


if __name__ == "__main__":
    sample_rate = 44100
    audio = duh_nee_duh_noo(sample_rate)
    plt.ion()
    #audio = sawtooth(1, 2, 3, sample_rate, form="positive")
    sf.write("decay.wav", audio, sample_rate)

