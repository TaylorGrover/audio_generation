import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def sine(vol, duration, hz, sr):
    assert duration > 0
    t = np.arange(0, duration, 1.0 / sr)
    return vol * np.sin(2 * np.pi * hz * t)


def square(vol, duration, hz, sr):
    assert duration > 0
    t = np.arange(0, duration, 1.0 / sr)
    interval = 1.0 / hz / 2



def aug_note(vol, duration, hz, sr):
    """
    Add timbre
    """
    assert duration > 0
    t = np.arange(0, duration, 1.0 / sr)

    raw = .4 * sine(vol, duration, hz, sr)
    add1 = .3 * vol * np.sin(2 * np.pi * hz * 2 ** (4 / 12) * t)
    add2 = .3 * vol * np.sin(2 * np.pi * hz * 2 ** (7 / 12) * t)
    return raw + add1 + add2


def duh_nee_duh_noo(sample_rate):
    sample1 = aug_note(2, .5, 440, sample_rate)
    sample2 = aug_note(2, .5, 440 * 2 ** (1 / 12), sample_rate)
    sample3 = aug_note(2, .5, 440 * 2 ** (-2 / 12), sample_rate)
    audio = np.concatenate((sample1, sample2, sample1, sample3))
    fig, ax = plt.subplots()
    ax.plot(audio)
    fig.savefig(f"{sample_rate}.png")
    sf.write(f"{sample_rate}.wav", audio, sample_rate)


sample_rate = 44100
duh_nee_duh_noo(sample_rate)
