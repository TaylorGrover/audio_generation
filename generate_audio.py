import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import time


def get_relative_minor(hz):
    minor = [
        hz / 2,
        hz * 2 ** (-5 / 12),
        hz,
        hz * 2 ** (1 / 6),
        hz * 2 ** (1 / 4),
        hz * 2 ** (5 / 12),
        hz * 2 ** (7 / 12),
        hz * 2 ** (5 / 6),
        hz * 2
    ]
    return minor


def softmax(z):
    assert type(z) is np.ndarray
    exp = np.exp(z)
    if len(exp.shape) == 2:
        
        return exp / np.sum(exp, axis=1).reshape(-1, 1)
    return np.exp(z) / np.sum(exp)


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
    fade = np.concatenate((audio[:start_index], audio[start_index:] - audio[start_index:] * np.arange(0, 1, 1 / (n - start_index))))
    fig, ax = plt.subplots()
    ax.plot(fade)
    return fade


def random_note(vol, duration, hz, sr, n):
    dist = softmax(np.random.random(n))
    t = np.arange(0, duration, 1.0 / sr)
    minor_choices = get_relative_minor(hz)
    #minor_choices = [hz / 2, hz, 2 ** (1 / 6) * hz, 2 ** (1 / 4) * hz, 2 ** (7 / 12) * hz, 2 ** (-5 / 12) * hz, 2 * hz]
    
    seq = vol * np.array([np.sin(np.random.random() * t) * sine(vol, duration, np.random.choice(minor_choices) + np.random.normal(0, 1), sr) for i in range(n)])
    combined = np.sum(seq, axis=0)
    normalized = combined / np.max([np.abs(np.min(combined)), np.abs(np.max(combined))])
    return normalized


def duh_nee_duh_noo(sample_rate):
    sample1 = aug_note(1, .5, 440, sample_rate)
    sample2 = aug_note(1, .5, 440 * 2 ** (1 / 12), sample_rate)
    sample3 = aug_note(1, .5, 440 * 2 ** (-2 / 12), sample_rate)
    sample4 = aug_note(1, .5, 440 * 2 ** (6 / 12), sample_rate)
    audio = np.concatenate((sample1, sample2, sample1, sample3, sample4, sample1))
    fig, ax = plt.subplots()
    ax.plot(audio)
    fig.savefig(f"images/{sample_rate}.png")
    sf.write(f"audio/{sample_rate}.wav", audio, sample_rate)
    return audio


def pan(audio, direction):
    assert -1 <= direction <= 1
    panned = np.zeros((2, len(audio)))


def stereo_test(sample_rate):
    sample1 = aug_note(1, .5, 440, sample_rate)
    sample2 = aug_note(1, .5, 440 * 2 ** (3 / 12), sample_rate)
    return np.array([sample1, sample2]).T


if __name__ == "__main__":
    sample_rate = 44100
    audio = random_note(1, 10, 220, sample_rate, 50)
    plt.ion()
    faded = fade_out(audio, 5, sample_rate)
    #audio = sawtooth(1, 2, 3, sample_rate, form="positive")
    sf.write("audio/faded.wav", faded, sample_rate)
