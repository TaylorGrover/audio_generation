import sys
sys.path.append("src/")

from waveform import *

def test_sine_zero_amp():
    """
    Ensure that each amplitude is zero if we set vol = 0
    """
    sr = 44100
    res = sine(0, 1, 2, sr)
    assert all(res == 0), "All amplitudes should equal 0"

def test_sine_zero_sr():
    """
    Ensure that sine throws an error for zero sample rate
    """
    sr = 0
    zero_error = False
    assertion_error = False
    try:
        sine(1, 1, 1, sr)
    except AssertionError:
        # This is the intended outcome
        assertion_error = True
        pass
    except ZeroDivisionError:
        zero_error = True
    assert not zero_error, "sine should not allow ZeroDivisionError"
    assert assertion_error, "sine should throw AssertionError"

def test_square_zero_sr():
    """
    Ensure square throws AssertionError for 0 sample rate
    """
    sr = 0
    zero_error = False
    assertion_error = False
    try:
        square(1, 1, 1, sr)
    except AssertionError:
        # This is the intended outcome
        assertion_error = True
        pass
    except ZeroDivisionError:
        zero_error = True
    assert not zero_error, "square should not allow ZeroDivisionError"
    assert assertion_error, "square should throw AssertionError"


def test_soundplayer_play():
    """
    """
    form = Waveform(1, 1, 440, 0, form=sine)
    sp = SoundPlayer(form, 44100)
