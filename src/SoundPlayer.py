import numpy as np
import os
import playsound3
import utilities
import waveform

class SoundPlayer:
    def __init__(self):
        self.sound = None
    def play(self, wave: np.ndarray, sr:int):
        path = waveform.generateWaveFilepath()
        waveform.saveWavFile(path, wave, sr)
        self.sound = playsound3.playsound(path, block=False)
            

    def stop(self):
        if self.sound is not None:
            self.sound.stop()
