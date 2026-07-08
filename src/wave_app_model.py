import bisect
import numpy as np
import waveform

class WaveModel:
    def __init__(self):
        self.waveDict = {}
        self.point_key_str = "points"
        self.name_key_str = "name"
        self.linear_interp_str = "linear_interp"
        self.sine_interp_str = "sine_interp"
        self.freq_str = "frequency"
        self.amp_str = "amplitude"
        self.sine_count_str = "sine_count"
        self.duration = 5
        self.sample_rate = 44100
        self.t = np.linspace(0, self.duration, int(self.duration * self.sample_rate))

    def createEmptyWave(self, key_index:int, name:str):
        self.waveDict[key_index] = {
            self.name_key_str: name
            , self.point_key_str: []
            , self.linear_interp_str: np.array([[]])
            , self.sine_interp_str: np.array([])
            , self.amp_str: 1.0
            , self.freq_str: waveform.F
            , self.sine_count_str: 13
        }

    def nameExists(self, name:str) -> bool:
        for keyIndex in self.waveDict:
            if self.waveDict[keyIndex][self.name_key_str] == name:
                return True
        return False
    def hasKey(self, key:int) -> bool:
        return key in self.waveDict

    def getPoints(self, key):
        return self.waveDict[key][self.point_key_str]

    def getPointCount(self, key):
        return len(self.getPoints(key))

    def getPointsXY(self, key):
        return zip(*self.getPoints(key))

    def getInterpolated(self, key):
        return self.waveDict[key][self.linear_interp_str]

    def getInterpolatedXY(self, key):
        return self.getInterpolated(key).T

    def getSineInterpolatedXY(self, key):
        return self.waveDict[key][self.sine_interp_str]

    def addPoint(self, key, x, y):
        """
        Add a new point to a component waveform.  
        TODO: Fix update sine interpolation
        """
        if self.hasKey(key):
            bisect.insort(self.waveDict[key][self.point_key_str], [x, y], key=lambda t: t[0])
            if self.getPointCount(key) >= 2:
                self.updateLinearInterpolation(key)
                self.updateSineInterpolation(key)

    def clearGraphPoints(self, keyIndex):
        self.waveDict[keyIndex][self.point_key_str] = []
        self.waveDict[keyIndex][self.linear_interp_str] = np.array([[]])
        self.waveDict[keyIndex][self.sine_interp_str] = np.array([])

    def updateLinearInterpolation(self, key, interp_factor:int=2):
        x, y = self.getPointsXY(key)
        new_x = np.linspace(x[0], x[-1], interp_factor * len(x))
        new_y = np.interp(new_x, x, y)
        self.waveDict[key][self.linear_interp_str] = np.array([new_x, new_y]).T

    def updateSineInterpolation(self, key:int):
        x, y = self.getInterpolatedXY(key)
        duration = x[-1] - x[0]
        sine_time = np.linspace(x[0], x[-1], int(duration * self.sample_rate))
        amplitude = np.max(np.abs(y))
        wave = waveform.seeded_waveform(
            amplitude
            , duration
            , 1.0 / duration
            , y
            , self.sample_rate
            , sine_count=13 # TODO: Fix this 
        ).T[0]
        self.waveDict[key][self.sine_interp_str] = np.array([sine_time, wave])

    def updateFrequency(self, key, baseFreq, cents, octave):
        self.waveDict[key][self.freq_str] = waveform.NOTE_FREQUENCY_MAP[baseFreq] * 2 ** (cents / 1200) * 2 ** (octave - 1)

    def getFrequency(self, key):
        return self.waveDict[key][self.freq_str]

    def getDuration(self):
        return self.duration
    def getAmplitude(self, key):
        return self.waveDict[key][self.amp_str]
    def getSineCount(self, key):
        return self.waveDict[key][self.sine_count_str]

    def getExtrapolatedWave(self, key):
        wavelength_sample_count = int(self.sample_rate / self.getFrequency(key))
        total_sample_count = int(self.duration * self.sample_rate)
        x, y = self.getInterpolatedXY(key)
        t_base = np.linspace(0, x[-1] - x[0], wavelength_sample_count)
        t_indices = np.linspace(0, total_sample_count, total_sample_count).astype(int)
        t_modulo = t_indices % wavelength_sample_count
        y_interp = np.interp(t_base, x, y)
        y_interp /= np.max(np.abs(y_interp), axis=0)
        wave = self.getAmplitude(key) * y_interp[t_modulo]
        return wave

    def getSineExtrapolatedWave(self, key):
        volume = self.getAmplitude(key)
        frequency = self.getFrequency(key)
        sine_count = self.getSineCount(key)
        x, y = self.getInterpolatedXY(key)
        wave = volume * waveform.seeded_waveform(1, self.duration, frequency, y, self.sample_rate, sine_count)
        return wave

    def getCombinedWave(self, norm=True):
        return np.zeros_like(self.t)

class Project:
    def __init__(self, project_name:str):
        self.project_name = project_name
        self.wave_catalog = dict()
    
