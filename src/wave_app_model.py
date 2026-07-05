import bisect
import numpy as np
import waveform

class WaveModel:
    def __init__(self):
        self.waveDict = {}
        self.point_key_str = "points"
        self.name_key_str = "name"
        self.linear_interp_str = "linear_interp"
        self.duration = 5
        self.sample_rate = 44100
        self.t = np.linspace(0, self.duration, int(self.duration * self.sample_rate))

    def createEmptyWave(self, key_index:int, name:str):
        self.waveDict[key_index] = {
            self.name_key_str: name,
            self.point_key_str: [],
            self.linear_interp_str: np.array([[]]),
            self.sine_interp_str: np.array([])
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

    def addPoint(self, key, x, y):
        """
        Add a new point to a component waveform.  
        """
        if self.hasKey(key):
            bisect.insort(self.waveDict[key][self.point_key_str], [x, y], key=lambda t: t[0])
            if self.getPointCount() >= 2:
                self.updateLinearInterpolation(key)
                self.updateSineInterpolation(key)

    def updateLinearInterpolation(self, key, interp_factor:int=2):
        x, y = self.getPointsXY()
        new_x = np.linspace(x[0], x[-1], interp_factor * len(x))
        new_y = np.interp(new_x, x, y)
        self.waveDict[key][self.linear_interp_str] = np.array([new_x, new_y]).T

    def updateSineInterpolation(self, key:int, amp:float, sine_count:int):
        x, y = self.getInterpolatedXY(key)

        


class Project:
    def __init__(self, project_name:str):
        self.project_name = project_name
        self.wave_catalog = dict()
    