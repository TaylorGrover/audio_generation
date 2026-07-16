from action_monitor import ActionMonitor
import multiprocessing
import numpy as np
import SoundPlayer
import waveform


"""
This is the interface between GUI and wave data model.
"""
class WaveController:
    """
    Communicate wave information between GUI and wave transformation functionality

    TODO: Build interface for each of the signals returned by the GUI
    """
    
    def __init__(self, view, model):
        """
        Handle signals from widgets
        """
        self.currentKeyIndex = 1
        self.keyIndexCounter = 1
        self.view = view 
        self.model = model
        self.isPlaying = False
        self.soundPlayer = SoundPlayer.SoundPlayer()
        
        # Set the view's duration button based on the model
        self.view.setDurationWidgetValue(self.model.getDuration())

        # TODO: Decide between initializing the model parameters based on the view defaults or vice versa.
        

        self.actionMonitor = ActionMonitor()

        ##### SIGNALS #####

        # Graph the current toggled waveform
        self.view.graphSignal.connect(self.graphComponentWaveform)

        # Play the current toggled waveform
        self.view.playSignal.connect(self.playCurrentWaveform)

        # Stop the currently playing audio
        self.view.stopAudioSignal.connect(self.stopAudio)

        # Add a wave to the catalog
        # TODO: Finish implementing this
        self.view.initiateCatalogAdditionSignal.connect(self.openCatalogAdditionDialog)
        self.view.createCatalogWaveSignal.connect(self.createCatalogWave)

        # TODO: Delete a wave from the catalog

        # Swap a component wave with another one and update widget parameter values
        self.view.adjustParameterWidgetsSignal.connect(self.adjustParameterWidgets)

        # TODO: Add an interpolating point to a wave, referenced by index or name
        self.view.pointAdditionSignal.connect(self.addPointToWave)

        # TODO: Check or uncheck sine interpolation
        self.view.sineStateChangedSignal.connect(self.updateSineInterpolationChecked)

        # TODO: Update sine count for a component wave
        self.view.sineCountChangedSignal.connect(self.updateSineCount)

        # TODO: Add a catalog wave to the synthesizing workspace as a component oscillator

        # TODO: Remove a catalog wave from the component list of the synthesizing workspace

        # TODO: Add an amplitude envelope to a component oscillator in the synthesizing workspace. Referenced by indexes for envelope and oscillator waves

        # TODO: Remove an amplitude envelope from a component oscillator in the synthesizing workspace. Refrenced by indexes for envelope and oscillator waves

        # TODO: Add a frequency envelope to a component oscillator in the synthesizing workspace. Referenced by indexes for envelope and oscillator

        # TODO: Remove a frequency envelope from a component oscillator in the synthesizing workspace. Referenced by indexes for envelope and oscillator 

        # TODO: Adjust volume for component oscillator
        self.view.volumeUpdateSignal.connect(self.updateVolume)

        # TODO: Change point position for component oscillator

        # TODO: Change fundamental frequency of component oscillator
        # TODO: Change cents of oscillator
        # TODO: Change octave of oscillator
        self.view.frequencyChangedSignal.connect(self.updateFrequency)

        # TODO: Change duration of global waveform
        self.view.durationChangedSignal.connect(self.updateDuration)
        # TODO: Clear component waveform
        self.view.clearGraphSignal.connect(self.clearGraphPoints)

        # TODO: Copy waveform to envelope folder

        # TODO: Add global amplitude envelope to synthesizing workspace

        # TODO: Remove global amplitude envelope from synthesizing workspace

        # TODO: Add global frequency envelope to synthesizing workspace

        
        # Button shortcuts
        #self.actionMonitor = actionMonitor
    
    def adjustParameterWidgets(self, key:int):
        """
        * Freq
        * Volume
        * Interpolation
        * 
        Need to block all signals from the parameter widgets while updating the values
        """
        freq_cents = self.model.getCents(key)
        freq_letter = self.model.getFrequencyLetter(key)
        freq_octave = self.model.getOctave(key)
        self.view.setFrequencyWidgetParametersBlocked(freq_cents, freq_letter, freq_octave)
    
    def openCatalogAdditionDialog(self, event):
        """
        Add a wave graph to the catalogSection vbox
        """
        # Get next available key
        self.view.openCatalogAdditionDialog(event)

    def updateVolume(self, key:int, vol:float):
        self.model.updateVolume(key, vol)
    
    def updateSineInterpolationChecked(self, key:int, isChecked:bool):
        self.graphComponentWaveform(key)
        self.graphCombinedWave()
        self.model.updateSineInterpolationChecked(key, isChecked)

    def updateSineCount(self, key:int, count:int):
        self.graphComponentWaveform(key)
        self.graphCombinedWave()
        self.model.updateSineCount(key, count)

    def updateFrequency(self, keyIndex, baseFreq, cents, octave):
        self.graphCombinedWave()
        self.model.updateFrequency(keyIndex, baseFreq, cents, octave)

    # Updates all waves
    def updateDuration(self, duration:float):
        self.model.updateDuration(duration)

    def playCurrentWaveform(self, key):
        if key == 0: # This might be bad design, but the zero index is the global view
            wave = self.model.getPlayableCombinedWave()
        else:
            # To play specific component
            if self.model.getPointCount(key) >= 2:
                # Check that there is a minimum of 2 points
                wave = self.model.getPlayableComponentWave(key, recalculate=True)
            else:
                return
        duration = self.model.getDuration()
        self.view.setStopTimer(duration + .006) # TODO: Fix later
        self.view.setCurrentlyPlayingStatus()
        self.soundPlayer.play(wave, self.model.getSampleRate())

    def stopAudio(self):
        self.soundPlayer.stop()

    def graphComponentWaveform(self, key):
        points = self.model.getPoints(key)
        if len(points) == 0:
            self.clearComponentGraph()
            return
        x, y = zip(*points)
        if len(x) >= 2:
            interp_x, interp_y = self.model.getInterpolatedXY(key)
            sine_x, sine_y = self.model.getSineInterpolatedXY(key)
        else:
            interp_x = x
            interp_y = y
            sine_x = x
            sine_y = y
        self.view.graphComponentWaveform(key, x, y, interp_x, interp_y, sine_x, sine_y)

    def createCatalogWave(self, name:str):
        if not self.model.nameExists(name):
            self.view.addWaveToCatalog(self.keyIndexCounter, name)
            self.view.closeWaveNameInputWidget()
            self.model.createEmptyWave(self.keyIndexCounter, name)
            self.clearComponentGraph()
            self.adjustParameterWidgets(self.keyIndexCounter)
            self.view.swapGraphs()
            self.keyIndexCounter += 1
        else:
            self.view.displayDupNameErrMsg()

    def addPointToWave(self, keyIndex, x, y):
        self.model.addPoint(keyIndex, x, y)
        self.graphComponentWaveform(keyIndex)
        self.graphCombinedWave()

    def graphCombinedWave(self):
        t = self.model.getTime()
        wave = self.model.getCombinedWave()
        self.view.graphCombinedWave(t, wave, self.model.getSampleRate())

    def clearGraphPoints(self, keyIndex):
        self.model.clearGraphPoints(keyIndex)
        self.graphCombinedWave()

    def clearComponentGraph(self):
        """
        This is distinct from clearGraphPoints in that it just
        clears a component graph; it doesn't actually remove the 
        points from the model.
        """
        self.view.clearComponentGraph()
