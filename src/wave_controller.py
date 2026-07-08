from action_monitor import ActionMonitor
import playsound
import utilities
import waveform
if utilities.getOS() == "windows":
    import winsound

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

        self.actionMonitor = ActionMonitor()

        ##### SIGNALS #####

        # Graph the current toggled waveform
        self.view.graphSignal.connect(self.graphComponentWaveform)

        # Play the current toggled waveform
        self.view.playSignal.connect(self.playCurrentWaveform)

        # Add a wave to the catalog
        # TODO: Finish implementing this
        self.view.initiateCatalogAdditionSignal.connect(self.openCatalogAdditionDialog)
        self.view.createCatalogWaveSignal.connect(self.createCatalogWave)
        

        # TODO: Delete a wave from the catalog

        # TODO: Add an interpolating point to a wave, referenced by index or name
        self.view.pointAdditionSignal.connect(self.addPointToWave)

        # TODO: Add a catalog wave to the synthesizing workspace as a component oscillator

        # TODO: Remove a catalog wave from the component list of the synthesizing workspace

        # TODO: Add an amplitude envelope to a component oscillator in the synthesizing workspace. Referenced by indexes for envelope and oscillator waves

        # TODO: Remove an amplitude envelope from a component oscillator in the synthesizing workspace. Refrenced by indexes for envelope and oscillator waves

        # TODO: Add a frequency envelope to a component oscillator in the synthesizing workspace. Referenced by indexes for envelope and oscillator

        # TODO: Remove a frequency envelope from a component oscillator in the synthesizing workspace. Referenced by indexes for envelope and oscillator 

        # TODO: Adjust volume for component oscillator

        # TODO: Change point position for component oscillator

        # TODO: Change fundamental frequency of component oscillator
        # TODO: Change cents of oscillator
        # TODO: Change octave of oscillator
        self.view.frequencyChangedSignal.connect(self.updateFrequency)

        # TODO: Change duration of global waveform

        # TODO: Clear component waveform

        # TODO: Copy waveform to envelope folder

        # TODO: Add global amplitude envelope to synthesizing workspace

        # TODO: Remove global amplitude envelope from synthesizing workspace

        # TODO: Add global frequency envelope to synthesizing workspace

        
        # Button shortcuts
        #self.actionMonitor = actionMonitor
    
    def openCatalogAdditionDialog(self, event):
        """
        Add a wave graph to the catalogSection vbox
        """
        # Get next available key
        self.view.openCatalogAdditionDialog(event)

    def playCurrentWaveform(self, key):
        operating_system = utilities.getOS()
        if key == 0: # This might be bad design, but the zero index is the global view
            wave = self.model.getCombinedWave()
        wave = self.model.getSineExtrapolatedWave(key)
        path = waveform.generateWaveFilepath()
        print(path)
        waveform.saveWavFile(path, wave, self.model.sample_rate)
        if operating_system == "windows":
            winsound.PlaySound(path, winsound.SND_ASYNC)
        elif operating_system == "linux":
            whatisit = playsound.playsound(path, block=False)
            print(whatisit)
            print(type(whatisit))

    def graphComponentWaveform(self, key):
        x, y = self.model.getPointsXY(key)
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
            self.view.showComponentGraph(self.keyIndexCounter)
            self.keyIndexCounter += 1
        else:
            self.view.displayDupNameErrMsg()

    def updateFrequency(self, keyIndex, baseFreq, cents, octave):
        self.model.updateFrequency(keyIndex, baseFreq, cents, octave)

    def addPointToWave(self, keyIndex, x, y):
        self.model.addPoint(keyIndex, x, y)
        self.graphComponentWaveform(keyIndex)
