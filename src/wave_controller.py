from action_monitor import ActionMonitor
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

    def playCurrentWaveform(self, event):
        print(event)

    def graphComponentWaveform(self, key):
        if self.model.hasKey(key):
            points = self.model.getPoints(key)
            self.view.graphComponentWaveform(self, key, points)

    def createCatalogWave(self, name:str):
        if not self.model.nameExists(name):
            self.view.addWaveToCatalog(self.keyIndexCounter, name)
            self.view.closeWaveNameInputWidget()
            self.model.createEmptyWave(self.keyIndexCounter, name)
            self.view.showComponentGraph(self.keyIndexCounter)
            self.keyIndexCounter += 1
        else:
            self.view.displayDupNameErrMsg()

    def addPointToWave(self, keyIndex, x, y):
        self.model.addPoint(keyIndex, x, y)
        points = self.model.getPoints(keyIndex)
        self.view.graphComponentWaveforms(points)