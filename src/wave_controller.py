from action_monitor import ActionMonitor
"""
This is the interface between GUI and wave data model.
"""
class WaveController:
    """
    Communicate wave information between GUI and wave transformation functionality

    TODO: Build interface for each of the signals returned by the GUI
    """
    def __init__(self, view):
        """
        Handle signals from widgets
        """
        self.view = view 

        self.actionMonitor = ActionMonitor()

        ##### SIGNALS #####

        # Add a wave to the catalog
        # TODO: Finish implementing this
        self.view.catalogAdditionSignal.connect(self.addWaveToCatalog)

        # TODO: Delete a wave from the catalog

        # TODO: Add an interpolating point to a wave, referenced by index or name

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
    
    def addWaveToCatalog(self, event):
        """
        Add a wave graph to the catalogSection vbox
        """
        print("Connected: ", event)

    
