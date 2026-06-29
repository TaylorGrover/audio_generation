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

        # Add wave button clicked: Register graph button signals
        self.view.catalogAdditionSignal.connect(self.addWaveToCatalog)
        
        # Button shortcuts
        #self.actionMonitor = actionMonitor
    
    def addWaveToCatalog(self, event):
        """
        Add a wave graph to the catalogSection vbox
        """
        
        print("Connected: ", event)

    
