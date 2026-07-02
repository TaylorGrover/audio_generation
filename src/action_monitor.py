from abc import abstractmethod, ABC

class UserAction(ABC):
    """
    TODO
    """
    def __init__(self):
        """
        """
    @abstractmethod
    def doAction(self):
        pass
    @abstractmethod
    def reverseAction(self):
        pass

"""
TODO: Implement the following classes

class AddPointAction(UserAction):

class MovePointAction(UserAction):

class DeletePointAction(UserAction):

class ClearWaveAction(UserAction)
"""
class AddPointAction(UserAction):
    def __init__(self):
        super().__init__()

class ActionMonitor:
    def __init__(self):
        """
        """
        self.undo_history = []
        self.redo_history = []
