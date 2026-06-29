from abc import abstractmethod, ABC

class UserAction(ABC):
    """
    TODO
    """
    def __init__(self):
        """
        """
    @abstractmethod
    def reverseAction(self):
        pass

"""
TODO:

class AddPointAction(UserAction):

class MovePointAction(UserAction):

class DeletePointAction(UserAction):

class ClearWaveACtion(UserAction)
"""

class ActionMonitor:
    def __init__(self):
        """
        """
        self.undo_history = []
        self.redo_history = []