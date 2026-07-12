"""
Proof-of-concept for displaying simple hand-traced points; sine and linear interpolation;
and audio playback of each form.
"""
import bisect
import json
import numpy as np
import os
from PySide6 import QtCore
from PySide6.QtCore import QUrl, QSize, QTimer, Signal
from PySide6.QtGui import QAction, QImage
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDial, QDoubleSpinBox, QFrame, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox, QSizePolicy, QSlider, QToolBar, QVBoxLayout, QWidget)
import waveform

import pyqtgraph as pg

class WaveView(QMainWindow):
    undoSignal = Signal(int)
    redoSignal = Signal(int)

    clearGraphSignal = Signal(int)
    createCatalogWaveSignal = Signal(str)
    durationChangedSignal = Signal(float)
    frequencyChangedSignal = Signal(int, str, int, int)
    graphSignal = Signal(int) # Emits the key for the specific graph to redraw
    initiateCatalogAdditionSignal = Signal(int)
    playSignal = Signal(int)
    pointAdditionSignal = Signal(int, float, float)
    sineCountChangedSignal = Signal(int, int)
    sineStateChangedSignal = Signal(int, bool)
    volumeUpdateSignal = Signal(int, float)
    stopAudioSignal = Signal()

    def __init__(self):
        super().__init__()

        self.maximized = False
        toolBar = QToolBar()
        self.addToolBar(toolBar)
        fileMenu = self.menuBar().addMenu("&File")
        self.populateFileMenu(fileMenu)
        editMenu = self.menuBar().addMenu("&Edit")
        self.populateEditMenu(editMenu)
        viewMenu = self.menuBar().addMenu("&View")
        self.maxWidth = self.screen().availableGeometry().width()
        self.maxHeight = self.screen().availableGeometry().height()
        self.workspaceWidget = WorkspaceWidget(self.maxWidth, self.maxHeight)
        self.setCentralWidget(self.workspaceWidget)
        
        # Actions
        fullScreenAction = QAction("&Fullscreen", self, shortcut="F11", triggered=self.maximize)
        self.addAction(fullScreenAction)

        # Signals and slots
        self.workspaceWidget.initiateCatalogAdditionSignal.connect(self.emitCatalogWaveAdded)
        self.workspaceWidget.createCatalogWaveSignal.connect(lambda name: self.createCatalogWaveSignal.emit(name))
        self.workspaceWidget.playSignal.connect(self.emitPlaySignal)
        self.workspaceWidget.pointAdditionSignal.connect(self.emitPointAdditionSignal)
        self.workspaceWidget.frequencyChangedSignal.connect(self.emitFrequencyChanged)
        self.workspaceWidget.clearGraphSignal.connect(self.emitClearGraphSignal)
        self.workspaceWidget.sineStateChangedSignal.connect(self.emitSineStateChanged)
        self.workspaceWidget.sineCountChangedSignal.connect(self.emitSineCountChanged)
        self.workspaceWidget.durationChangedSignal.connect(self.emitDurationChanged)
        self.workspaceWidget.volumeUpdateSignal.connect(self.emitVolumeChanged)
        self.workspaceWidget.stopAudioSignal.connect(self.emitStopAudioSignal)

    def maximize(self):
        if self.maximized:
            self.showNormal()
        else:
            self.showMaximized()
        self.maximized = not self.maximized

    def emitStopAudioSignal(self):
        self.stopAudioSignal.emit()

    def setDurationWidgetValue(self, duration:float):
        self.workspaceWidget.setDurationWidgetValue(duration)

    def emitVolumeChanged(self, keyIndex:int, vol:float):
        self.volumeUpdateSignal.emit(keyIndex, vol)

    def emitDurationChanged(self, duration: float):
        self.durationChangedSignal.emit(duration)

    def setStopTimer(self, duration:float):
        self.workspaceWidget.setStopTimer(duration)

    def emitSineStateChanged(self, keyIndex, isChecked):
        self.sineStateChangedSignal.emit(keyIndex, isChecked)

    def emitSineCountChanged(self, keyIndex:int, count:int):
        self.sineCountChangedSignal.emit(keyIndex, count)
    
    def addWaveToCatalog(self, keyIndex:int, name:str):
        self.workspaceWidget.addWaveToCatalog(keyIndex, name)
    
    def swapGraphs(self):
        self.workspaceWidget.swapGraphs()

    def displayDupNameErrMsg(self):
        self.workspaceWidget.displayDupNameErrMsg()
    
    def closeWaveNameInputWidget(self):
        self.workspaceWidget.closeWaveNameInputWidget()

    def emitFrequencyChanged(self, keyIndex, baseFreq, cents, octave):
        self.frequencyChangedSignal.emit(keyIndex, baseFreq, cents, octave)

    def emitCatalogWaveAdded(self, event):
        self.initiateCatalogAdditionSignal.emit(event)

    def emitGraphSignal(self, key):
        self.graphSignal.emit(key)

    def emitClearGraphSignal(self, keyIndex):
        self.clearGraphSignal.emit(keyIndex)

    def populateFileMenu(self, fileMenu):
        """
        File menu functions:
        * Save
        * Open
        """
        saveWaveAction = QAction("&Save Waveform", self, shortcut="Ctrl+S", triggered=self.saveWaveformTrigger)
        loadWaveAction = QAction("&Load Waveform", self, shortcut="Ctrl+O", triggered=self.loadWaveformTrigger)
        # fileMenu.addAction(saveWaveAction)
        # fileMenu.addAction(loadWaveAction)

    def populateEditMenu(self, editMenu):
        """
        File menu functions:
        * Undo
        * Redo
        """
        undoAction = QAction("&Undo", self, shortcut="Ctrl+Z", triggered=self.emitUndoSignal)
        redoAction = QAction("&Redo", self, shortcut="Ctrl+Y", triggered=self.emitRedoSignal)
        editMenu.addAction(undoAction)
        editMenu.addAction(redoAction)

    def populateViewMenu(self, viewMenu):
        """
        TODO
        """

    def emitUndoSignal(self, event):
        """
        """
        self.undoSignal.emit(event)

    def emitRedoSignal(self, event):
        self.redoSignal.emit(event)

    def emitPlaySignal(self, index):
        self.playSignal.emit(index)

    def emitPointAdditionSignal(self, keyIndex, x, y):
        self.pointAdditionSignal.emit(keyIndex, x, y)

    def showComponentGraph(self, keyIndex:int):
        self.workspaceWidget.showComponentGraph(keyIndex)

    def saveWaveformTrigger(self):
        """
        Open a QFileDialog to save the waveform
        TODO: Fix these saving and loading functions.
        TODO: These need to save the project in its entirety rather than just a single waveform.
        """
        filePath, fileType = QFileDialog.getSaveFileName(
            self
            , "Save a File:"
            , os.getcwd()
            , "JSON Files (*.json);;"
        )
        if not filePath.lower().endswith(".json"):
            filePath = filePath + ".json"
        x, y = self.graphWidget.getPoints()
        with open(filePath, "w") as f:
            json.dump({"x": x, "y": y}, f)

    def loadWaveformTrigger(self):
        points = self.graphWidget.getPoints()
        if len(points) > 2:
            self.saveWaveformTrigger()
        filePath, fileType = QFileDialog.getOpenFileName(
            self
            , "Open a file:"
            , os.getcwd()
            , "JSON Files (*.json);;"
        )
        with open(filePath, "r") as f:
            data = json.load(f)
            self.graphWidget.setPoints(list(zip(data['x'], data['y'])))
            self.graphWidget.graphPoints()

    def graphComponentWaveform(self, key, x, y, interp_x, interp_y, sine_x, sine_y):
        self.workspaceWidget.graphComponentWaveform(key, x, y, interp_x, interp_y, sine_x, sine_y)

    def graphCombinedWave(self, t, wave):
        self.workspaceWidget.graphCombinedWave(t, wave)

    def openCatalogAdditionDialog(self, key) -> bool:
        self.workspaceWidget.openCatalogAdditionDialog(key)

    def closeEvent(self, event):
        """
        Ensure all windows are closed on the close event
        """
        QApplication.closeAllWindows()
        event.accept()

class WorkspaceWidget(QWidget):
    """
    This is the main widget containing the main graph view 
    (for editors and the summed signal)
    """
    initiateCatalogAdditionSignal = Signal(int)
    createCatalogWaveSignal = Signal(str)
    swapGraphViewSignal = Signal(int)
    
    playSignal = Signal(int)
    graphSignal = Signal(int)
    clearGraphSignal = Signal(int)
    pointAdditionSignal = Signal(int, float, float)
    frequencyChangedSignal = Signal(int, str, int, int)
    sineStateChangedSignal = Signal(int, bool)
    sineCountChangedSignal = Signal(int, int)
    durationChangedSignal = Signal(float)
    volumeUpdateSignal = Signal(int, float)
    stopAudioSignal = Signal()

    def __init__(self, maxWidth, maxHeight):
        super().__init__()
        self.maxWidth = maxWidth
        self.maxHeight = maxHeight
        self.waveformNameInputWidget = WaveNameInputWidget(maxWidth, maxHeight)
        self.waveformNameInputWidget.createCatalogWaveSignal.connect(self.createNewWaveformWindow)
        self.waveformNameInputWidget.move(self.maxWidth//2, self.maxHeight//2)
        self.waveformWidgetDict = {}
        self.gridLayout = QGridLayout(self)
        self.componentGraph = ComponentGraphWidget()
        self.centralGraph = CentralGraphWidget()

        self.catalogWidget = WaveformCatalogWidget()
        self.gridLayout.addWidget(self.catalogWidget, 0, 0)
        self.gridLayout.addWidget(self.centralGraph, 0, 1)

        # Signal management
        self.catalogWidget.initiateCatalogAdditionSignal.connect(self.emitCatalogWaveAddInitiate)
        self.catalogWidget.swapGraphsSignal.connect(self.swapGraphs)

        self.centralGraph.playSignal.connect(self.emitPlaySignal)
        self.centralGraph.durationChangedSignal.connect(self.emitDurationChanged)
        self.centralGraph.stopAudioSignal.connect(self.emitStopAudio)

        self.componentGraph.pointAdditionSignal.connect(self.emitPointAdditionSignal)
        self.componentGraph.playSignal.connect(self.emitPlaySignal)
        self.componentGraph.frequencyChangedSignal.connect(self.emitFrequencyParameters)
        self.componentGraph.clearGraphSignal.connect(self.emitClearGraphSignal)
        self.componentGraph.sineStateChangedSignal.connect(self.emitSineStateChanged)
        self.componentGraph.sineCountChangedSignal.connect(self.emitSineCountChanged)
        self.componentGraph.volumeUpdateSignal.connect(self.emitVolume)
        self.componentGraph.stopAudioSignal.connect(self.emitStopAudio)

    def swapGraphs(self):
        # Only swap graphs if the current component index is >0,
        # as this implies there actually exists a component wave
        if self.componentGraph.getKeyIndex() > 0:
            item = self.gridLayout.itemAtPosition(0, 1)
            widget = item.widget()
            if widget == self.centralGraph:
                self.catalogWidget.toggleMainButton()
                newWidget = self.componentGraph
            elif widget == self.componentGraph:
                self.catalogWidget.toggleComponentButton()
                newWidget = self.centralGraph
            self.gridLayout.removeWidget(widget)
            widget.hide()
            self.gridLayout.addWidget(newWidget, 0, 1)
            newWidget.setVisible(True)

    def emitVolume(self, key:int, vol:float):
        self.volumeUpdateSignal.emit(key, vol)
    
    def emitStopAudio(self):
        self.stopAudioSignal.emit()
    
    def emitDurationChanged(self, duration:float):
        self.durationChangedSignal.emit(duration)

    def showCentralGraph(self):
        item = self.gridLayout.itemAtPosition(0, 1)
        widget = item.widget()
        if widget == self.componentGraph:
            self.gridLayout.removeWidget(widget)
            widget.hide()
            self.gridLayout.addWidget(self.centralGraph, 0, 1)
            self.centralGraph.setVisible(True)
            self.catalogWidget.setToggleComponentButton()

    def setDurationWidgetValue(self, duration:float):
        self.centralGraph.setDurationWidgetValue(duration)

    def setStopTimer(self, duration:float):
        self.componentGraph.setStopTimer(duration)
        self.centralGraph.setStopTimer(duration)

    def emitSineStateChanged(self, keyIndex, isChecked):
        self.sineStateChangedSignal.emit(keyIndex, isChecked)

    def emitSineCountChanged(self, key, count:int):
        self.sineCountChangedSignal.emit(key, count)

    def emitClearGraphSignal(self, keyIndex):
        self.clearGraphSignal.emit(keyIndex)

    def showComponentGraph(self, keyIndex:int):
        # Check if the centralGraph is currently in place
        item = self.gridLayout.itemAtPosition(0, 1)
        self.componentGraph.setKeyIndex(keyIndex)
        widget = item.widget()
        if widget == self.centralGraph:
            self.gridLayout.removeWidget(widget)
            widget.hide()
            self.gridLayout.addWidget(self.componentGraph, 0, 1)
            self.componentGraph.setVisible(True)
            self.catalogWidget.toggleMainButton()

    def displayDupNameErrMsg(self):
        self.waveformNameInputWidget.displayDupNameErrMsg()

    def closeWaveNameInputWidget(self):
        self.waveformNameInputWidget.closeWindowAndClearInput()

    def emitFrequencyParameters(self, keyIndex, baseFreq, cents, octave):
        self.frequencyChangedSignal.emit(keyIndex, baseFreq, cents, octave)

    def emitCatalogWaveAddInitiate(self, event):
        self.initiateCatalogAdditionSignal.emit(event)

    def emitPlaySignal(self, index):
        self.playSignal.emit(index)

    def emitGraphSignal(self, key):
        self.graphSignal.emit(key)

    def emitPointAdditionSignal(self, key, x, y):
        self.pointAdditionSignal.emit(key, x, y)

    def graphComponentWaveform(self, key, x, y, interp_x, interp_y, sine_x, sine_y):
        self.componentGraph.graphPoints(x, y, interp_x, interp_y, sine_x, sine_y)

    def graphCombinedWave(self, t:np.ndarray, wave:np.ndarray):
        self.centralGraph.graphCombinedWave(t, wave)
        
    def createNewWaveformWindow(self):
        name = self.waveformNameInputWidget.getName()
        self.createCatalogWaveSignal.emit(name)

    def openCatalogAdditionDialog(self, key) -> bool:
        """ Attempt to get name for waveform. If failure, return False.
        """
        self.waveformNameInputWidget.setVisible(True)
        self.waveformNameInputWidget.focusInput()

    def addWaveToCatalog(self, keyIndex:int, name:str):
        self.catalogWidget.addWaveWidgetToCatalog(keyIndex, name)
        self.componentGraph.setKeyIndex(keyIndex)
        self.waveformNameInputWidget.closeWindowAndClearInput()

class CentralGraphWidget(QWidget):
    playSignal = Signal(int)
    durationChangedSignal = Signal(float)
    stopAudioSignal = Signal()

    def __init__(self):
        super().__init__()
        self.isPlaying = False
        self.gridLayout = QGridLayout(self)
        self.graphParametersWidget = GenericGraphParametersWidget()
        self.pen = pg.mkPen("#ff3300", width=3)

        self.graphParametersWidget.playSignal.connect(lambda e: self.playSignal.emit(0))
        self.window = pg.GraphicsLayoutWidget()
        self.graph = self.window.addPlot(title="Global Graph", row=0, col=0)
        self.graph.showGrid(True, True, .2)
        self.graph.setLimits(xMin=0, xMax=10, yMin=-10, yMax=10)
        self.mouseMovedProxy = pg.SignalProxy(self.graph.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

        self.vertical = pg.InfiniteLine(angle=90, movable=False, pen="#00ccff")
        self.horizontal = pg.InfiniteLine(angle=0, movable=False, pen="#00ccff")
        
        self.gridLayout.addWidget(self.graphParametersWidget, 0, 0)
        self.gridLayout.addWidget(self.window, 0, 1)

        self.graphParametersWidget.durationChangedSignal.connect(self.emitDurationChanged)
        self.graphParametersWidget.stopAudioSignal.connect(self.emitStopAudio)

    def graphCombinedWave(self, t:np.ndarray, wave:np.ndarray):
        self.clearGraph()
        self.graph.plot(t, wave, pen=self.pen)
        self.graph.setLimits(xMin=0, xMax=10, yMin=np.min(wave), yMax=np.max(wave))

    def mouseMoved(self, event):
        """
        Draw vertical and horizontal lines at the cursor position
        """
        pos = event[0]
        if self.graph.sceneBoundingRect().contains(pos):
            mousePoint = self.graph.vb.mapSceneToView(pos)
            self.vertical.setPos(mousePoint.x())
            self.horizontal.setPos(mousePoint.y())

    def clearGraph(self):
        self.graph.clear()
        self.graph.addItem(self.vertical)
        self.graph.addItem(self.horizontal)

    def emitStopAudio(self):
        self.stopAudioSignal.emit()

    def setDurationWidgetValue(self, duration: float):
        self.graphParametersWidget.setDurationWidgetValue(duration)

    def emitDurationChanged(self, duration):
        self.durationChangedSignal.emit(duration)

    def setStopTimer(self, duration:float):
        self.graphParametersWidget.setStopTimer(duration)


class ComponentGraphWidget(QWidget):
    pointAdditionSignal = Signal(int, float, float)
    playSignal = Signal(int)
    graphSignal = Signal(int)
    clearGraphSignal = Signal(int)
    regraphSignal = Signal(int)
    # Throw the index in as well 
    frequencyChangedSignal = Signal(int, str, int, int)
    sineStateChangedSignal = Signal(int, bool)
    sineCountChangedSignal = Signal(int, int)
    volumeUpdateSignal = Signal(int, float)
    stopAudioSignal = Signal()

    # If a component frequency or other parameters are adjusted
    componentChangedSignal = Signal(int) 

    def __init__(self):
        super().__init__()
        self.isPlaying = False
        self.keyIndex = 0
        self.seed = np.array([])
        self.points = []
        self.gridLayout = QGridLayout(self)
        self.window = pg.GraphicsLayoutWidget()
        #self.amplitudeEnvelope = self.window.addPlot(title="Amplitude Envelope", row=0, col=0)
        #self.frequencyEnvelope = self.window.addPlot(title="Frequency Envelope", row=1, col=0)
        self.oscillator = self.window.addPlot(title="Oscillator", row=2, col=0)
        #self.graph2 = self.window.addPlot(title="Frequency Env", row=0, col=0)
        self.oscillator.setXRange(0, 1)
        self.oscillator.setYRange(-1, 1)
        self.viewBox = self.oscillator.vb
        self.vertical = pg.InfiniteLine(angle=90, movable=False, pen="#00ccff")
        self.horizontal = pg.InfiniteLine(angle=0, movable=False, pen="#00ccff")
        self.oscillator.addItem(self.vertical)
        self.oscillator.addItem(self.horizontal)
        self.mouseMovedProxy = pg.SignalProxy(self.oscillator.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        # Prevent dragging
        self.oscillator.setLimits(xMin=0, xMax=1, yMin=-1, yMax=1)

        # Connect to self.addPoint
        self.mouseClickedProxy = pg.SignalProxy(self.oscillator.scene().sigMouseClicked, rateLimit=120, slot=self.addPoint)

        self.oscillator.showGrid(True, True, .5)

        self.plotPen = pg.mkPen("#0099bb", width=2)
        self.sineInterpolatedPen = pg.mkPen("#ff0000", width=2)
        self.linearInterpolatedPen = pg.mkPen("#ff00ff", width=2)

        self.scatterPen = pg.mkPen("#aa0000", width=2)

        self.graphParametersWidget = GraphParametersWidget()

        self.graphParametersWidget.regraphSignal.connect(self.graphPoints)
        self.graphParametersWidget.clearGraphSignal.connect(self.emitClearGraphSignal)
        self.graphParametersWidget.playSignal.connect(self.emitPlaySignal)
        self.graphParametersWidget.frequencyChangedSignal.connect(self.emitFrequencyParameters)
        self.graphParametersWidget.sineStateChangedSignal.connect(self.emitSineStateChanged)
        self.graphParametersWidget.sineCountChangedSignal.connect(self.emitSineCountChanged)
        self.graphParametersWidget.volumeUpdateSignal.connect(self.emitVolume)
        self.graphParametersWidget.stopAudioSignal.connect(self.emitStopAudio)

        self.gridLayout.addWidget(self.graphParametersWidget, 0, 0)
        self.gridLayout.addWidget(self.window, 0, 1)
        #self.gridLayout.setSpacing(0)
        #self.gridLayout.setContentsMargins(0, 0, 0, 0)

    def emitStopAudio(self):
        self.stopAudioSignal.emit()

    def getKeyIndex(self):
        return self.keyIndex

    def setStopTimer(self, duration:float):
        self.graphParametersWidget.setStopTimer(duration)

    def emitVolume(self, vol:float):
        self.volumeUpdateSignal.emit(self.keyIndex, vol)

    def emitSineCountChanged(self, count):
        self.sineCountChangedSignal.emit(self.keyIndex, count)

    def emitSineStateChanged(self, isChecked:bool):
        self.sineStateChangedSignal.emit(self.keyIndex, isChecked)

    def emitFrequencyParameters(self, baseFreq, cents, octave):
        self.frequencyChangedSignal.emit(self.keyIndex, baseFreq, cents, octave)

    def emitPlaySignal(self):
        self.playSignal.emit(self.keyIndex)

    def emitClearGraphSignal(self):
        self.clearGraph()
        self.clearGraphSignal.emit(self.keyIndex)

    def setKeyIndex(self, keyIndex):
        self.keyIndex = keyIndex

    def addPoint(self, event):
        click_event, = event
        scenePos = click_event.scenePos()
        if self.viewBox.sceneBoundingRect().contains(click_event.scenePos()):
            plotPosition = self.viewBox.mapSceneToView(scenePos)
            self.pointAdditionSignal.emit(self.keyIndex, plotPosition.x(), plotPosition.y())

    def clearGraph(self):
        self.oscillator.clear()
        self.oscillator.addItem(self.vertical)
        self.oscillator.addItem(self.horizontal)


    def plot(self, x, y, pen=None):
        self.oscillator.plot(x, y, pen=pen)

    def scatterPlot(self, x, y, pen=None):
        self.oscillator.scatterPlot(x, y, pen=pen)

    def graphPoints(self, x, y, interp_x, interp_y, sine_x, sine_y):
        self.clearGraph()
        #x, y = zip(*self.points)
        self.oscillator.plot(x, y, pen=self.plotPen)
        self.oscillator.scatterPlot(x, y, pen=self.scatterPen)
        self.oscillator.scatterPlot(interp_x, interp_y, pen=self.sineInterpolatedPen)
        self.oscillator.plot(interp_x, interp_y, pen=self.linearInterpolatedPen)
        self.oscillator.plot(sine_x, sine_y, pen=self.sineInterpolatedPen)

    def getVolume(self):
        return self.volumeSlider.getVolume()

    def getFrequency(self):
        return self.frequencyWidget.getFrequency()

    def getDuration(self):
        return self.durationSpin.value()

    def getPoints(self):
        return list(zip(*self.points))

    def setPoints(self, points):
        self.points = points

    def mouseMoved(self, event):
        """
        Draw vertical and horizontal lines at the cursor position
        """
        pos = event[0]
        if self.oscillator.sceneBoundingRect().contains(pos):
            mousePoint = self.oscillator.vb.mapSceneToView(pos)
            self.vertical.setPos(mousePoint.x())
            self.horizontal.setPos(mousePoint.y())

    def plot(self, t, waveform):
        hPen = pg.mkPen("#0099bb", width=3)
        self.oscillator.plot(t, waveform, pen=hPen)
        #self.oscillator.setMinimumWidth(self.graphWidget.height())

    def clear(self):
        self.oscillator.clear()


class GenericGraphParametersWidget(QWidget):
    # Contains the index of the waveform
    playSignal = Signal(int)
    durationChangedSignal = Signal(float)
    volumeUpdateSignal = Signal(float)
    stopAudioSignal = Signal()

    def __init__(self):
        super().__init__()
        self.isPlaying = False
        self.controlLayout = QFormLayout(self)
        self.parameterWidgetMaximumSize = QSize(350, 700)
        self.parameterWidgetMinimumSize = QSize(30, 30)

        self.volumeSlider = VolumeControlsWidget()

        self.playButton = QPushButton("Play")
        self.playButton.setMinimumSize(self.parameterWidgetMinimumSize)
        self.playButton.setMaximumSize(QSize(100, 100))
        #self.setMinimumSize(self.parameterWidgetMinimumSize)
        self.setMaximumSize(QSize(250, 800))

        self.durationSpin = QDoubleSpinBox()
        self.durationSpin.setRange(1, 10)
        self.durationSpin.setValue(3)
        self.durationSpin.setMaximumSize(self.parameterWidgetMaximumSize)
        self.durationSpin.valueChanged.connect(self.emitDurationChanged)

        self.playButton.clicked.connect(self.playWaveform)
        self.volumeSlider.volumeUpdateSignal.connect(self.emitVolume)

        self.controlLayout.addRow("Volume:", self.volumeSlider)
        self.controlLayout.addRow("Duration:", self.durationSpin)
        self.controlLayout.addRow("", self.playButton)
    
    def setDurationWidgetValue(self, duration: float):
        self.durationSpin.setValue(duration)

    def setStopTimer(self, duration:float):
        self.playTimer = QTimer()
        self.playTimer.setSingleShot(True)
        self.playTimer.timeout.connect(self.resetPlayButton)
        self.playTimer.start(int(round(duration * 1000)))

    def emitDurationChanged(self):
        self.durationChangedSignal.emit(self.durationSpin.value())

    def emitVolume(self, vol:float):
        self.volumeUpdateSignal.emit(vol)

    def playWaveform(self, event):
        if not self.isPlaying:
            self.playSignal.emit(0)
            self.isPlaying = True
            self.playButton.setText("Stop")
        else:
            self.stopAudioSignal.emit()
            self.resetPlayButton()

    def resetPlayButton(self):
        self.playButton.setText("Play")
        self.isPlaying = False

    def emitStopAudio(self):
        self.stopAudioSignal.emit()

class GraphParametersWidget(GenericGraphParametersWidget):
    regraphSignal = Signal()
    clearGraphSignal = Signal()
    frequencyChangedSignal = Signal(str, int, int)
    sineStateChangedSignal = Signal(bool)
    sineCountChangedSignal = Signal(int)

    def __init__(self):
        super().__init__()

        self.clearButton = QPushButton("Clear")
        self.clearButton.setMinimumSize(self.parameterWidgetMinimumSize)
        self.clearButton.setMaximumSize(self.parameterWidgetMaximumSize)
        self.clearButton.clicked.connect(self.clearGraphAndPoints)


        self.sineInterpolatorWidget = SineInterpolatorWidget()
        self.sineInterpolatorWidget.setMinimumSize(self.parameterWidgetMinimumSize)
        self.sineInterpolatorWidget.setMaximumSize(self.parameterWidgetMaximumSize)
        self.frequencyWidget = FrequencyWidget()

        self.controlLayout.addRow("Sine:", self.sineInterpolatorWidget)
        self.controlLayout.addRow("Freq:", self.frequencyWidget)
        self.controlLayout.addRow("", self.clearButton)
        self.controlLayout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        self.controlLayout.setSpacing(10)
        self.controlLayout.setContentsMargins(0, 0, 0, 0)

        self.controlLayout.removeRow(1)

        # Slots and Signals
        self.sineInterpolatorWidget.changedSignal.connect(self.emitSineStateChanged)
        self.sineInterpolatorWidget.sineCountChangedSignal.connect(self.emitSineCountChanged)
        self.frequencyWidget.frequencyChangedSignal.connect(self.emitFrequencyParameters)

    def emitSineCountChanged(self, count):
        self.sineCountChangedSignal.emit(count)

    def emitSineStateChanged(self, isChecked):
        self.sineStateChangedSignal.emit(isChecked)

    def emitFrequencyParameters(self, baseFreq, cents, octave):
        self.frequencyChangedSignal.emit(baseFreq, cents, octave)

    def graphPoints(self, event):
        self.clearGraphSignal.emit()
        self.regraphSignal.emit()

    def clearGraphAndPoints(self, event):
        self.clearGraphSignal.emit()

class VolumeControlsWidget(QWidget):
    volumeUpdateSignal = Signal(float)
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.volumeSlider = QSlider()
        self.volumeSlider.setMaximumSize(QSize(100, 100))
        self.volumeSlider.setOrientation(QtCore.Qt.Horizontal)
        self.volumeSlider.setRange(0, 10)
        self.volumeSlider.setValue(10)
        self.volumeSlider.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.layout.addWidget(self.volumeSlider)
        self.layout.addStretch()

        self.volumeSlider.sliderReleased.connect(self.emitVolume)

    def getVolume(self):
        return self.volumeSlider.value() / 10.0

    def emitVolume(self):
        self.volumeUpdateSignal.emit(self.getVolume())

class SineInterpolatorWidget(QWidget):
    changedSignal = Signal(bool)
    sineCountChangedSignal = Signal(int)

    def __init__(self):
        super().__init__()
        self.formLayout = QFormLayout(self)
        self.sinePlotCheckBox = QCheckBox()
        self.sinePlotCheckBox.setMaximumSize(QSize(100, 100))
        self.sinePlotCheckBox.setCheckState(QtCore.Qt.CheckState.Checked)
        self.sinePlotCheckBox.checkStateChanged.connect(self.stateChanged)

        self.sineCountSpin = QSpinBox()
        self.sineCountSpin.setRange(1, 50)
        self.sineCountSpin.setValue(13)
        self.sineCountSpin.setMinimumSize(QSize(100, 30))
        self.sineCountSpin.setMaximumSize(QSize(100, 100))
        #self.sineCountSpin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.sineCountSpin.valueChanged.connect(self.emitSineCountChanged)

        self.formLayout.addRow("Use sine interpolation:", self.sinePlotCheckBox)
        self.formLayout.addRow("Sine Count:", self.sineCountSpin)


    def stateChanged(self, event):
        self.changedSignal.emit(event == QtCore.Qt.CheckState.Checked)

    def emitSineCountChanged(self, value):
        self.sineCountChangedSignal.emit(value)

    def isChecked(self):
        return self.sinePlotCheckBox.isChecked()

    def sineCount(self):
        return self.sineCountSpin.value()
        

class FrequencyWidget(QWidget):
    """
    Upon changing any of the frequency combo box, cents, or octave spin boxes, emit a signal with the widget values.
    The new frequency should be recalculated outside of the widget
    """
    frequencyChangedSignal = Signal(str, int, int)
    def __init__(self):
        super().__init__()
        self.formLayout = QFormLayout(self)
        
        self.frequencySelector = QComboBox()
        self.frequencySelector.addItems(waveform.NOTE_LETTERS)
        self.frequencySelector.setMaximumSize(QSize(100, 100))
        self.frequencySelector.currentTextChanged.connect(self.emitFrequencyParameters)

        self.centsSpin = QSpinBox()
        self.centsSpin.setRange(-100, 100)
        self.centsSpin.setMaximumSize(QSize(100, 100))
        self.centsSpin.setValue(0)
        self.centsSpin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.centsSpin.valueChanged.connect(self.emitFrequencyParameters)

        self.octaveSpin = QSpinBox()
        self.octaveSpin.setRange(1, 7)
        self.octaveSpin.setValue(1)
        self.octaveSpin.setMaximumSize(QSize(100, 100))
        self.octaveSpin.valueChanged.connect(self.emitFrequencyParameters)

        self.formLayout.addRow("Note:", self.frequencySelector)
        self.formLayout.addRow("Cents:", self.centsSpin)
        self.formLayout.addRow("Octave:", self.octaveSpin)

    def emitFrequencyParameters(self):
        baseFreq = self.frequencySelector.currentText()
        cents = self.centsSpin.value()
        octave = self.octaveSpin.value()
        self.frequencyChangedSignal.emit(baseFreq, cents, octave)

    def getOctave(self):
        return self.octaveSpin.value()

    def getCents(self):
        return self.centsSpin.value()

    def getFrequency(self):
        cents = self.centsSpin.value()
        octave = self.octaveSpin.value()
        return waveform.NOTE_FREQUENCY_MAP[self.frequencySelector.currentText()] * 2 ** (cents / 1200) * 2 ** (octave - 1)


class LabeledWaveImageWidget(QFrame):
    def __init__(self, name: str):
        super().__init__()
        self.vboxLayout = QVBoxLayout(self)
        self.label = QLabel(name)
        self.image = QImage()
        self.vboxLayout.addWidget(self.label)
        #self.setStyleSheet("border: 1px solid blue")
        #self.vboxLayout.addWidget(self.image)

class WaveNameInputWidget(QWidget):
    createCatalogWaveSignal = Signal(str)
    cancelSignal = Signal(int)
    def __init__(self, maxWidth, maxHeight):
        super().__init__()
        closeWindowAction = QAction("&Close Window", self, shortcut="Esc", triggered=self.closeWindowAndClearInput)
        enterAction = QAction("&Accept Input", self, shortcut="Return", triggered=self.enterPressed)
        self.addAction(closeWindowAction)
        self.addAction(enterAction)
        self.maxWidth = maxWidth
        self.maxHeight = maxHeight
        self.move(self.maxWidth // 2, self.maxHeight // 2)
        self.formLayout = QFormLayout(self)
        self.userInput = QLineEdit()
        self.formLayout.addRow("Enter waveform name:", self.userInput)
        self.ok = QPushButton("Ok")
        self.cancel = QPushButton("Cancel")
        self.okCancelComplex = QWidget()
        self.errorLabel = QLabel()
        self.errorLabel.setStyleSheet("color: #ff0000")
        self.hboxLayout = QHBoxLayout(self.okCancelComplex)
        self.hboxLayout.addWidget(self.ok)
        self.hboxLayout.addWidget(self.cancel)
        self.formLayout.addRow(self.okCancelComplex)
        self.formLayout.addRow(self.errorLabel)
        self.ok.clicked.connect(self.okClicked)
        self.cancel.clicked.connect(lambda e: self.cancelSignal.emit(e))
    def enterPressed(self, event):
        if self.cancel.hasFocus():
            self.closeWindowAndClearInput()
        else:
            self.okClicked(event)
    def okClicked(self, event):
        if len(self.userInput.text()) == 0:
            self.errorLabel.setText("Need to supply a name for new waveform")
        else:
            name = self.userInput.text()
            self.createCatalogWaveSignal.emit(name)
    def deleteUserInput(self):
        self.userInput.setText("")
    def getName(self):
        return self.userInput.text()
    def closeWindowAndClearInput(self):
        self.clearTextFields()
        self.setVisible(False)
    def clearTextFields(self):
        self.userInput.setText("")
        self.errorLabel.setText("")
    def focusInput(self):
        self.userInput.setFocus()
    def displayDupNameErrMsg(self):
        self.errorLabel.setText("Name '{}' already exists in catalog".format(self.getName()))

class WaveformCatalogWidget(QWidget):
    """
    """
    initiateCatalogAdditionSignal = Signal(bool)
    swapGraphsSignal = Signal(int)
    switchToMainSignal = Signal()
    switchToComponentSignal = Signal()

    def __init__(self):
        super().__init__()
        self.vboxLayout = QVBoxLayout(self)
        self.vboxLayout.setSpacing(0)
        self.labeledWavesDict = {}
        self.catalogSection = QWidget()
        self.catalogVBox = QVBoxLayout(self.catalogSection)
        self.addWaveButton = QPushButton("+")
        self.addWaveButton.setMaximumSize(QSize(100, 100))
        self.toggleGraphButton = QPushButton("Component")

        self.vboxLayout.addWidget(self.toggleGraphButton)
        self.vboxLayout.addWidget(self.catalogSection)
        self.vboxLayout.addWidget(self.addWaveButton)
        self.vboxLayout.addStretch()

        # Signals 
        self.addWaveButton.clicked.connect(self.emitInititateAddCatalogWave)
        self.toggleGraphButton.clicked.connect(self.emitSwapGraphs)
    
    def emitInititateAddCatalogWave(self, event):
        self.initiateCatalogAdditionSignal.emit(event)

    def toggleMainButton(self):
        self.toggleGraphButton.setText("Main")

    def toggleComponentButton(self):
        self.toggleGraphButton.setText("Component")

    def addWaveWidgetToCatalog(self, keyIndex, name:str):
        self.labeledWavesDict[keyIndex] = {
            "name": name,
        }
        self.emitSwapGraphs(keyIndex)
        
    def emitSwapGraphs(self, keyIndex):
        self.swapGraphsSignal.emit(keyIndex)
