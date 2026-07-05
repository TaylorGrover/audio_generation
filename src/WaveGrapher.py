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

    initiateCatalogAdditionSignal = Signal(int)
    createCatalogWaveSignal = Signal(str)
    playSignal = Signal(bool)
    graphSignal = Signal(int) # Emits the key for the specific graph to redraw
    pointAdditionSignal = Signal(int, float, float)

    def __init__(self):
        super().__init__()
        
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

        # Signals and slots
        self.workspaceWidget.initiateCatalogAdditionSignal.connect(self.emitCatalogWaveAdded)
        self.workspaceWidget.createCatalogWaveSignal.connect(lambda name: self.createCatalogWaveSignal.emit(name))
        self.workspaceWidget.playSignal.connect(self.emitPlaySignal)
        self.workspaceWidget.pointAdditionSignal.connect(self.emitPointAdditionSignal)
    
    def addWaveToCatalog(self, key_index:int, name:str):
        self.workspaceWidget.addWaveToCatalog(key_index, name)

    def displayDupNameErrMsg(self):
        self.workspaceWidget.displayDupNameErrMsg()
    
    def closeWaveNameInputWidget(self):
        self.workspaceWidget.closeWaveNameInputWidget()

    def emitCatalogWaveAdded(self, event):
        self.initiateCatalogAdditionSignal.emit(event)

    def emitGraphSignal(self, key):
        self.graphSignal.emit(key)

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

    def emitPlaySignal(self, event):
        self.playSignal.emit(event)
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

    def graphComponentWaveform(self, key, points):
        self.workspaceWidget.graphComponentWaveform(key, points)

    def openCatalogAdditionDialog(self, key) -> bool:
        self.workspaceWidget.openCatalogAdditionDialog(key)

    def closeEvent(self, event):
        """
        Ensure all windows are closed on the close event
        """
        QApplication.closeAllWindows()
        event.accept()

class VolumeControlsWidget(QWidget):
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

    def getVolume(self):
        return self.volumeSlider.value() / 10.0

class SineInterpolatorWidget(QWidget):
    changedSignal = Signal(QtCore.Qt.CheckState)

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
        self.sineCountSpin.setMinimumSize(QSize(30, 30))
        self.sineCountSpin.setMaximumSize(QSize(100, 100))
        self.sineCountSpin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.sineCountSpin.valueChanged.connect(self.stateChanged)

        self.formLayout.addRow("Use sine interpolation:", self.sinePlotCheckBox)
        self.formLayout.addRow("Sine Count:", self.sineCountSpin)


    def stateChanged(self, event):
        print("State changed: ", event)
        print(type(event))
        self.changedSignal.emit(event)

    def isChecked(self):
        return self.sinePlotCheckBox.isChecked()

    def sineCount(self):
        return self.sineCountSpin.value()
        

class FrequencyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.formLayout = QFormLayout(self)
        
        self.frequencySelector = QComboBox()
        self.frequencySelector.addItems(waveform.NOTE_LETTERS)
        self.frequencySelector.setMaximumSize(QSize(100, 100))

        self.centsSpin = QSpinBox()
        self.centsSpin.setRange(-100, 100)
        self.centsSpin.setMaximumSize(QSize(100, 100))
        self.centsSpin.setValue(0)
        self.centsSpin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.octaveSpin = QSpinBox()
        self.octaveSpin.setRange(1, 7)
        self.octaveSpin.setValue(1)
        self.octaveSpin.setMaximumSize(QSize(100, 100))

        self.formLayout.addRow("Note:", self.frequencySelector)
        self.formLayout.addRow("Cents:", self.centsSpin)
        self.formLayout.addRow("Octave:", self.octaveSpin)

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


class WorkspaceWidget(QWidget):
    """
    This is the main widget containing the main graph view 
    (for editors and the summed signal)
    """
    initiateCatalogAdditionSignal = Signal(int)
    createCatalogWaveSignal = Signal(str)
    swapGraphViewSignal = Signal(int)
    
    playSignal = Signal(bool)
    graphSignal = Signal(int)
    pointAdditionSignal = Signal(int, float, float)

    def __init__(self, maxWidth, maxHeight):
        super().__init__()
        self.maxWidth = maxWidth
        self.maxHeight = maxHeight
        self.waveformNameInputWidget = WaveNameInputWidget(maxWidth, maxHeight)
        self.waveformNameInputWidget.createCatalogWaveSignal.connect(self.createNewWaveformWindow)
        self.waveformNameInputWidget.move(self.maxWidth//2, self.maxHeight//2)
        self.waveformWidgetDict = {}
        self.gridLayout = QGridLayout(self)
        self.componentGraph = ComponentGraphWidget(0)
        self.centralGraph = CentralGraphWidget()

        self.centralGraph.playSignal.connect(self.emitPlaySignal)
        self.componentGraph.pointAdditionSignal.connect(self.emitPointAdditionSignal)

        self.catalogWidget = WaveformCatalogWidget()
        self.gridLayout.addWidget(self.catalogWidget, 0, 0)
        self.gridLayout.addWidget(self.centralGraph, 0, 1)

        # Signal management
        self.catalogWidget.initiateCatalogAdditionSignal.connect(self.emitCatalogWaveAddInitiate)
    def showComponentGraph(self, keyIndex:int):
        # Check if the centralGraph is currently in place
        item = self.gridLayout.itemAtPosition(0, 1)
        widget = item.widget()
        print("showCOmponentGraph")
        if widget == self.centralGraph:
            print("Yes")
            self.gridLayout.removeWidget(widget)
            widget.hide()
            self.componentGraph.setKeyIndex(keyIndex)
            self.gridLayout.addWidget(self.componentGraph, 0, 1)
    def displayDupNameErrMsg(self):
        self.waveformNameInputWidget.displayDupNameErrMsg()

    def closeWaveNameInputWidget(self):
        self.waveformNameInputWidget.closeWindowAndClearInput()

    def emitCatalogWaveAddInitiate(self, event):
        self.initiateCatalogAdditionSignal.emit(event)

    def emitPlaySignal(self, event):
        self.playSignal.emit(event)

    def emitGraphSignal(self, key):
        self.graphSignal.emit(key)

    def emitPointAdditionSignal(self, key, x, y):
        self.pointAdditionSignal.emit(key, x, y)

    def graphComponentWaveform(self, key, points):
        if key in self.waveformWidgetDict:
            self.waveformWidgetDict[key].graph(points)
    def createNewWaveformWindow(self):
        name = self.waveformNameInputWidget.getName()
        self.createCatalogWaveSignal.emit(name)

    def openCatalogAdditionDialog(self, key) -> bool:
        """ Attempt to get name for waveform. If failure, return False.
        """
        self.waveformNameInputWidget.setVisible(True)
        self.waveformNameInputWidget.focusInput()

    def addWaveToCatalog(self, key_index:int, name:str):
        self.catalogWidget.addWaveWidgetToCatalog(key_index, name)
        self.waveformNameInputWidget.closeWindowAndClearInput()

class WaveformCatalogWidget(QWidget):
    """
    """
    initiateCatalogAdditionSignal = Signal(bool)
    swapGraphSignal = Signal(int)

    def __init__(self):
        super().__init__()
        self.vboxLayout = QVBoxLayout(self)
        self.vboxLayout.setSpacing(0)
        self.labeledWavesDict = {}
        self.catalogSection = QWidget()
        self.catalogVBox = QVBoxLayout(self.catalogSection)
        self.addWaveButton = QPushButton("+")
        self.addWaveButton.setMaximumSize(QSize(100, 100))
        self.addWaveButton.clicked.connect(self.emitInititateAddCatalogWave)
        self.vboxLayout.addWidget(self.catalogSection)
        self.vboxLayout.addWidget(self.addWaveButton)
        self.vboxLayout.addStretch()

    def emitInititateAddCatalogWave(self, event):
        self.initiateCatalogAdditionSignal.emit(event)

    def addWaveWidgetToCatalog(self, key_index, name:str):
        print("add wave to catalog")
        self.labeledWavesDict[key_index] = {
            "name": name,
        }
        self.emitSwapGraph(key_index)
        
    def emitSwapGraph(self, key_index):
        self.swapGraphSignal.emit(key_index)

class GenericGraphParametersWidget(QWidget):
    playSignal = Signal(bool)

    def __init__(self):
        super().__init__()
        self.controlLayout = QFormLayout(self)
        self.parameterWidgetMaximumSize = QSize(350, 700)
        self.parameterWidgetMinimumSize = QSize(30, 30)

        self.volumeSlider = VolumeControlsWidget()

        self.playButton = QPushButton("Play")
        self.playButton.setMinimumSize(self.parameterWidgetMinimumSize)
        self.playButton.setMaximumSize(QSize(100, 100))
        #self.setMinimumSize(self.parameterWidgetMinimumSize)
        self.setMaximumSize(QSize(250, 800))

        self.playButton.clicked.connect(self.playWaveform)

        self.controlLayout.addRow("Volume:", self.volumeSlider)
        self.controlLayout.addRow("", self.playButton)

    def playWaveform(self, event):
        self.playSignal.emit(event)

class GraphParametersWidget(GenericGraphParametersWidget):
    regraphSignal = Signal()
    def __init__(self):
        super().__init__()

        self.clearButton = QPushButton("Clear")
        self.clearButton.setMinimumSize(self.parameterWidgetMinimumSize)
        self.clearButton.setMaximumSize(self.parameterWidgetMaximumSize)
        self.clearButton.clicked.connect(self.clearGraphAndPoints)

        self.durationSpin = QDoubleSpinBox()
        self.durationSpin.setRange(1, 10)
        self.durationSpin.setValue(3)
        self.durationSpin.setMaximumSize(self.parameterWidgetMaximumSize)

        self.sineInterpolatorWidget = SineInterpolatorWidget()
        self.sineInterpolatorWidget.changedSignal.connect(self.graphPoints)

        self.frequencyWidget = FrequencyWidget()

        self.controlLayout.addRow("Sine:", self.sineInterpolatorWidget)
        self.controlLayout.addRow("Freq:", self.frequencyWidget)
        self.controlLayout.addRow("Duration:", self.durationSpin)
        self.controlLayout.addRow("", self.clearButton)
        self.controlLayout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        self.controlLayout.setSpacing(4)
        self.controlLayout.setContentsMargins(0, 0, 0, 0)

    def graphPoints(self, event):
        print(event)
        self.regraphSignal()

    def clearGraphAndPoints(self, event):
        print(event)
        print(type(event))

class CentralGraphWidget(QWidget):
    playSignal = Signal(bool)

    def __init__(self):
        super().__init__()
        self.isPlaying = False
        self.gridLayout = QGridLayout(self)
        self.graphParametersWidget = GenericGraphParametersWidget()

        self.graphParametersWidget.playSignal.connect(lambda e: self.playSignal.emit(e))
        self.window = pg.GraphicsLayoutWidget()
        self.graph = self.window.addPlot(title="Global Graph", row=0, col=0)

        self.vertical = pg.InfiniteLine(angle=90, movable=False, pen="#00ccff")
        self.horizontal = pg.InfiniteLine(angle=0, movable=False, pen="#00ccff")
        
        self.gridLayout.addWidget(self.graphParametersWidget, 0, 0)
        self.gridLayout.addWidget(self.window, 0, 1)

class ComponentGraphWidget(QWidget):
    pointAdditionSignal = Signal(int, float, float)
    playSignal = Signal(bool)
    graphSignal = Signal(int)

    def __init__(self, keyIndex:int):
        super().__init__()
        self.keyIndex = keyIndex # For tracking the widgets in the controller
        self.isPlaying = False

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

        self.gridLayout.addWidget(self.graphParametersWidget, 0, 0)
        self.gridLayout.addWidget(self.window, 0, 1)
        #self.gridLayout.setSpacing(0)
        #self.gridLayout.setContentsMargins(0, 0, 0, 0)

    def calculateSeed(self, points):
        """
        ASSUMPTION: points is an unzippable nx2 matrix sorted by the left column
        """
        if len(points) < 2:
            return np.array([])
        x, y = zip(*points)
        new_x = np.linspace(x[0], x[-1], len(x))
        seed = np.interp(new_x, x, y)
        return seed 
    def setKeyIndex(self, keyIndex):
        self.keyIndex = keyIndex

    def get_zero_shifted_x(self, points):
        x, y = zip(*points)
        x_shifted = list(map(lambda t: t - x[0], x))
        return x_shifted


    def addPoint(self, event):
        click_event, = event
        scenePos = click_event.scenePos()
        if self.viewBox.sceneBoundingRect().contains(click_event.scenePos()):
            plotPosition = self.viewBox.mapSceneToView(scenePos)
            self.pointAdditionSignal.emit(self.keyIndex, plotPosition.x(), plotPosition.y())
            #bisect.insort(self.points, [plotPosition.x(), plotPosition.y()], key=lambda t: t[0])
            #self.graphPoints()
            #self.seed = self.calculateSeed(self.points)

    def clearGraphAndPoints(self):
        self.points = []
        self.clearGraph()

    def clearGraph(self):
        self.oscillator.clear()
        self.oscillator.addItem(self.vertical)
        self.oscillator.addItem(self.horizontal)


    def plot(self, x, y, pen=None):
        self.oscillator.plot(x, y, pen=pen)

    def scatterPlot(self, x, y, pen=None):
        self.oscillator.scatterPlot(x, y, pen=pen)

    def graphPoints(self):
        self.graphSignal.emit(self.keyIndex)
        self.clearGraph()
        x, y = zip(*self.points)
        self.oscillator.plot(x, y, pen=self.plotPen)
        self.oscillator.scatterPlot(x, y, pen=self.scatterPen)
        new_x = np.linspace(x[0], x[-1], 2 * len(x))
        self.seed = np.interp(new_x, x, y)
        self.oscillator.scatterPlot(new_x, self.seed, pen=self.sineInterpolatedPen)
        self.oscillator.plot(new_x, self.seed, pen=self.linearInterpolatedPen)
        if self.sineInterpolatorWidget.isChecked():
            # Need to make evenly spaced xs 
            if len(x) >= 2:
                duration = new_x[-1] - new_x[0]
                amplitude = np.max(np.abs(self.seed))
                sine_count = self.sineInterpolatorWidget.sineCount()
                t = np.linspace(new_x[0], new_x[-1], int(duration * waveform.SAMPLE_RATE))
                wave = waveform.seeded_waveform(amplitude, duration, 1 / duration, self.seed, waveform.SAMPLE_RATE, sine_count)
                self.oscillator.plot(t, wave.T[0], pen=self.sineInterpolatedPen)

    def playWaveform(self):
        if self.isPlaying:
            self.effect.stop()
        if len(self.points) < 2:
            return
        frequency = self.getFrequency()
        duration = self.getDuration()
        volume = self.getVolume()
        if self.sineInterpolatorWidget.isChecked():
            sine_count = self.sineInterpolatorWidget.sineCount()
            self.wave = volume * waveform.seeded_waveform(1, duration, frequency, self.seed, waveform.SAMPLE_RATE, sine_count)
        else:
            x, y = zip(*self.points)
            x = np.array(x)
            y = np.array(y)
            wavelength_sample_count = int(waveform.SAMPLE_RATE / frequency)
            t_base = np.linspace(0, x[-1] - x[0], wavelength_sample_count)
            duration_sample_count = int(waveform.SAMPLE_RATE * duration)
            t_indices = np.linspace(0, duration_sample_count, duration_sample_count).astype(int)
            t_modulo = t_indices % wavelength_sample_count
            y_interp = np.interp(t_base, x, y)
            y_interp /= np.max(np.abs(y_interp), axis=0)
            self.wave = volume * y_interp[t_modulo]

        self.effect = waveform.play(self.wave)
        self.effect.play()
        self.playTimer = QTimer()
        self.playTimer.timeout.connect(self.resetPlayButton)
        self.playTimer.start(int(duration * 1000))
        self.playButton.setText("Stop")
        self.isPlaying = True

    def resetPlayButton(self):
        self.playButton.setText("Play")
        self.isPlaying = False

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