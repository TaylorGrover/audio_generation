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
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDial, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QMainWindow, QMessageBox, QPushButton, QSpinBox, QSizePolicy, QSlider, QToolBar, QWidget)
import waveform

import pyqtgraph as pg

import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        toolBar = QToolBar()
        self.addToolBar(toolBar)
        fileMenu = self.menuBar().addMenu("&File")
        self.populateFileMenu(fileMenu)
        editMenu = self.menuBar().addMenu("&Edit")
        viewMenu = self.menuBar().addMenu("&View")
        self.graphWidget = GraphWidget()
        self.setCentralWidget(self.graphWidget)

    def populateFileMenu(self, fileMenu):
        """
        File menu functions:
        * Save
        * Open
        """
        saveWaveAction = QAction("&Save Waveform", self, shortcut="Ctrl+S", triggered=self.saveWaveformTrigger)
        loadWaveAction = QAction("&Load Waveform", self, shortcut="Ctrl+O", triggered=self.loadWaveformTrigger)
        fileMenu.addAction(saveWaveAction)
        fileMenu.addAction(loadWaveAction)

    def saveWaveformTrigger(self):
        """
        Open a QFileDialog to save the waveform
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
    changedSignal = Signal(int)

    def __init__(self):
        super().__init__()
        self.formLayout = QFormLayout(self)
        self.sinePlotCheckBox = QCheckBox()
        self.sinePlotCheckBox.setMaximumSize(QSize(100, 100))
        self.sinePlotCheckBox.setCheckState(QtCore.Qt.CheckState.Checked)
        self.sinePlotCheckBox.checkStateChanged.connect(self.stateChanged)

        self.sineCountSpin = QSpinBox()
        self.sineCountSpin.setRange(1, 50)
        self.sineCountSpin.setValue(15)
        self.sineCountSpin.setMinimumSize(QSize(30, 30))
        self.sineCountSpin.setMaximumSize(QSize(100, 100))
        self.sineCountSpin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.sineCountSpin.valueChanged.connect(self.stateChanged)

        self.formLayout.addRow("Use sine interpolation:", self.sinePlotCheckBox)
        self.formLayout.addRow("Sine Count:", self.sineCountSpin)


    def stateChanged(self, event):
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
        self.octaveSpin.setValue(2)
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

class GraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.isPlaying = False

        self.seed = np.array([])
        self.points = []
        self.lines = []
        self.gridLayout = QGridLayout(self)
        lowerWidgetMaximumSize = QSize(100, 100)
        self.window = pg.GraphicsLayoutWidget()
        self.graph = self.window.addPlot(title="Grapher", row=1, col=0)
        self.graph.setXRange(0, 1)
        self.graph.setYRange(-1, 1)
        self.viewBox = self.graph.vb
        self.vertical = pg.InfiniteLine(angle=90, movable=False, pen="#00ccff")
        self.horizontal = pg.InfiniteLine(angle=0, movable=False, pen="#00ccff")
        self.graph.addItem(self.vertical)
        self.graph.addItem(self.horizontal)
        self.mouseMovedProxy = pg.SignalProxy(self.graph.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        # Prevent dragging
        self.graph.setLimits(xMin=0, xMax=1, yMin=-1, yMax=1)

        # Connect to self.addPoint
        self.mouseClickedProxy = pg.SignalProxy(self.graph.scene().sigMouseClicked, rateLimit=60, slot=self.addPoint)

        self.graph.showGrid(True, True, .5)

        self.plotPen = pg.mkPen("#0099bb", width=2)
        self.sineInterpolatedPen = pg.mkPen("#ff0000", width=2)
        self.linearInterpolatedPen = pg.mkPen("#ff00ff", width=2)

        self.scatterPen = pg.mkPen("#aa0000", width=2)
        self.graphButtonWidget = QWidget()

        #self.lowerButtonLayout = QHBoxLayout(self.graphButtonWidget)
        self.controlLayout = QFormLayout(self.graphButtonWidget)
        self.graphButtonWidget.setMaximumSize(QSize(300, 5000))

        self.playButton = QPushButton("Play")
        self.playButton.setMinimumSize(QSize(30, 30))
        self.playButton.setMaximumSize(lowerWidgetMaximumSize)
        self.playButton.clicked.connect(self.playWaveform)

        self.clearButton = QPushButton("Clear")
        self.clearButton.setMinimumSize(QSize(30, 30))
        self.clearButton.setMaximumSize(lowerWidgetMaximumSize)
        self.clearButton.clicked.connect(self.clearGraphAndPoints)

        self.volumeSlider = VolumeControlsWidget()

        self.durationSpin = QDoubleSpinBox()
        self.durationSpin.setRange(1, 10)
        self.durationSpin.setValue(3)
        self.durationSpin.setMaximumSize(lowerWidgetMaximumSize)

        self.sineInterpolatorWidget = SineInterpolatorWidget()
        self.sineInterpolatorWidget.changedSignal.connect(self.graphPoints)

        self.frequencyWidget = FrequencyWidget()

        # Left top right bottom
        #self.lowerButtonLayout.setContentsMargins(0, 0, 0, 20)
        # Needed to reduce spacing between widgets
        #self.lowerButtonLayout.addStretch()

        self.controlLayout.addRow("Volume:", self.volumeSlider)
        #self.controlLayout.addRow("Sine Interp:", self.sine
        self.controlLayout.addRow("", self.playButton)
        self.controlLayout.addRow("Sine:", self.sineInterpolatorWidget)
        self.controlLayout.addRow("Freq:", self.frequencyWidget)
        self.controlLayout.addRow("Duration:", self.durationSpin)
        self.controlLayout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        self.controlLayout.setSpacing(4)
        self.controlLayout.setContentsMargins(0, 0, 0, 0)

        self.gridLayout.addWidget(self.window, 0, 1)
        self.gridLayout.addWidget(self.graphButtonWidget, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)


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


    def get_zero_shifted_x(self, points):
        x, y = zip(*points)
        x_shifted = list(map(lambda t: t - x[0], x))
        return x_shifted


    def addPoint(self, event):
        click_event, = event
        scenePos = click_event.scenePos()
        if self.viewBox.sceneBoundingRect().contains(click_event.scenePos()):
            plotPosition = self.viewBox.mapSceneToView(scenePos)
            bisect.insort(self.points, [plotPosition.x(), plotPosition.y()], key=lambda t: t[0])
            self.graphPoints()
            #self.seed = self.calculateSeed(self.points)

    def clearGraphAndPoints(self):
        self.points = []
        self.clearGraph()

    def clearGraph(self):
        self.graph.clear()
        self.graph.addItem(self.vertical)
        self.graph.addItem(self.horizontal)


    def graphPoints(self):
        self.clearGraph()
        x, y = zip(*self.points)
        self.graph.plot(x, y, pen=self.plotPen)
        self.graph.scatterPlot(x, y, pen=self.scatterPen)
        new_x = np.linspace(x[0], x[-1], len(x))
        self.seed = np.interp(new_x, x, y)
        self.graph.scatterPlot(new_x, self.seed, pen=self.sineInterpolatedPen)
        self.graph.plot(new_x, self.seed, pen=self.linearInterpolatedPen)
        if self.sineInterpolatorWidget.isChecked():
            # Need to make evenly spaced xs 
            if len(new_x) >= 2:
                duration = new_x[-1] - new_x[0]
                amplitude = np.max(np.abs(self.seed))
                sine_count = self.sineInterpolatorWidget.sineCount()
                t = np.linspace(new_x[0], new_x[-1], int(duration * waveform.SAMPLE_RATE))
                wave = waveform.seeded_waveform(amplitude, duration, 1 / duration, self.seed, waveform.SAMPLE_RATE, sine_count)
                self.graph.plot(t, wave.T[0], pen=self.sineInterpolatedPen)

    def playWaveform(self):
        if self.isPlaying:
            self.effect.stop()
            self.resetPlayButton()
            return 
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
        if self.graph.sceneBoundingRect().contains(pos):
            mousePoint = self.graph.vb.mapSceneToView(pos)
            self.vertical.setPos(mousePoint.x())
            self.horizontal.setPos(mousePoint.y())

    def plot(self, t, waveform):
        hPen = pg.mkPen("#0099bb", width=3)
        self.graph.plot(t, waveform, pen=hPen)
        #self.graph.setMinimumWidth(self.graphWidget.height())

    def clear(self):
        self.graph.clear()


def start_gui():
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.setWindowTitle("Grapher Utility")
    availableGeometry = mainWindow.screen().availableGeometry()
    mainWindow.resize(availableGeometry.width(), availableGeometry.height())
    mainWindow.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    start_gui()
