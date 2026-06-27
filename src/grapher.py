import bisect
import json
import numpy as np
import os
from PySide6 import QtCore
from PySide6.QtCore import QUrl, QSize
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QDial, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QMainWindow, QMessageBox, QPushButton, QSpinBox, QToolBar, QWidget)
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
        fileMenu.addAction(saveWaveAction)

    def saveWaveformTrigger(self):
        """
        Open a QFileDialog to save the waveform
        """
        filePath, fileType = QFileDialog.getSaveFileName(
            self
            , "Open a File:"
            , os.getcwd()
            , "JSON Files (*.json);;"
        )
        if not filePath.lower().endswith(".json"):
            filePath = filePath + ".json"
        x, y = self.graphWidget.getPoints()
        with open(filePath, "w") as f:
            json.dump({"x": x, "y": y}, f)



class GraphWidget(QWidget):
    def __init__(self):
        super().__init__()
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
        self.interpolatedPen = pg.mkPen("#ff0000", width=2)
        self.scatterPen = pg.mkPen("#aa0000", width=2)
        self.graphButtonWidget = QWidget()
        self.buttonLayout = QHBoxLayout(self.graphButtonWidget)
        self.playButton = QPushButton("Play")
        self.clearButton = QPushButton("Clear")
        self.clearButton.setMinimumSize(QSize(30, 30))
        self.clearButton.setMaximumSize(lowerWidgetMaximumSize)
        self.clearButton.clicked.connect(self.clearGraphAndPoints)
        self.playButton.setMinimumSize(QSize(30, 30))
        self.playButton.setMaximumSize(lowerWidgetMaximumSize)
        self.playButton.clicked.connect(self.playWaveform)
        self.interpolateButton = QPushButton("Interp")
        self.interpolateButton.setMinimumSize(QSize(30, 30))
        self.interpolateButton.setMaximumSize(lowerWidgetMaximumSize)
        self.sineCountSpin = QSpinBox()
        self.sineCountSpin.setRange(1, 20)
        self.sineCountSpin.setValue(15)
        self.sineCountSpin.setMinimumSize(QSize(30, 30))
        self.sineCountSpin.setMaximumSize(lowerWidgetMaximumSize)
        self.sineCountSpin.valueChanged.connect(self.graphPoints)
        self.frequencySpin = QDoubleSpinBox()
        self.frequencySpin.setRange(30, 4000)
        self.frequencySpin.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self.frequencySpin.setDecimals(8)
        self.frequencySpin.valueChanged.connect(self.adjustFrequency)
        self.frequencySpin.setValue(220 * 2 ** (-8/12))
        self.frequencySpin.setMaximumSize(lowerWidgetMaximumSize)
        self.durationSpin = QDoubleSpinBox()
        self.durationSpin.setRange(1, 10)
        self.durationSpin.setValue(3)
        self.sinePlotCheckBox = QCheckBox("Plot Sine")
        self.sinePlotCheckBox.setMaximumSize(lowerWidgetMaximumSize)
        self.sinePlotCheckBox.setCheckState(QtCore.Qt.CheckState.Checked)
        self.buttonLayout.addWidget(self.playButton)
        self.buttonLayout.addWidget(self.clearButton)
        self.buttonLayout.addWidget(self.sineCountSpin)
        self.buttonLayout.addWidget(self.sinePlotCheckBox)
        self.buttonLayout.addWidget(self.frequencySpin)
        self.buttonLayout.addWidget(self.durationSpin)
        self.gridLayout.addWidget(self.window, 0, 0)
        self.gridLayout.addWidget(self.graphButtonWidget, 1, 0)


    def adjustFrequency(self, event):
        pass


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
        if self.sinePlotCheckBox.isChecked():
            # Need to make evenly spaced xs 
            new_x = np.linspace(x[0], x[-1], len(x))
            self.seed = np.interp(new_x, x, y)
            self.graph.scatterPlot(new_x, self.seed, pen=self.interpolatedPen)
            if len(new_x) >= 2:
                duration = new_x[-1] - new_x[0]
                amplitude = np.max(np.abs(self.seed))
                sine_count = self.sineCountSpin.value()
                t = np.linspace(new_x[0], new_x[-1], int(duration * waveform.SAMPLE_RATE))
                wave = waveform.seeded_waveform(amplitude, duration, 1 / duration, self.seed, waveform.SAMPLE_RATE, sine_count)
                self.graph.plot(t, wave.T[0], pen=self.interpolatedPen)

    def playWaveform(self):
        if len(self.points) < 2:
            return
        frequency = self.getFrequency()
        duration = self.getDuration()
        if self.sinePlotCheckBox.isChecked():
            sine_count = self.sineCountSpin.value()
            self.wave = waveform.seeded_waveform(1, duration, frequency, self.seed, waveform.SAMPLE_RATE, sine_count)
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
            self.wave = y_interp[t_modulo]

        self.effect = waveform.play(self.wave)
        self.effect.play()

    def getFrequency(self):
        return self.frequencySpin.value()

    def getDuration(self):
        return self.durationSpin.value()

    def getPoints(self):
        return list(zip(*self.points))

    def mouseMoved(self, event):
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
