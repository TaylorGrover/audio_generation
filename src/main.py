import numpy as np
import os
import soundfile as sf
import sys

import waveform

from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import QUrl, QSize
from PySide6.QtGui import QAction, QColor, QIcon, QPalette, QPixmap, QRegularExpressionValidator
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import (QApplication, QPushButton, QCheckBox, QComboBox,
    QHBoxLayout, QFormLayout, QGridLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
    QMessageBox, QSpinBox, QDoubleSpinBox, QSplitter, QTabWidget, QToolBar, QWidget, QDial)
    
import pyqtgraph as pg
from typing import List

SAMPLE_RATE = 44100
IMAGE_DIR = "images"
MAX_VOLUME = .25
DURATION = 3

# TODO: Do I need this regex?
lineEditValidator = QRegularExpressionValidator(r"([0-9]+(_?[0-9])*\.[0-9]*(_?[0-9])*)|([0-9]*(_?[0-9])*\.[0-9]+(_?[0-9])*)")

# Notes
LETTERS = [
    "A",  "A#", "B", "C", "C#", "D",
    "D#", "E", "F", "F#", "G", "G#"
]
COMPUTED_NOTES = [440 * 2 ** (i / 12) for i in range(12)]
NOTE_MAP = dict(zip(LETTERS, COMPUTED_NOTES))

class MainWindow(QMainWindow):
    """
    """
    def __init__(self):
        super().__init__()
        toolBar = QToolBar()
        self.addToolBar(toolBar)
        self.waveformWindow = WaveformWindow()
        fileMenu = self.menuBar().addMenu("&File")
        editMenu = self.menuBar().addMenu("&Edit")
        viewMenu = self.menuBar().addMenu("&View")

        insertSampleAction = QAction("&Add Sample", self, shortcut="Insert", triggered=self.insertSample)
        exitAction = QAction("&Exit", self, shortcut="Ctrl+Q", triggered=self.close)
        fullScreenAction = QAction("&Fullscreen", self, shortcut="F11", triggered=self.maximize)
        addTabAction = QAction("&AddTab", self, shortcut="Ctrl+A", triggered=self.addWave)
        deleteTabAction = QAction("&DeleteWaveform", self, shortcut="Ctrl+W", triggered=self.deleteWave)
        playSoundAction = QAction("&PlaySound", self, shortcut="space", triggered=self.waveformWindow.playWaveform)
        nextTabAction = QAction("&ForwardTab", self, shortcut="Ctrl+Tab", triggered=self.nextTab)
        prevTabAction = QAction("&BackwardTab", self, shortcut="Ctrl+Shift+Tab", triggered=self.prevTab)

        fileMenu.addAction(exitAction)
        viewMenu.addAction(fullScreenAction)
        self.waveformWindow.addAction(deleteTabAction)
        self.waveformWindow.addAction(playSoundAction)
        self.waveformWindow.addAction(addTabAction)
        self.waveformWindow.addAction(nextTabAction)
        self.waveformWindow.addAction(prevTabAction)

        self.setCentralWidget(self.waveformWindow)
        self.maximized = False
        self.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        self.createPalette()

    def createPalette(self):
        self.palette = QPalette()
        self.palette.setColor(QPalette.Window, QColor(53, 53, 53))
        self.palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        self.palette.setColor(QPalette.Base, QColor(25, 25, 25))
        self.palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        self.palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        self.palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        self.palette.setColor(QPalette.PlaceholderText, QColor(140, 140, 140))
        self.palette.setColor(QPalette.Text, QColor(255, 255, 255))
        self.palette.setColor(QPalette.Button, QColor(63, 63, 63))
        self.palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        self.palette.setColor(QPalette.BrightText, QColor(23, 23, 200))
        self.palette.setColor(QPalette.Link, QColor(42, 130, 218))
        self.palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        self.palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.palette.setColor(QPalette.Disabled, QPalette.WindowText,
                         QColor(127, 127, 127))
        self.palette.setColor(QPalette.Disabled, QPalette.Text,
                         QColor(127, 127, 127))
        self.palette.setColor(QPalette.Disabled, QPalette.ButtonText,
                         QColor(127, 127, 127))
        self.palette.setColor(QPalette.Disabled, QPalette.Highlight,
                         QColor(80, 80, 80))
        self.palette.setColor(QPalette.Disabled, QPalette.HighlightedText,
                         QColor(127, 127, 127))
        self.palette.setColor(QPalette.Disabled, QPalette.Base, QColor(49, 49, 49))
        self.setPalette(self.palette)

    def insertSample(self):
        """
        Accept WAV, MP3, or OGG
        """
        print("Insert")

    def maximize(self):
        if self.maximized:
            self.showNormal()
        else:
            self.showMaximized()
        self.maximized = not self.maximized
    def deleteWave(self):
        """
        Remove the currently selected wave
        """
        self.waveformWindow.deleteTab()
    def addWave(self):
        """
        """
        self.waveformWindow.addTab()
    def nextTab(self):
        self.waveformWindow.nextTab()

    def prevTab(self):
        self.waveformWindow.prevTab()


class WaveformWindow(QWidget):
    """
    Contains the graph to display a single period of the current waveform, and 
    a menu to add waveforms.
    """
    def __init__(self):
        super().__init__()
        self.tabCounter = 0
        self.playStarted = False
        self.gridLayout = QGridLayout(self)
        self.graphWidget = GraphWidget()
        self.configContainer = QTabWidget()
        #self.addWaveButton = QPushButton("Add Waveform")
        #self.addWaveButton.clicked.connect(self.createWaveTab)
        self.t = np.arange(0, DURATION, 1.0 / SAMPLE_RATE)
        firstWaveform = waveform.Waveform(1, DURATION, 0, 0, waveform.sine, SAMPLE_RATE)
        print(firstWaveform)
        self.waveforms = [firstWaveform]
        self.combined = np.sum(self.waveforms, axis=0) * MAX_VOLUME
        print(self.combined)
        self.soundPlayer = waveform.SoundPlayer(self.combined, sr=SAMPLE_RATE)
        self.addTab()
        self.plot()
        splitter = QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.graphWidget)
        splitter.addWidget(self.configContainer)
        self.gridLayout.addWidget(splitter, 0, 0)

    def playWaveform(self):
        """
        Implement
        """
        if not self.playStarted:
            self.soundPlayer.play()
            self.playStarted = True
        else:
            self.playStarted = False
            self.soundPlayer.stop()

    def plot(self):
        """
        Call plot on graphWidget
        """
        self.plot_item = self.graphWidget.plot(self.t, self.combined)

    def updateWaveform(self):
        """
        """
        waveforms = self.getWaves()
        self.combined = np.sum(waveforms, axis=0)
        self.combined /= np.max(np.abs(self.combined))
        self.graphWidget.clear()
        self.plot()
        self.soundPlayer.updateWaveform(self.combined, sr=SAMPLE_RATE)
        if self.playStarted:
            self.soundPlayer.stop()
            self.soundPlayer.play()

    def getWaves(self) -> List[waveform.Waveform]:
        """
        """
        self.configs = [self.configContainer.widget(i) for i in range(self.configContainer.count())]
        waveforms = []
        for config in self.configs:
            frequency = 2 ** (config.getOctave() - 4) * 2 ** (config.getCents() / 1200) * NOTE_MAP[config.getNote()]
            phase = config.getPhase()
            waveforms.append(waveform.Waveform(config.getAmplitude(), DURATION, frequency, phase, waveform.STRING_TO_FORM_MAP[config.getForm()]))
        self.waveforms = waveforms
        return self.waveforms

    def addTab(self):
        """
        """
        wcw = WaveformConfigWindow()
        wcw.paramsChanged.connect(self.updateWaveform)
        self.configContainer.addTab(wcw, str(self.tabCounter + 1))
        self.tabCounter += 1
        self.updateWaveform()
        self.configContainer.setCurrentIndex(self.configContainer.count() - 1)

    def deleteTab(self):
        """
        Remove the tab
        """
        if self.configContainer.count() <= 1:
            # Don't remove current tab
            None
        else:
            self.configContainer.removeTab(self.configContainer.currentIndex())
            # Update waveform
            self.updateWaveform()

    def nextTab(self):
        count = self.configContainer.count()
        if count > 0:
            index = self.configContainer.currentIndex()
            index = (index + 1) % count
            self.configContainer.setCurrentIndex(index)
    def prevTab(self):
        count = self.configContainer.count()
        if count > 0:
            index = self.configContainer.currentIndex()
            index = (index - 1) % count
            self.configContainer.setCurrentIndex(index)


class WaveformConfigWindow(QWidget):
    """
    paramsChanged signal emitted to parent window (WaveformWindow)
    """
    paramsChanged = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.formLayout = QFormLayout(self)
        self.formLayout.setVerticalSpacing(3)

        self.ampSpinBox = QDoubleSpinBox()
        self.octaveSpinBox = QSpinBox()
        self.phaseSpinBox = QDoubleSpinBox()
        self.centsSpinBox = QSpinBox()

        self.noteCombo = QComboBox()
        self.noteCombo.addItems(LETTERS)

        self.formCombo = QComboBox()
        self.formCombo.addItems(list(waveform.STRING_TO_FORM_MAP.keys()))

        # Set minimum, maximum, and interval values for the parameter widgets
        self.ampSpinBox.setRange(1, 100)
        self.octaveSpinBox.setRange(1, 9)
        self.octaveSpinBox.setValue(3)
        self.phaseSpinBox.setRange(0, 20000)
        self.centsSpinBox.setRange(-99, 99)

        self.formLayout.addRow("Amplitude:", self.ampSpinBox)
        self.formLayout.addRow("Octave:", self.octaveSpinBox)
        self.formLayout.addRow("Phase Angle:", self.phaseSpinBox)
        self.formLayout.addRow("Note:", self.noteCombo)
        self.formLayout.addRow("Cents:", self.centsSpinBox)
        self.formLayout.addRow("Waveform:", self.formCombo)

        # Connect widgets to paramsChanged signal
        self.ampSpinBox.valueChanged.connect(self.onParamsChanged)
        self.octaveSpinBox.valueChanged.connect(self.onParamsChanged)
        self.phaseSpinBox.valueChanged.connect(self.onParamsChanged)
        self.centsSpinBox.valueChanged.connect(self.onParamsChanged)
        self.formCombo.currentIndexChanged.connect(self.onParamsChanged)
        self.noteCombo.currentIndexChanged.connect(self.onParamsChanged)

    def onParamsChanged(self):
        self.paramsChanged.emit()
    
    def getAmplitude(self) -> float:
        return self.ampSpinBox.value()
    def getOctave(self) -> int:
        return self.octaveSpinBox.value()
    def getPhase(self) -> float:
        return self.phaseSpinBox.value()
    def getNote(self) -> str:
        return self.noteCombo.currentText()
    def getCents(self) -> float:
        return self.centsSpinBox.value()
    def getForm(self) -> str:
        return self.formCombo.currentText()


class GraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.gridLayout = QGridLayout(self)
        self.graph = pg.PlotWidget()
        self.graphButtonWidget = QWidget()
        self.buttonLayout = QHBoxLayout(self.graphButtonWidget)
        self.playButton = QPushButton("Play")
        self.playButton.setMinimumSize(QSize(30, 30))
        self.playButton.setMaximumSize(QSize(100, 100))
        self.durationDial = QDial()
        self.buttonLayout.addWidget(self.playButton)
        #self.buttonLayout.addWidget(self.durationDial)
        self.gridLayout.addWidget(self.graph, 0, 0)
        self.gridLayout.addWidget(self.graphButtonWidget, 1, 0)

    def plot(self, t, waveform):
        hPen = pg.mkPen("#0099bb", width=3)
        fPen = pg.mkPen("r", width=3)
        self.graph.plot(t, waveform, pen=hPen)
        #self.graph.setMinimumWidth(self.graphWidget.height())

    def clear(self):
        self.graph.clear()



def start_gui():
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.setWindowTitle("Waveform Constructor")
    availableGeometry = mainWin.screen().availableGeometry()
    mainWin.resize(availableGeometry.width(), availableGeometry.height())
    mainWin.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    start_gui()
