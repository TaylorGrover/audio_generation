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
    QMessageBox, QSplitter, QTabWidget, QToolBar, QWidget, QDial)
    
import pyqtgraph as pg
from typing import List

SAMPLE_RATE = 44100
IMAGE_DIR = "images"

lineEditValidator = QRegularExpressionValidator(r"([0-9]+(_?[0-9])*\.[0-9]*(_?[0-9])*)|([0-9]*(_?[0-9])*\.[0-9]+(_?[0-9])*)")

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
        self.maximized = True
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
        self.t = np.arange(0, 1, 1.0 / SAMPLE_RATE)
        firstWaveform = waveform.Waveform(.25, 1, 440, 0, waveform.sine, SAMPLE_RATE)
        self.waveforms = [firstWaveform]
        self.addTab()
        self.plot()
        splitter = QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.graphWidget)
        splitter.addWidget(self.configContainer)
        self.gridLayout.addWidget(splitter, 0, 0)
        self.soundPlayer = waveform.SoundPlayer(np.sum(self.waveforms, axis=0), sr=SAMPLE_RATE)

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
        self.graphWidget.plot(self.t, self.waveforms)

    def getCombinedWaves(self):
        """
        Get the sum of the waves from each of the tabs
        TODO
        """

    def getWaves(self) -> List[waveform.Waveform]:
        """
        """
        self.configs = [self.configContainer.widget(i) for i in range(self.configContainer.count())]
        

    def addTab(self):
        """
        """
        self.configContainer.addTab(WaveformConfigWindow(), str(self.tabCounter + 1))
        self.tabCounter += 1

    def deleteTab(self):
        """
        Remove the tab
        """
        if self.configContainer.count() <= 1:
            # Don't remove current tab
            None
        else:
            self.configContainer.removeTab(self.configContainer.currentIndex())

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
    def __init__(self):
        super().__init__()
        self.formLayout = QFormLayout(self)
        self.ampDial, self.ampEdit, ampWidget = self._createHBoxRow("Amplitude")
        self.freqDial, self.freqEdit, freqWidget = self._createHBoxRow("Frequency")
        self.phaseDial, self.phaseEdit, phaseWidget = self._createHBoxRow("Phase Shift")
        self.formLayout.setVerticalSpacing(1)

        # Set minimum, maximum, and interval values for the parameter widgets
        self.ampEdit.setValidator(lineEditValidator)
        self.freqEdit.setValidator(lineEditValidator)
        self.phaseEdit.setValidator(lineEditValidator)

        self.ampEdit.setText("1")
        self.ampEdit.setMaxLength(6)
        self.freqEdit.setText("440")
        self.freqEdit.setMaxLength(8)
        self.phaseEdit.setText("0")


        self.formLayout.addWidget(ampWidget)
        self.formLayout.addWidget(freqWidget)
        self.formLayout.addWidget(phaseWidget)
        #self.

    def _createHBoxRow(self, text):
        qwidget = QWidget()
        hBoxLayout = QHBoxLayout(qwidget)
        text = QLabel(text)
        dial = QDial()
        dial.setWrapping(False)
        
        lineEdit = QLineEdit()
        hBoxLayout.addWidget(text)
        hBoxLayout.addWidget(dial)
        hBoxLayout.addWidget(lineEdit)
        return dial, lineEdit, qwidget



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

    def plot(self, t, waveforms):
        hPen = pg.mkPen("#0099bb", width=3)
        fPen = pg.mkPen("r", width=3)
        self.graph.plot(t, np.sum(waveforms, axis=0), pen=hPen)
        #self.graph.setMinimumWidth(self.graphWidget.height())



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
