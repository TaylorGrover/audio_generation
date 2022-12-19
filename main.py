import numpy as np
import os
import soundfile as sf
import sys

from waveforms import *

from PySide6 import QtWidgets
from PySide6.QtGui import QAction, QColor, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import (QApplication, QPushButton, QCheckBox, QComboBox,
    QHBoxLayout, QFormLayout, QGridLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
    QMessageBox, QTabWidget, QToolBar, QWidget)
import pyqtgraph as pg


sample_rate = 44100
IMAGE_DIR = "images"

class MainWindow(QMainWindow):
    """
    """
    def __init__(self):
        super().__init__()
        toolBar = QToolBar()
        self.addToolBar(toolBar)
        self.tabWindow = QTabWidget()
        self.waveformTab = WaveformTab()
        self.tabWindow.addTab(self.waveformTab, "Instrument Visualizer")
        fileMenu = self.menuBar().addMenu("&File")
        editMenu = self.menuBar().addMenu("&Edit")
        viewMenu = self.menuBar().addMenu("&View")
        exitAction = QAction("&Exit", self, shortcut="Ctrl+Q", triggered=self.close)
        fullScreenAction = QAction("&Fullscreen", self, shortcut="F11", triggered = self.maximize)
        fileMenu.addAction(exitAction)
        viewMenu.addAction(fullScreenAction)
        self.setCentralWidget(self.tabWindow)
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

    def maximize(self):
        if self.maximized:
            self.showNormal()
        else:
            self.showMaximized()
        self.maximized = not self.maximized


class WaveformTab(QWidget):
    """
    Contains the graph to display a single period of the current waveform, and 
    a menu to add waveforms.
    """
    def __init__(self):
        super().__init__()
        self.formLayout = QFormLayout(self)
        self.graphWidget = pg.PlotWidget()
        self.addWaveButton = QPushButton("Add Waveform")
        self.addWaveButton.clicked.connect(self.createWaveTab)
        hPen = pg.mkPen("#00ffff", width=3)
        fPen = pg.mkPen("r", width=3)
        self.t = np.arange(0, 1, 1.0 / sample_rate)
        self.waveforms = np.array([])
        self.graphWidget.plot(self.t, np.sum(self.waveforms, axis=0), pen=hPen)
        self.graphWidget.setMinimumWidth(self.graphWidget.height())
        self.formLayout.addRow("", self.graphWidget)
        self.formLayout.addRow("", self.addWaveButton)

    def playWaveform(self):
        """
        TODO: Implement
        """
        sf.write("audio/waveform.wav", self.waveform, sample_rate)

    def createWaveTab(self):
        """
        """
        print("asdf")


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
