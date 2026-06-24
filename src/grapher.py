import bisect
from PySide6 import QtCore
from PySide6.QtCore import QUrl, QSize
from PySide6.QtWidgets import (QApplication, QDial, QGridLayout, QHBoxLayout, QMainWindow, QPushButton, QToolBar, QWidget)

import pyqtgraph as pg

import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        toolBar = QToolBar()
        self.addToolBar(toolBar)
        fileMenu = self.menuBar().addMenu("&File")
        editMenu = self.menuBar().addMenu("&Edit")
        viewMenu = self.menuBar().addMenu("&View")
        self.graphWidget = GraphWidget()
        self.setCentralWidget(self.graphWidget)

class GraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.points = []
        self.lines = []
        self.gridLayout = QGridLayout(self)
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
        self.graphButtonWidget = QWidget()
        self.buttonLayout = QHBoxLayout(self.graphButtonWidget)
        self.playButton = QPushButton("Play")
        self.clearButton = QPushButton("Clear")
        self.playButton.setMinimumSize(QSize(30, 30))
        self.playButton.setMaximumSize(QSize(100, 100))
        self.durationDial = QDial()
        self.buttonLayout.addWidget(self.playButton)
        #self.buttonLayout.addWidget(self.durationDial)
        self.gridLayout.addWidget(self.window, 0, 0)
        self.gridLayout.addWidget(self.graphButtonWidget, 1, 0)

    def addPoint(self, event):
        click_event, = event
        scenePos = click_event.scenePos()
        if self.viewBox.sceneBoundingRect().contains(click_event.scenePos()):
            plotPosition = self.viewBox.mapSceneToView(scenePos)
            bisect.insort(self.points, [plotPosition.x(), plotPosition.y()], key=lambda t: t[0])
            self.graphPoints(self.points)


    def clearGraph(self):
        self.graph.clear()
        self.graph.addItem(self.vertical)
        self.graph.addItem(self.horizontal)


    def graphPoints(self, points):
        self.clearGraph()
        self.graph.plot(*zip(*points))
        

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
