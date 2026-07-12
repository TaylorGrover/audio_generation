import sys

from wave_app_model import WaveModel
from wave_controller import WaveController
from WaveGrapher import WaveView

from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    waveView = WaveView()
    waveView.setWindowTitle("Graph Synthesis")
    availableGeometry = waveView.screen().availableGeometry()
    waveView.resize(availableGeometry.width(), availableGeometry.height())
    waveModel = WaveModel()
    waveController = WaveController(waveView, waveModel)
    screens = QApplication.screens()
    if len(screens) > 0:
        waveView.move(screens[0].geometry().topLeft())

    waveView.showMaximized()

    sys.exit(app.exec())