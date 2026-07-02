import sys

from wave_controller import WaveController
from WaveGrapher import WaveView

from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    waveView = WaveView()
    waveView.setWindowTitle("Graph Synthesis")
    availableGeometry = waveView.screen().availableGeometry()
    waveView.resize(availableGeometry.width(), availableGeometry.height())
    waveController = WaveController(waveView)
    waveView.show()


    sys.exit(app.exec())