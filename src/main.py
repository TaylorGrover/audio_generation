import sys

from wave_app_model import WaveModel
from wave_controller import WaveController
from WaveGrapher import WaveView

from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    waveView = WaveView()
    waveView.showFullScreen()
    waveView.setWindowTitle("Graph Synthesis")
    availableGeometry = waveView.screen().availableGeometry()
    waveView.resize(availableGeometry.width(), availableGeometry.height())
    waveModel = WaveModel()
    waveController = WaveController(waveView, waveModel)
    waveView.show()
    screens = QApplication.screens()
    if len(screens) > 0:
        print(waveView)
        print(waveView.windowHandle())
        waveView.windowHandle().setScreen(screens[1])
    waveView.showFullScreen()

    sys.exit(app.exec())