import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QColor
from gui_web_layout import WebStyleApp
from tumor_classifier import tumor_predict


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 1. Load the Logo Image
    pixmap = QPixmap("logo.png")
    if pixmap.isNull():
        pixmap = QPixmap(800, 400)
        pixmap.fill(Qt.black)

    # 2. Create the Splash Screen
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.showMessage("Initializing OncoSim AI Engine...", Qt.AlignBottom | Qt.AlignCenter, QColor(180, 180, 180))
    splash.show()

    # 3. Define a function to launch the main app
    def launch_main_app():
        global window
        window = WebStyleApp()
        window.show()
        splash.close()

    QTimer.singleShot(3000, launch_main_app)

    sys.exit(app.exec_())
