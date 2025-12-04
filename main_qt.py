import sys, os
from PyQt5.QtWidgets import QApplication
from gui.qt_app import MainWindow

def run_qt_app():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 建议加上

    qss_path = os.path.join(os.path.dirname(__file__), "gui", "theme.qss")
    with open(qss_path, "r", encoding="utf-8") as f:
        app.setStyleSheet(f.read())

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_qt_app()
