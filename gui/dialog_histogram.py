import numpy as np
import cv2

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt5.QtCore import Qt


class DialogHistogram(QDialog):
    """
    显示当前图像灰度直方图（专业感 up）
    """

    def __init__(self, img: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("直方图")
        self.setMinimumSize(400, 300)

        layout = QVBoxLayout()

        # 计算灰度直方图
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        if hist.max() > 0:
            hist = hist / hist.max()

        # 画到 QImage 上
        w, h = 256, 200
        qimg = QImage(w, h, QImage.Format_RGB888)
        qimg.fill(Qt.black)

        painter = QPainter(qimg)
        painter.setPen(QColor(0, 255, 0))

        for x in range(256):
            v = hist[x]
            y = int(v * (h - 1))
            painter.drawLine(x, h - 1, x, h - 1 - y)

        painter.end()

        label = QLabel()
        label.setPixmap(QPixmap.fromImage(qimg))
        layout.addWidget(label)

        self.setLayout(layout)
