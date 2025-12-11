import math
from typing import List, Tuple, Optional

from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from PyQt5.QtGui import QPainter, QPixmap, QPen, QColor, QFont
from PyQt5.QtWidgets import QWidget


class ImageViewer(QWidget):
    """
    功能：
    - 鼠标滚轮缩放
    - 左键拖拽平移（在 view 模式）
    - ROI 框选（在 roi 模式）
    - 显示标注框
    - 显示 AI 文本
    - 自适应图像显示
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None

        self._scale: float = 1.0
        self._min_scale: float = 0.1
        self._max_scale: float = 10.0

        self._offset: QPoint = QPoint(0, 0)
        self._last_mouse_pos: Optional[QPoint] = None

        # 交互模式：view / roi
        self._mode: str = "view"

        # ROI 相关（view 坐标）
        self._roi_active: bool = False
        self._roi_start_view: Optional[QPoint] = None
        self._roi_end_view: Optional[QPoint] = None

        # 最终 ROI（图像坐标 x, y, w, h）
        self._last_roi_rect_img: Optional[Tuple[int, int, int, int]] = None

        # 标注列表：[(x1, y1, x2, y2, label), ...] 图像坐标
        self._annotations: List[Tuple[int, int, int, int, str]] = []

        # AI 文本
        self._ai_text: Optional[str] = None

        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

    # ---------------- 对外接口 ----------------

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._last_roi_rect_img = None

        # 使用 QTimer 延迟执行自适应，确保窗口已经正确渲染
        if pixmap and not pixmap.isNull():
            QTimer.singleShot(100, self._fit_to_view)
        else:
            self._scale = 1.0
            self._offset = QPoint(0, 0)
            self.update()

    def _fit_to_view(self):
        """自适应缩放并居中显示图像"""
        if not self._pixmap or self._pixmap.isNull():
            return

        view_width = self.width()
        view_height = self.height()
        img_width = self._pixmap.width()
        img_height = self._pixmap.height()

        if view_width <= 0 or view_height <= 0 or img_width <= 0 or img_height <= 0:
            self._scale = 1.0
            self._offset = QPoint(0, 0)
            self.update()
            return

        # 计算缩放比例，留40px边距
        margin = 40
        scale_w = (view_width - margin) / img_width
        scale_h = (view_height - margin) / img_height
        self._scale = min(scale_w, scale_h, 1.0)  # 不放大，只缩小

        # 居中显示
        scaled_w = img_width * self._scale
        scaled_h = img_height * self._scale
        self._offset = QPoint(
            int((view_width - scaled_w) / 2),
            int((view_height - scaled_h) / 2)
        )

        self.update()

    def clear(self):
        self._pixmap = None
        self._annotations = []
        self._ai_text = None
        self._last_roi_rect_img = None
        self.update()

    def set_annotations(self, annotations: List[Tuple[int, int, int, int, str]]):
        self._annotations = annotations or []
        self.update()

    def set_ai_text(self, text: Optional[str]):
        self._ai_text = text
        self.update()

    def set_mode(self, mode: str):
        """
        mode: "view" / "roi"
        view: 左键拖动平移
        roi: 左键框选 ROI
        """
        if mode not in ("view", "roi"):
            return
        self._mode = mode
        self._roi_active = False
        self._roi_start_view = None
        self._roi_end_view = None
        self.update()

    def get_mode(self) -> str:
        return self._mode

    def get_last_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """
        返回最近一次 ROI（图像坐标 x, y, w, h）
        """
        return self._last_roi_rect_img

    # ---------------- 坐标转换 ----------------

    def _view_to_image(self, p: QPoint) -> Optional[Tuple[int, int]]:
        if self._pixmap is None or self._scale == 0:
            return None
        x = (p.x() - self._offset.x()) / self._scale
        y = (p.y() - self._offset.y()) / self._scale
        return int(x), int(y)

    def _image_to_view(self, x: int, y: int) -> QPoint:
        vx = int(x * self._scale + self._offset.x())
        vy = int(y * self._scale + self._offset.y())
        return QPoint(vx, vy)

    # ---------------- 事件 ----------------

    def wheelEvent(self, event):
        if self._pixmap is None:
            return

        angle_delta = event.angleDelta().y() / 120.0
        factor = 1.2 ** angle_delta

        old_scale = self._scale
        new_scale = max(self._min_scale, min(self._max_scale, self._scale * factor))
        if math.isclose(old_scale, new_scale):
            return

        mouse_pos = event.pos()
        self._offset = mouse_pos - (mouse_pos - self._offset) * (new_scale / old_scale)
        self._scale = new_scale
        self.update()

    def mousePressEvent(self, event):
        if self._pixmap is None:
            return

        if self._mode == "view":
            if event.button() == Qt.LeftButton:
                self._last_mouse_pos = event.pos()

        elif self._mode == "roi":
            if event.button() == Qt.LeftButton:
                self._roi_active = True
                self._roi_start_view = event.pos()
                self._roi_end_view = event.pos()

    def mouseMoveEvent(self, event):
        if self._pixmap is None:
            return

        if self._mode == "view":
            if self._last_mouse_pos is not None and event.buttons() & Qt.LeftButton:
                delta = event.pos() - self._last_mouse_pos
                self._offset += delta
                self._last_mouse_pos = event.pos()
                self.update()

        elif self._mode == "roi":
            if self._roi_active and event.buttons() & Qt.LeftButton:
                self._roi_end_view = event.pos()
                self.update()

    def mouseReleaseEvent(self, event):
        if self._mode == "view":
            if event.button() == Qt.LeftButton:
                self._last_mouse_pos = None

        elif self._mode == "roi":
            if self._roi_active and event.button() == Qt.LeftButton:
                self._roi_active = False

                if self._roi_start_view is None or self._roi_end_view is None:
                    return

                # 转换到图像坐标
                p1 = self._view_to_image(self._roi_start_view)
                p2 = self._view_to_image(self._roi_end_view)
                if p1 is None or p2 is None:
                    return

                x1, y1 = p1
                x2, y2 = p2
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])

                if self._pixmap is not None:
                    w_img = self._pixmap.width()
                    h_img = self._pixmap.height()
                    x1 = max(0, min(w_img - 1, x1))
                    x2 = max(0, min(w_img, x2))
                    y1 = max(0, min(h_img - 1, y1))
                    y2 = max(0, min(h_img, y2))

                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                self._last_roi_rect_img = (x1, y1, w, h)
                self.update()

    # ---------------- 绘制 ----------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._pixmap is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "暂无图像")
            return

        painter.save()
        painter.translate(self._offset)
        painter.scale(self._scale, self._scale)

        # 图像
        painter.drawPixmap(0, 0, self._pixmap)

        # 标注框（图像坐标）
        if self._annotations:
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            font = QFont()
            font.setPointSize(10)
            painter.setFont(font)

            for (x1, y1, x2, y2, label) in self._annotations:
                w = x2 - x1
                h = y2 - y1
                painter.drawRect(x1, y1, w, h)
                if label:
                    painter.drawText(x1, y1 - 5, label)

        painter.restore()

        # ROI 矩形（用视图坐标画）
        if self._last_roi_rect_img is not None:
            x, y, w, h = self._last_roi_rect_img
            p1 = self._image_to_view(x, y)
            p2 = self._image_to_view(x + w, y + h)
            roi_rect = QRect(p1, p2)

            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(roi_rect)

        # 当前正在拖动的 ROI
        if self._roi_active and self._roi_start_view and self._roi_end_view:
            pen = QPen(QColor(255, 255, 0))
            pen.setStyle(Qt.DashLine)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawRect(QRect(self._roi_start_view, self._roi_end_view))

        # AI 文本（固定在右下角）
        if self._ai_text:
            painter.setPen(QColor(255, 255, 0))
            font = QFont()
            font.setPointSize(11)
            painter.setFont(font)
            margin = 10
            rect = self.rect().adjusted(margin, margin, -margin, -margin)
            painter.drawText(rect, Qt.AlignRight | Qt.AlignBottom, f"AI: {self._ai_text}")