from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox
)
from PyQt5.QtCore import Qt


class DialogCrop(QDialog):
    """
    裁剪参数设置对话框：
    x, y 为起点
    w, h 为宽度和高度
    """

    def __init__(self, parent=None, img_width=0, img_height=0):
        super().__init__(parent)
        self.setWindowTitle("裁剪设置")
        self.setMinimumWidth(300)
        self.result = None

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"当前图像尺寸：{img_width} x {img_height}"))

        # 输入区域
        self.spin_x = QSpinBox()
        self.spin_y = QSpinBox()
        self.spin_w = QSpinBox()
        self.spin_h = QSpinBox()

        self.spin_x.setRange(0, img_width)
        self.spin_y.setRange(0, img_height)
        self.spin_w.setRange(1, img_width)
        self.spin_h.setRange(1, img_height)

        # 行布局
        def add_row(label, widget):
            hl = QHBoxLayout()
            hl.addWidget(QLabel(label))
            hl.addWidget(widget)
            layout.addLayout(hl)

        add_row("起点 X：", self.spin_x)
        add_row("起点 Y：", self.spin_y)
        add_row("宽度 W：", self.spin_w)
        add_row("高度 H：", self.spin_h)

        # OK/Cancel
        hl_btn = QHBoxLayout()
        btn_ok = QPushButton("确定")
        btn_cancel = QPushButton("取消")
        hl_btn.addWidget(btn_ok)
        hl_btn.addWidget(btn_cancel)
        layout.addLayout(hl_btn)

        btn_ok.clicked.connect(self.apply)
        btn_cancel.clicked.connect(self.reject)

        self.setLayout(layout)

    def apply(self):
        x = self.spin_x.value()
        y = self.spin_y.value()
        w = self.spin_w.value()
        h = self.spin_h.value()

        self.result = {"x": x, "y": y, "w": w, "h": h}
        self.accept()
