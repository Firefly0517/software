from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QWidget
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


class DialogCrop(QDialog):
    """裁剪参数设置对话框"""

    def __init__(self, parent=None, img_width=0, img_height=0):
        super().__init__(parent)
        self.setWindowTitle("裁剪设置")
        self.setMinimumWidth(400)
        self.result = None

        # 获取父窗口字体大小
        if parent:
            font_size = getattr(parent, 'font_size', 13)
        else:
            font_size = 13

        # 应用深色主题（字体大小动态）
        self.setStyleSheet(f"""
            QDialog {{
                background: #2D2D2D;
            }}
            QLabel {{
                color: #E0E0E0;
                font-size: {font_size}pt;
            }}
            QSpinBox {{
                background: #3A3A3A;
                color: #E0E0E0;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px;
                min-height: 28px;
                font-size: {font_size}pt;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                background: #4A4A4A;
                border: none;
            }}
            QSpinBox::up-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 6px solid #E0E0E0;
            }}
            QSpinBox::down-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #E0E0E0;
            }}
            QPushButton {{
                background: #4A90E2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                min-height: 36px;
                font-size: {max(11, font_size - 1)}pt;
            }}
            QPushButton:hover {{
                background: #5BA3F5;
            }}
            QPushButton#cancelBtn {{
                background: #555555;
            }}
            QPushButton#cancelBtn:hover {{
                background: #666666;
            }}
        """)

        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

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
        btn_cancel.setObjectName("cancelBtn")
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