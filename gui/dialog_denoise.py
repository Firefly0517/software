from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox
)
from PyQt5.QtCore import Qt


class DialogDenoise(QDialog):
    """降噪参数选择对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("降噪设置")
        self.setMinimumWidth(400)
        self.result = None

        # 应用深色主题
        self.setStyleSheet("""
            QDialog {
                background: #2D2D2D;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 13px;
            }
            QComboBox {
                background: #3A3A3A;
                color: #E0E0E0;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px;
                min-height: 28px;
            }
            QComboBox::drop-down {
                border: none;
                background: #4A4A4A;
            }
            QComboBox::down-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 7px solid #E0E0E0;
            }
            QComboBox QAbstractItemView {
                background: #3A3A3A;
                color: #E0E0E0;
                selection-background-color: #4A90E2;
            }
            QSpinBox {
                background: #3A3A3A;
                color: #E0E0E0;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px;
                min-height: 28px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background: #4A4A4A;
                border: none;
            }
            QSpinBox::up-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 6px solid #E0E0E0;
            }
            QSpinBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #E0E0E0;
            }
            QPushButton {
                background: #4A90E2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                min-height: 36px;
            }
            QPushButton:hover {
                background: #5BA3F5;
            }
            QPushButton#cancelBtn {
                background: #555555;
            }
            QPushButton#cancelBtn:hover {
                background: #666666;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # 方法选择
        hl_method = QHBoxLayout()
        hl_method.addWidget(QLabel("降噪方法："))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["median", "gaussian", "bilateral"])
        hl_method.addWidget(self.combo_method)
        layout.addLayout(hl_method)

        # ksize 输入
        hl_ksize = QHBoxLayout()
        hl_ksize.addWidget(QLabel("核大小 ksize（奇数）："))
        self.spin_ksize = QSpinBox()
        self.spin_ksize.setRange(1, 31)
        self.spin_ksize.setSingleStep(2)
        self.spin_ksize.setValue(3)
        hl_ksize.addWidget(self.spin_ksize)
        layout.addLayout(hl_ksize)

        # 高斯 sigmaX
        hl_sigma = QHBoxLayout()
        hl_sigma.addWidget(QLabel("sigmaX（仅高斯）："))
        self.spin_sigma = QSpinBox()
        self.spin_sigma.setRange(0, 50)
        self.spin_sigma.setValue(0)
        hl_sigma.addWidget(self.spin_sigma)
        layout.addLayout(hl_sigma)

        # 双边参数
        hl_bi = QHBoxLayout()
        hl_bi.addWidget(QLabel("bilateral d："))
        self.spin_bi_d = QSpinBox()
        self.spin_bi_d.setRange(1, 50)
        self.spin_bi_d.setValue(5)
        hl_bi.addWidget(self.spin_bi_d)
        layout.addLayout(hl_bi)

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
        method = self.combo_method.currentText()
        ksize = self.spin_ksize.value()
        sigma = self.spin_sigma.value()
        bi_d = self.spin_bi_d.value()

        self.result = {
            "method": method,
            "ksize": ksize,
            "sigmaX": sigma,
            "bilateral_d": bi_d,
            "bilateral_sigma_color": 75,
            "bilateral_sigma_space": 75,
        }
        self.accept()