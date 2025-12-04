from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox
)
from PyQt5.QtCore import Qt


class DialogDenoise(QDialog):
    """
    降噪参数选择对话框
    支持：
    - 中值滤波
    - 高斯滤波
    - 双边滤波
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("降噪设置")
        self.setMinimumWidth(300)
        self.result = None  # 返回结果

        layout = QVBoxLayout()

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

        # 双边参数（可简化）
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
        hl_btn.addWidget(btn_ok)
        hl_btn.addWidget(btn_cancel)
        layout.addLayout(hl_btn)

        btn_ok.clicked.connect(self.apply)
        btn_cancel.clicked.connect(self.reject)

        self.setLayout(layout)

    def apply(self):
        """收集参数并返回"""
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
