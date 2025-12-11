# gui/text_analysis_widget.py
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QSizePolicy, QMessageBox, QLineEdit
)

from core.nlp_module import NLPEngine, NLPConfig


class _NLPWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, engine: NLPEngine, mode: str, text: str, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.mode = mode
        self.text = text

    def run(self):
        try:
            if self.mode == "summary":
                res = self.engine.summarize_text(self.text)
            elif self.mode == "analyze":
                res = self.engine.analyze_medical_record(self.text)
            elif self.mode == "diagnose":
                res = self.engine.suggest_diagnosis(self.text)
            else:
                raise ValueError(f"未知模式：{self.mode}")
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class TextAnalysisWidget(QWidget):
    """右侧信息面板里用的 NLP 文本分析区"""

    def __init__(self, parent=None, engine: Optional[NLPEngine] = None):
        super().__init__(parent)
        self.engine = engine or NLPEngine()
        self._worker: Optional[_NLPWorker] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # 顶部：模型信息
        top_layout = QHBoxLayout()
        lbl = QLabel("模型：")
        self.edit_model = QLineEdit(self.engine.config.model_name)
        self.edit_model.setReadOnly(True)
        self.edit_model.setStyleSheet("color: #AAAAAA;")
        top_layout.addWidget(lbl)
        top_layout.addWidget(self.edit_model)
        layout.addLayout(top_layout)

        # 输入文本
        layout.addWidget(QLabel("输入病历 / 报告文本："))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("在这里粘贴或输入病历文本...")
        self.text_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.text_input.setMinimumHeight(120)
        layout.addWidget(self.text_input)

        # 按钮区
        btn_layout = QHBoxLayout()
        self.btn_summary = QPushButton("生成摘要")
        self.btn_analyze = QPushButton("病历解析")
        self.btn_diagnose = QPushButton("诊断建议")

        for b in [self.btn_summary, self.btn_analyze, self.btn_diagnose]:
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumHeight(30)

        btn_layout.addWidget(self.btn_summary)
        btn_layout.addWidget(self.btn_analyze)
        btn_layout.addWidget(self.btn_diagnose)
        layout.addLayout(btn_layout)

        # 输出文本
        layout.addWidget(QLabel("NLP 结果："))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setMinimumHeight(150)
        layout.addWidget(self.text_output)

        self.setLayout(layout)

        # 信号
        self.btn_summary.clicked.connect(lambda: self._run("summary"))
        self.btn_analyze.clicked.connect(lambda: self._run("analyze"))
        self.btn_diagnose.clicked.connect(lambda: self._run("diagnose"))

    def _run(self, mode: str):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请先输入要分析的文本。")
            return

        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "提示", "正在处理上一个请求，请稍候。")
            return

        self.text_output.clear()
        self.text_output.setPlainText("正在调用本地 Qwen 模型，请稍等...")

        self._worker = _NLPWorker(self.engine, mode, text, self)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, result: str):
        self.text_output.clear()
        self.text_output.setPlainText(result)

    def _on_error(self, msg: str):
        self.text_output.clear()
        self.text_output.setPlainText(f"发生错误：\n{msg}")
