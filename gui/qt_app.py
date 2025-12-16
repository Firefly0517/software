import sys
import cv2
import numpy as np
from typing import List, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QLabel, QComboBox,
    QMessageBox, QCheckBox, QFrame, QGroupBox, QSizePolicy,
    QDockWidget, QMenu, QAction
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

from gui.image_viewer import ImageViewer
from gui.dialog_denoise import DialogDenoise
from gui.dialog_crop import DialogCrop
from gui.dialog_histogram import DialogHistogram
from gui.text_analysis_widget import TextAnalysisWidget
from core.processing_manager import ProcessingManager
from core.annotation import AnnotationManager
from core.ai_module import SimpleAIDiagnosis
from core.metadata_utils import parse_metadata
from core.nlp_module import NLPEngine, NLPConfig


def cv2_to_pixmap(img: np.ndarray) -> QPixmap:
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        h, w = img.shape
        q_img = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        q_img = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)


def parse_annotations(annotations: List[dict]) -> List[Tuple[int, int, int, int, str]]:
    result = []
    for ann in annotations:
        loc = ann.get("病灶位置", "")
        label = ann.get("病灶类型", "")
        x1 = y1 = x2 = y2 = 0
        try:
            parts = loc.split(",")
            for p in parts:
                p = p.strip()
                if p.startswith("x1="):
                    x1 = int(p.split("=")[1])
                elif p.startswith("y1="):
                    y1 = int(p.split("=")[1])
                elif p.startswith("x2="):
                    x2 = int(p.split("=")[1])
                elif p.startswith("y2="):
                    y2 = int(p.split("=")[1])
            if x2 <= x1:
                x2 = x1 + 10
            if y2 <= y1:
                y2 = y1 + 10
        except:
            x1, y1, x2, y2 = 30, 30, 80, 80
        result.append((x1, y1, x2, y2, label))
    return result


class ModernButton(QPushButton):
    """现代化按钮样式"""

    def __init__(self, text, primary=False, parent=None):
        super().__init__(text, parent)
        self.primary = primary
        self.setMinimumHeight(36)
        self.setMaximumHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self._apply_style()

    def _apply_style(self):
        main_window = self.window()
        font_size = getattr(main_window, 'font_size', 13)
        btn_font_size = max(10, font_size - 1)

        if self.primary:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4A90E2, stop:1 #357ABD);
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: {btn_font_size}pt;
                    font-weight: 600;
                    padding: 8px 16px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5BA3F5, stop:1 #4A90E2);
                }}
                QPushButton:pressed {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #357ABD, stop:1 #2868AA);
                }}
                QPushButton:disabled {{
                    background: #555555;
                    color: #888888;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: #3A3A3A;
                    color: #E0E0E0;
                    border: 1px solid #555555;
                    border-radius: 6px;
                    font-size: {btn_font_size}pt;
                    padding: 6px 12px;
                }}
                QPushButton:hover {{
                    background: #454545;
                    border-color: #4A90E2;
                }}
                QPushButton:pressed {{
                    background: #2A2A2A;
                }}
                QPushButton:disabled {{
                    background: #2A2A2A;
                    color: #666666;
                    border-color: #333333;
                }}
            """)


class InfoCard(QFrame):
    """信息卡片组件"""

    def __init__(self, title, value="--", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: #2D2D2D;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 12px;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(5)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #888888;")

        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("color: #E0E0E0; font-weight: 600;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        self.setLayout(layout)

    def set_value(self, value):
        self.value_label.setText(str(value))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能医疗影像处理系统（Dock 版）")
        self.resize(1600, 900)

        # 字体大小设置
        self.font_size = 14

        # 深色主题
        self.setStyleSheet("""
            QMainWindow {
                background: #1E1E1E;
            }
            QLabel {
                color: #E0E0E0;
            }
            QComboBox {
                background: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 6px;
                min-height: 28px;
            }
            QComboBox:hover {
                border-color: #4A90E2;
            }
            QComboBox::drop-down {
                border: none;
                background: #3A3A3A;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 7px solid #E0E0E0;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #404040;
                selection-background-color: #4A90E2;
                selection-color: #FFFFFF;
            }
            QTextEdit {
                background: #252525;
                color: #E0E0E0;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            QCheckBox {
                color: #E0E0E0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #404040;
                border-radius: 4px;
                background: #2D2D2D;
            }
            QCheckBox::indicator:hover {
                border-color: #4A90E2;
            }
            QCheckBox::indicator:checked {
                background: #4A90E2;
                border-color: #4A90E2;
            }
            QScrollBar:vertical {
                background: #2D2D2D;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #505050;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #606060;
            }
        """)

        self.manager = ProcessingManager()
        self.current_params = None

        # NLP 引擎（Ollama + Qwen）
        self.nlp_engine = NLPEngine(NLPConfig(
            model_name="qwen2.5:3b"
        ))

        # central widget 只占位，所有内容用 dock 实现
        central = QWidget()
        self.setCentralWidget(central)

        self._create_menu_bar()
        self._create_docks()

        self.change_font_size(self.font_size)

    # ----------------- 菜单栏 -----------------
    def _create_menu_bar(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background: #2D2D2D;
                color: #E0E0E0;
                border-bottom: 1px solid #404040;
                padding: 4px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background: #4A90E2;
            }
            QMenu {
                background: #2D2D2D;
                color: #E0E0E0;
                border: 1px solid #404040;
            }
            QMenu::item {
                padding: 8px 30px;
            }
            QMenu::item:selected {
                background: #4A90E2;
            }
        """)


        # 设置菜单
        settings_menu = menubar.addMenu("⚙️ 设置")
        font_menu = QMenu("字体大小", self)
        sizes = [10, 12, 14, 16, 18]
        for size in sizes:
            action = QAction(f"{size}pt", self)
            action.triggered.connect(lambda checked, s=size: self.change_font_size(s))
            font_menu.addAction(action)
        settings_menu.addMenu(font_menu)

        # 视图菜单（Dock 控制）
        self.view_menu = menubar.addMenu("🧩 视图")

    def _create_docks(self):
        # 工具面板 Dock（左侧）
        tools_widget = self._create_left_panel()
        self.dock_tools = QDockWidget("工具面板", self)
        self.dock_tools.setWidget(tools_widget)
        self._config_dock(self.dock_tools)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_tools)

        self.setDockOptions(
            QMainWindow.AllowTabbedDocks |
            QMainWindow.AllowNestedDocks |
            QMainWindow.AnimatedDocks
        )

        # ====== 关键修复：影像视图放在工具栏右侧，作为主 Dock ======
        image_widget = self._create_center_area()
        self.dock_image = QDockWidget("影像视图", self)
        self.dock_image.setWidget(image_widget)
        self._config_dock(self.dock_image)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_image)
        self.splitDockWidget(self.dock_tools, self.dock_image, Qt.Horizontal)

        # 影像信息 Dock（右侧 Tab）
        info_widget = self._create_info_panel()
        self.dock_info = QDockWidget("影像信息", self)
        self.dock_info.setWidget(info_widget)
        self._config_dock(self.dock_info)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_info)

        # NLP Dock（右侧 Tab）
        nlp_widget = self._create_nlp_panel()
        self.dock_nlp = QDockWidget("文本分析（NLP）", self)
        self.dock_nlp.setWidget(nlp_widget)
        self._config_dock(self.dock_nlp)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_nlp)

        # 右侧：信息 + NLP 合并成 Tab
        self.tabifyDockWidget(self.dock_info, self.dock_nlp)
        self.dock_info.raise_()

        # 日志 Dock（底部）
        log_widget = self._create_log_panel()
        self.dock_log = QDockWidget("处理日志", self)
        self.dock_log.setWidget(log_widget)
        self._config_dock(self.dock_log)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_log)

        # ====== 默认提升影像视图为主窗口焦点 ======
        self.dock_image.raise_()

        # ====== 将 Dock 的切换选项加入菜单 ======
        for dock in [self.dock_tools, self.dock_image, self.dock_info, self.dock_log, self.dock_nlp]:
            self.view_menu.addAction(dock.toggleViewAction())

    def _config_dock(self, dock: QDockWidget):
        dock.setFeatures(
            QDockWidget.DockWidgetClosable |
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable
        )
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea |
            Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea |
            Qt.BottomDockWidgetArea
        )
        dock.setContentsMargins(4, 4, 4, 4)

    # ----------------- 字体全局更新 -----------------
    def change_font_size(self, size):
        self.font_size = size
        app = QApplication.instance()
        base_font = QFont("Microsoft YaHei", size)
        app.setFont(base_font)

        def update_all_widgets(widget):
            widget.setFont(QFont("Microsoft YaHei", size))
            for child in widget.children():
                if isinstance(child, QWidget):
                    update_all_widgets(child)

        update_all_widgets(self)

        btn_font_size = max(10, size - 1)
        for btn in self.findChildren(ModernButton):
            if btn.primary:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #4A90E2, stop:1 #357ABD);
                        color: white;
                        border: none;
                        border-radius: 6px;
                        font-size: {btn_font_size}pt;
                        font-weight: 600;
                        padding: 8px 16px;
                    }}
                    QPushButton:hover {{
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #5BA3F5, stop:1 #4A90E2);
                    }}
                    QPushButton:pressed {{
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #357ABD, stop:1 #2868AA);
                    }}
                    QPushButton:disabled {{
                        background: #555555;
                        color: #888888;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: #3A3A3A;
                        color: #E0E0E0;
                        border: 1px solid #555555;
                        border-radius: 6px;
                        font-size: {btn_font_size}pt;
                        padding: 6px 12px;
                    }}
                    QPushButton:hover {{
                        background: #454545;
                        border-color: #4A90E2;
                    }}
                    QPushButton:pressed {{
                        background: #2A2A2A;
                    }}
                    QPushButton:disabled {{
                        background: #2A2A2A;
                        color: #666666;
                        border-color: #333333;
                    }}
                """)

        # QGroupBox 标题字体
        for group in self.findChildren(QGroupBox):
            group.setStyleSheet(f"""
                QGroupBox {{
                    color: #CCCCCC;
                    font-weight: 600;
                    font-size: {size}pt;
                    border: 1px solid #404040;
                    border-radius: 8px;
                    margin-top: 12px;
                    padding-top: 12px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 8px;
                }}
            """)

        # 日志字体
        log_font_size = max(9, size - 2)
        if hasattr(self, "text_log") and self.text_log:
            self.text_log.setStyleSheet(f"""
                QTextEdit {{
                    background: #1E1E1E;
                    color: #E0E0E0;
                    border: 1px solid #404040;
                    border-radius: 6px;
                    padding: 8px;
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: {log_font_size}pt;
                }}
            """)

        # 信息卡片
        card_title_size = max(9, size - 2)
        for card in getattr(self, "info_cards", []):
            card.title_label.setStyleSheet(f"color: #888888; font-size: {card_title_size}pt;")
            card.value_label.setStyleSheet(f"color: #E0E0E0; font-weight: 600; font-size: {size}pt;")

        # 图像标题标签
        title_size = max(9, size - 3)
        if hasattr(self, 'label_history_title'):
            self.label_history_title.setStyleSheet(
                f"color: #F5A623; font-weight: 600; font-size: {title_size}pt;")
            self.label_history_title.setFont(QFont("Microsoft YaHei", title_size))
        if hasattr(self, 'label_current_title'):
            self.label_current_title.setStyleSheet(
                f"color: #7ED321; font-weight: 600; font-size: {title_size}pt;")
            self.label_current_title.setFont(QFont("Microsoft YaHei", title_size))

        # 大标题（工具 / 影像视图 / 信息面板）
        title_font_size = max(16, size + 2)
        if hasattr(self, 'title_tool_panel'):
            self.title_tool_panel.setStyleSheet(
                f"font-size: {title_font_size}pt; font-weight: 700; color: #FFFFFF; padding: 8px 0;")
            self.title_tool_panel.setFont(QFont("Microsoft YaHei", title_font_size, QFont.Bold))
        if hasattr(self, 'title_image_view'):
            self.title_image_view.setStyleSheet(
                f"font-size: {title_font_size}pt; font-weight: 700; color: #FFFFFF;")
            self.title_image_view.setFont(QFont("Microsoft YaHei", title_font_size, QFont.Bold))
        if hasattr(self, 'title_info_panel'):
            self.title_info_panel.setStyleSheet(
                f"font-size: {title_font_size}pt; font-weight: 700; color: #FFFFFF; padding: 8px 0;")
            self.title_info_panel.setFont(QFont("Microsoft YaHei", title_font_size, QFont.Bold))
        if hasattr(self, 'view_mode_label'):
            self.view_mode_label.setStyleSheet(f"color: #CCCCCC; font-size: {size}pt;")
            self.view_mode_label.setFont(QFont("Microsoft YaHei", size))

        if hasattr(self, "nlp_widget"):
            self.nlp_widget.text_input.setFont(QFont("Microsoft YaHei", size))
            self.nlp_widget.text_output.setFont(QFont("Microsoft YaHei", size))

        self.update()
        self.repaint()
        QApplication.processEvents()

        if hasattr(self, "text_log") and self.text_log:
            self.text_log.append(f"✓ 全局字体已更改为 {size}pt")


    # ----------------- 各 Panel Widget 构建 -----------------
    def _create_left_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: #252525;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # 标题
        self.title_tool_panel = QLabel("工具面板")
        self.title_tool_panel.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #FFFFFF; padding: 8px 0;")
        layout.addWidget(self.title_tool_panel)

        # 文件操作
        file_group = QGroupBox("文件操作")
        file_group.setStyleSheet("""
            QGroupBox {
                color: #CCCCCC;
                font-weight: 600;
                border: 1px solid #404040;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        file_layout = QVBoxLayout()
        self.btn_choose = ModernButton("📂 选择影像", primary=True)
        file_layout.addWidget(self.btn_choose)

        self.btn_save_image = ModernButton("💾 保存当前影像")
        file_layout.addWidget(self.btn_save_image)
        self.btn_save_image.clicked.connect(self.save_current_image)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 预处理操作
        preprocess_group = QGroupBox("预处理操作")
        preprocess_group.setStyleSheet(file_group.styleSheet())
        preprocess_layout = QVBoxLayout()
        preprocess_layout.setSpacing(10)

        self.combo_action = QComboBox()
        self.combo_action.addItems([
            "选择操作...",
            "🔇 降噪处理",
            "✂️ 裁剪区域",
            "🎨 格式转换",
            "📐 对齐校正",
            "🔄 旋转 90°",
            "🔃 水平翻转",
            "📊 直方图均衡"
        ])
        self.combo_action.setEnabled(False)

        self.btn_set_param = ModernButton("⚙️ 设置参数")
        self.btn_set_param.setEnabled(False)

        self.btn_apply = ModernButton("▶️ 执行", primary=True)
        self.btn_apply.setEnabled(False)

        preprocess_layout.addWidget(self.combo_action)
        preprocess_layout.addWidget(self.btn_set_param)
        preprocess_layout.addWidget(self.btn_apply)
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)

        # 高级功能
        advanced_group = QGroupBox("高级功能")
        advanced_group.setStyleSheet(file_group.styleSheet())
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(10)

        self.cb_roi_mode = QCheckBox("📍 ROI 框选模式")
        self.btn_crop_roi = ModernButton("✂️ 应用 ROI 裁剪")
        self.btn_hist = ModernButton("📈 查看直方图")
        self.btn_ai = ModernButton("🤖 AI 诊断", primary=True)

        self.btn_crop_roi.setEnabled(False)
        self.btn_hist.setEnabled(False)
        self.btn_ai.setEnabled(False)

        advanced_layout.addWidget(self.cb_roi_mode)
        advanced_layout.addWidget(self.btn_crop_roi)
        advanced_layout.addWidget(self.btn_hist)
        advanced_layout.addWidget(self.btn_ai)
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # 历史操作
        history_group = QGroupBox("历史操作")
        history_group.setStyleSheet(file_group.styleSheet())
        history_layout = QVBoxLayout()
        history_layout.setSpacing(8)

        undo_redo_layout = QHBoxLayout()
        self.btn_undo = ModernButton("↶ 撤销")
        self.btn_redo = ModernButton("↷ 恢复")
        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)
        undo_redo_layout.addWidget(self.btn_undo)
        undo_redo_layout.addWidget(self.btn_redo)

        history_layout.addLayout(undo_redo_layout)
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

        layout.addStretch()
        panel.setLayout(layout)

        # 信号连接
        self.btn_choose.clicked.connect(self.choose_file)
        self.combo_action.currentIndexChanged.connect(self._action_changed)
        self.btn_set_param.clicked.connect(self.set_params)
        self.btn_apply.clicked.connect(self.apply_action)
        self.btn_undo.clicked.connect(self.do_undo)
        self.btn_redo.clicked.connect(self.do_redo)
        self.cb_roi_mode.stateChanged.connect(self.toggle_roi_mode)
        self.btn_crop_roi.clicked.connect(self.crop_by_roi)
        self.btn_hist.clicked.connect(self.show_histogram)
        self.btn_ai.clicked.connect(self.run_ai)

        return panel

    def _create_center_area(self):
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background: #252525;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # 标题栏
        title_layout = QHBoxLayout()
        self.title_image_view = QLabel("影像视图")
        self.title_image_view.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #FFFFFF;")
        title_layout.addWidget(self.title_image_view)
        title_layout.addStretch()

        # 视图切换
        self.view_mode_label = QLabel("显示模式:")
        self.view_mode_label.setStyleSheet("color: #CCCCCC; font-size: 13px;")
        self.combo_history = QComboBox()
        self.combo_history.addItem("当前处理结果")
        self.combo_history.setMinimumWidth(200)
        self.combo_history.currentIndexChanged.connect(self.update_view)

        title_layout.addWidget(self.view_mode_label)
        title_layout.addWidget(self.combo_history)
        layout.addLayout(title_layout)

        # 图像显示，上：历史；下：当前
        viewers_layout = QVBoxLayout()
        viewers_layout.setSpacing(8)

        history_container = QFrame()
        history_container.setStyleSheet("""
            QFrame {
                background: #1E1E1E;
                border: 1px solid #404040;
                border-radius: 8px;
            }
        """)
        history_layout = QVBoxLayout()
        history_layout.setContentsMargins(4, 4, 4, 4)
        history_layout.setSpacing(2)
        self.label_history_title = QLabel("步骤")
        self.label_history_title.setStyleSheet(
            "color: #F5A623; font-weight: 600; font-size: 11px;")
        self.label_history_title.setAlignment(Qt.AlignCenter)
        self.label_history_title.setMaximumHeight(20)
        self.viewer_middle = ImageViewer()
        history_layout.addWidget(self.label_history_title)
        history_layout.addWidget(self.viewer_middle)
        history_container.setLayout(history_layout)

        current_container = QFrame()
        current_container.setStyleSheet(history_container.styleSheet())
        current_layout = QVBoxLayout()
        current_layout.setContentsMargins(4, 4, 4, 4)
        current_layout.setSpacing(2)
        self.label_current_title = QLabel("当前")
        self.label_current_title.setStyleSheet(
            "color: #7ED321; font-weight: 600; font-size: 11px;")
        self.label_current_title.setAlignment(Qt.AlignCenter)
        self.label_current_title.setMaximumHeight(20)
        self.viewer_current = ImageViewer()
        current_layout.addWidget(self.label_current_title)
        current_layout.addWidget(self.viewer_current)
        current_container.setLayout(current_layout)

        viewers_layout.addWidget(history_container, 1)
        viewers_layout.addWidget(current_container, 1)

        layout.addLayout(viewers_layout)
        container.setLayout(layout)
        return container

    def _create_info_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: #252525;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        self.title_info_panel = QLabel("信息面板")
        self.title_info_panel.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #FFFFFF; padding: 8px 0;")
        layout.addWidget(self.title_info_panel)

        info_group = QGroupBox("影像信息")
        info_group.setStyleSheet("""
            QGroupBox {
                color: #CCCCCC;
                font-weight: 600;
                border: 1px solid #404040;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        info_layout = QVBoxLayout()
        info_layout.setSpacing(10)

        self.card_filename = InfoCard("文件名", "--")
        self.card_dimensions = InfoCard("尺寸", "--")
        self.card_format = InfoCard("格式", "--")

        info_layout.addWidget(self.card_filename)
        info_layout.addWidget(self.card_dimensions)
        info_layout.addWidget(self.card_format)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        panel.setLayout(layout)

        # 用于批量调整卡片字体
        self.info_cards = [self.card_filename, self.card_dimensions, self.card_format]

        return panel

    def _create_log_panel(self):
        panel = QWidget()
        panel.setStyleSheet("""
            QFrame {
                background: #252525;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        log_group = QGroupBox("处理日志")
        log_group.setStyleSheet("""
            QGroupBox {
                color: #CCCCCC;
                font-weight: 600;
                border: 1px solid #404040;
                border-radius: 8px;
                margin-top: 4px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        log_layout = QVBoxLayout()

        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setStyleSheet("""
            QTextEdit {
                background: #1E1E1E;
                color: #E0E0E0;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)

        log_layout.addWidget(self.text_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        panel.setLayout(layout)
        return panel

    def _create_nlp_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: #252525;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        nlp_group = QGroupBox("文本分析（NLP）")
        nlp_group.setStyleSheet("""
            QGroupBox {
                color: #CCCCCC;
                font-weight: 600;
                border: 1px solid #404040;
                border-radius: 8px;
                margin-top: 4px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        nlp_layout = QVBoxLayout()

        self.nlp_widget = TextAnalysisWidget(self, engine=self.nlp_engine)
        nlp_layout.addWidget(self.nlp_widget)

        nlp_group.setLayout(nlp_layout)
        layout.addWidget(nlp_group)

        panel.setLayout(layout)
        return panel

    # ----------------- 功能逻辑 -----------------
    def _refresh_info(self, img: np.ndarray, display_name: str):
        self.card_filename.set_value(display_name)
        self.card_dimensions.set_value(f"{img.shape[1]} × {img.shape[0]}")

        if img.ndim == 2:
            fmt = "灰度"
        elif img.shape[2] == 3:
            fmt = "RGB"
        else:
            fmt = f"{img.shape[2]} 通道"
        self.card_format.set_value(fmt)

    def _refresh_history_combo(self):
        self.combo_history.clear()
        if not self.manager.has_image():
            return
        tags = self.manager.get_history_descriptions()
        for i, t in enumerate(tags):
            self.combo_history.addItem(f"步骤 {i}: {t}")

    def update_view(self):
        if not self.manager.has_image():
            return
        idx = self.combo_history.currentIndex()
        img = self.manager.get_image_at_step(idx)
        if img is None:
            return
        pix = cv2_to_pixmap(img)
        self.viewer_middle.set_pixmap(pix)
        self.label_history_title.setText(f"步骤 {idx}")

    def _update_viewers_post_action(self, img, display_name):
        pix = cv2_to_pixmap(img)
        self.viewer_current.set_pixmap(pix)
        self._refresh_info(img, display_name)
        self._refresh_history_combo()
        self.update_view()

        QApplication.processEvents()

        self.btn_undo.setEnabled(self.manager.can_undo())
        self.btn_redo.setEnabled(self.manager.can_redo())

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择影像", "", "Images (*.png *.jpg *.jpeg *.bmp *.dcm)"
        )
        if not path:
            return

        try:
            info = self.manager.load_original(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            return

        pix = cv2_to_pixmap(info["img"])
        self.viewer_current.set_pixmap(pix)
        self.viewer_middle.set_pixmap(pix)
        self._refresh_info(info["img"], info["display_name"])
        self._refresh_history_combo()

        QApplication.processEvents()

        self.combo_action.setEnabled(True)
        self.btn_hist.setEnabled(True)
        self.btn_ai.setEnabled(True)
        self.btn_crop_roi.setEnabled(True)

        self.text_log.append(f"✓ 成功加载: {info['display_name']}")

    def _action_changed(self):
        idx = self.combo_action.currentIndex()
        text = self.combo_action.currentText()

        if "降噪" in text or "裁剪" in text:
            self.btn_set_param.setEnabled(True)
        else:
            self.btn_set_param.setEnabled(False)

        self.btn_apply.setEnabled(idx > 0)

    def set_params(self):
        if not self.manager.has_image():
            return

        act = self.combo_action.currentText()
        img = self.manager.get_current_img()

        if "降噪" in act:
            dlg = DialogDenoise(self)
            dlg_font = QFont("Microsoft YaHei", self.font_size)
            dlg.setFont(dlg_font)
            for widget in dlg.findChildren(QWidget):
                widget.setFont(dlg_font)
            if dlg.exec_():
                self.current_params = dlg.result
                self.text_log.append(f"⚙️ 参数设置: {self.current_params}")

        elif "裁剪" in act:
            h, w = img.shape[:2]
            dlg = DialogCrop(self, img_width=w, img_height=h)
            dlg_font = QFont("Microsoft YaHei", self.font_size)
            dlg.setFont(dlg_font)
            for widget in dlg.findChildren(QWidget):
                widget.setFont(dlg_font)
            if dlg.exec_():
                self.current_params = dlg.result
                self.text_log.append(f"⚙️ 参数设置: {self.current_params}")

    def apply_action(self):
        if not self.manager.has_image():
            return

        act = self.combo_action.currentText()

        try:
            if "降噪" in act:
                res = self.manager.apply_denoise(**self.current_params)
            elif "裁剪" in act:
                res = self.manager.apply_crop(**self.current_params)
            elif "格式转换" in act:
                img = self.manager.get_current_img()
                mode = "gray" if img.ndim == 3 else "rgb"
                res = self.manager.apply_color_convert(mode=mode)
            elif "对齐" in act:
                res = self.manager.apply_align(dx=5, dy=5)
            elif "旋转" in act:
                res = self.manager.apply_rotate(angle=90)
            elif "翻转" in act:
                res = self.manager.apply_flip(mode="h")
            elif "直方图均衡" in act:
                res = self.manager.apply_hist_equalize()
            else:
                return
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            self.text_log.append(f"❌ 错误: {e}")
            return

        self._update_viewers_post_action(res["img"], res["display_name"])
        self.text_log.append(f"✓ 执行: {res['tag']}")
        self.current_params = None

    def do_undo(self):
        info = self.manager.undo()
        if info is None:
            return

        img = info["img"]
        self.viewer_current.set_pixmap(cv2_to_pixmap(img))
        self._refresh_info(img, info["display_name"])
        self._refresh_history_combo()
        self.update_view()
        self.btn_undo.setEnabled(self.manager.can_undo())
        self.btn_redo.setEnabled(self.manager.can_redo())
        self.text_log.append("↶ 撤销")

    def do_redo(self):
        info = self.manager.redo()
        if info is None:
            return

        img = info["img"]
        self.viewer_current.set_pixmap(cv2_to_pixmap(img))
        self._refresh_info(img, info["display_name"])
        self._refresh_history_combo()
        self.update_view()
        self.btn_undo.setEnabled(self.manager.can_undo())
        self.btn_redo.setEnabled(self.manager.can_redo())
        self.text_log.append("↷ 恢复")

    def toggle_roi_mode(self, state):
        if state == Qt.Checked:
            self.viewer_current.set_mode("roi")
            self.text_log.append("📍 ROI 模式开启")
        else:
            self.viewer_current.set_mode("view")
            self.text_log.append("👁️ 查看模式")

    def crop_by_roi(self):
        roi = self.viewer_current.get_last_roi()
        if roi is None:
            self.text_log.append("⚠️ 请先框选 ROI 区域")
            return

        x, y, w, h = roi
        try:
            res = self.manager.apply_crop(x=x, y=y, w=w, h=h)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            return

        self._update_viewers_post_action(res["img"], res["display_name"])
        self.text_log.append(f"✂️ ROI 裁剪: {res['tag']}")

    def show_histogram(self):
        if not self.manager.has_image():
            return
        img = self.manager.get_current_img()
        dlg = DialogHistogram(img, self)
        dlg.exec_()

    def run_ai(self):
        if not self.manager.has_image():
            return

        img = self.manager.get_current_img()
        metadata = parse_metadata(img)
        ann_mgr = AnnotationManager()
        anns = ann_mgr.auto_generate_dummy()

        ai = SimpleAIDiagnosis()
        ai_res = ai.predict(img, metadata, anns)

        ann_boxes = parse_annotations(anns)
        self.viewer_current.set_annotations(ann_boxes)
        self.viewer_current.set_ai_text(ai_res["诊断结论"])

        self.text_log.append(f"🤖 AI 诊断: {ai_res['诊断结论']}")

    def save_current_image(self):
        img = self.manager.get_current_img()
        if img is None:
            QMessageBox.warning(self, "提示", "当前没有可保存的影像")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存影像",
            "processed.png",
            "PNG Image (*.png);;JPG Image (*.jpg)"
        )
        if not path:
            return

        if img.ndim == 2:
            cv2.imwrite(path, img)
        else:
            cv2.imwrite(path, img)

        QMessageBox.information(self, "完成", f"影像已保存：\n{path}")


def run_qt_app():
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 14)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_qt_app()
