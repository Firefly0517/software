import cv2
import numpy as np
from typing import List, Tuple

from PyQt5.QtWidgets import (
    QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QFileDialog,
    QComboBox, QCheckBox, QMessageBox, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from gui.image_viewer import ImageViewer
from gui.dialog_denoise import DialogDenoise
from gui.dialog_crop import DialogCrop
from gui.dialog_histogram import DialogHistogram
from core.processing_manager import ProcessingManager
from core.annotation import AnnotationManager
from core.ai_module import SimpleAIDiagnosis
from core.metadata_utils import parse_metadata


# ============================================================
#  工具函数：OpenCV 图像转 Qt Pixmap
# ============================================================
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


# ============================================================
#  工具函数：解析 AI 标注
# ============================================================
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
        except Exception:
            x1, y1, x2, y2 = 30, 30, 80, 80

        result.append((x1, y1, x2, y2, label))
    return result


# ============================================================
#                  主窗口（Apple 风格美化版）
# ============================================================
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能医疗影像处理系统 - Apple 风格 GUI")
        self.setMinimumSize(1600, 900)

        # 核心处理管理器
        self.manager = ProcessingManager()
        self.current_params = None

        # UI
        self._setup_ui()

    # ========================================================
    #  🍎 Apple 风格 UI：卡片 + 圆角 + 留白布局
    # ========================================================
    def _setup_ui(self):
        central = QWidget()
        main = QVBoxLayout()
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(20)

        # ----------------------------------------------------
        #  顶部标题卡片
        # ----------------------------------------------------
        title_card = self._create_card()
        lbl_title = QLabel("智能医疗影像处理系统")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setObjectName("TitleLabel")
        lbl_title.setStyleSheet("font-size:30px; font-weight:700;")
        title_card.layout().addWidget(lbl_title)
        main.addWidget(title_card)

        # ----------------------------------------------------
        #  三图区域卡片（原图 / 中间 / 最终）
        # ----------------------------------------------------
        triple_card = self._create_card()
        triple_layout = QHBoxLayout()
        triple_layout.setSpacing(20)

        # 左：原始图像
        left_box = self._create_image_box("原始影像")
        self.viewer_original = left_box["viewer"]
        triple_layout.addWidget(left_box["card"])

        # 中：中间图像（历史步骤）
        middle_box = self._create_image_box("中间影像（历史步骤）")
        self.viewer_middle = middle_box["viewer"]
        self.combo_history = QComboBox()
        self.combo_history.addItem("（无历史）")
        self.combo_history.currentIndexChanged.connect(self.update_middle_view)
        # 插到标题和图像之间
        middle_box["card"].layout().insertWidget(1, self.combo_history)
        triple_layout.addWidget(middle_box["card"])

        # 右：最终图像
        right_box = self._create_image_box("最终处理影像")
        self.viewer_processed = right_box["viewer"]
        triple_layout.addWidget(right_box["card"])

        triple_card.layout().addLayout(triple_layout)
        main.addWidget(triple_card, stretch=4)



        # ----------------------------------------------------
        #  文件信息卡片（文件名 + shape）
        # ----------------------------------------------------
        info_card = self._create_card()
        info_layout = QHBoxLayout()
        info_layout.setSpacing(20)

        self.label_filename = QLabel("文件名：无")
        self.label_shape = QLabel("shape：无")

        info_layout.addWidget(self.label_filename)
        info_layout.addWidget(self.label_shape)
        info_layout.addStretch()
        info_card.layout().addLayout(info_layout)
        main.addWidget(info_card)

        # ----------------------------------------------------
        #  预处理操作卡片（选择动作 / 参数 / 执行）
        # ----------------------------------------------------
        op_card = self._create_card()
        op_layout = QHBoxLayout()
        op_layout.setSpacing(12)

        self.btn_choose = QPushButton("选择影像")
        # Apple 风里主按钮就保持默认样式即可

        self.combo_action = QComboBox()
        self.combo_action.addItems([
            "请选择预处理动作",
            "降噪（Denoise）",
            "裁剪（数值 Crop）",
            "格式转换（灰度/RGB）",
            "对齐（Align）",
            "旋转（Rotate 90°）",
            "翻转（水平 Flip）",
            "直方图均衡（Histogram Equalization）",
        ])
        self.combo_action.setEnabled(False)

        self.btn_set_param = QPushButton("设置参数")
        self.btn_set_param.setEnabled(False)

        self.btn_apply = QPushButton("执行预处理")
        self.btn_apply.setEnabled(False)

        op_layout.addWidget(self.btn_choose)
        op_layout.addWidget(self.combo_action)
        op_layout.addWidget(self.btn_set_param)
        op_layout.addWidget(self.btn_apply)
        op_layout.addStretch()

        op_card.layout().addLayout(op_layout)
        main.addWidget(op_card)

        # ----------------------------------------------------
        #  ROI / 直方图 / AI 卡片
        # ----------------------------------------------------
        extra_card = self._create_card()
        extra_layout = QHBoxLayout()
        extra_layout.setSpacing(12)

        self.cb_roi_mode = QCheckBox("ROI 框选模式")
        self.btn_crop_roi = QPushButton("使用 ROI 裁剪")
        self.btn_crop_roi.setEnabled(False)

        self.btn_hist = QPushButton("显示直方图")
        self.btn_hist.setEnabled(False)

        self.btn_ai = QPushButton("AI 诊断")
        self.btn_ai.setEnabled(False)

        extra_layout.addWidget(self.cb_roi_mode)
        extra_layout.addWidget(self.btn_crop_roi)
        extra_layout.addWidget(self.btn_hist)
        extra_layout.addWidget(self.btn_ai)
        extra_layout.addStretch()

        extra_card.layout().addLayout(extra_layout)
        main.addWidget(extra_card)

        # ----------------------------------------------------
        #  撤销 / 恢复卡片
        # ----------------------------------------------------
        undo_card = self._create_card()
        undo_layout = QHBoxLayout()
        undo_layout.setSpacing(12)

        self.btn_undo = QPushButton("撤销（Undo）")
        self.btn_redo = QPushButton("恢复（Redo）")
        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)

        undo_layout.addWidget(self.btn_undo)
        undo_layout.addWidget(self.btn_redo)
        undo_layout.addStretch()

        undo_card.layout().addLayout(undo_layout)
        main.addWidget(undo_card)

        # ----------------------------------------------------
        #  日志卡片
        # ----------------------------------------------------
        log_card = self._create_card()
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setMinimumHeight(180)
        log_card.layout().addWidget(self.text_log)
        main.addWidget(log_card)

        # ----------------------------------------------------
        central.setLayout(main)
        self.setCentralWidget(central)

        main.addWidget(info_card, stretch=1)
        main.addWidget(op_card, stretch=1)
        main.addWidget(extra_card, stretch=1)
        main.addWidget(undo_card, stretch=1)
        main.addWidget(log_card, stretch=2)

        # ----------------------------------------------------
        #  信号绑定
        # ----------------------------------------------------
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

    # ========================================================
    #  🍏 Apple 风格：卡片面板（白底 + 圆角 + 细边框）
    # ========================================================
    def _create_card(self):
        card = QFrame()
        card.setObjectName("Card")  # QSS 里用 #Card 来控制
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        card.setLayout(layout)
        return card

    # ========================================================
    #  图像区域：卡片 + 标题 + Viewer
    # ========================================================
    def _create_image_box(self, title: str):
        card = self._create_card()
        card.setMinimumHeight(350)

        # 标题 label
        lbl = QLabel(title)
        lbl.setAlignment(Qt.AlignLeft)
        lbl.setStyleSheet("font-size:18px; font-weight:600; color:#1C1C1E;")

        # 标题行
        title_row = QHBoxLayout()
        title_row.addWidget(lbl)
        title_row.addStretch()

        # 如果是中间卡片，把下拉框加到标题行
        if title.startswith("中间影像"):
            self.combo_history = QComboBox()
            self.combo_history.addItem("（无历史）")
            self.combo_history.currentIndexChanged.connect(self.update_middle_view)
            title_row.addWidget(self.combo_history)

        # 图像 viewer
        viewer = ImageViewer()
        viewer.setObjectName("Card")
        viewer.setMinimumHeight(260)

        # 组合布局
        card.layout().addLayout(title_row)
        card.layout().addWidget(viewer)

        return {"card": card, "viewer": viewer}

    # ========================================================
    #  工具：刷新信息 / 历史
    # ========================================================
    def _refresh_info(self, img, display_name: str):
        self.label_filename.setText(f"文件名：{display_name}")
        self.label_shape.setText(f"shape：{img.shape}")

    def _refresh_history_combo(self):
        self.combo_history.clear()
        tags = self.manager.get_history_descriptions()
        for i, t in enumerate(tags):
            self.combo_history.addItem(f"step{i}: {t}")

    # 中间图像 = 历史任一步
    def update_middle_view(self):
        if not self.manager.has_image():
            return
        idx = self.combo_history.currentIndex()
        img = self.manager.get_image_at_step(idx)
        if img is None:
            return
        self.viewer_middle.set_pixmap(cv2_to_pixmap(img))

    # ========================================================
    #  槽函数：核心行为
    # ========================================================

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择影像", "", "Images (*.png *.jpg *.jpeg *.bmp *.dcm)"
        )
        if not path:
            return

        try:
            info = self.manager.load_original(path)
        except Exception as e:
            QMessageBox.critical(self, "ERROR", str(e))
            return

        pix = cv2_to_pixmap(info["img"])
        self.viewer_original.set_pixmap(pix)
        self.viewer_processed.set_pixmap(pix)
        self.viewer_middle.set_pixmap(pix)

        self._refresh_info(info["img"], info["display_name"])
        self._refresh_history_combo()

        self.combo_action.setEnabled(True)
        self.btn_hist.setEnabled(True)
        self.btn_ai.setEnabled(True)
        self.btn_crop_roi.setEnabled(True)

        self.text_log.append(f"[INFO] 成功加载图像：{info['display_name']}")

    def _action_changed(self):
        act = self.combo_action.currentText()
        # 需要弹参数对话框的操作
        if act.startswith("降噪") or act.startswith("裁剪"):
            self.btn_set_param.setEnabled(True)
        else:
            self.btn_set_param.setEnabled(False)

        # 是否允许执行按钮
        self.btn_apply.setEnabled(self.combo_action.currentIndex() != 0)

    def set_params(self):
        if not self.manager.has_image():
            return

        act = self.combo_action.currentText()
        img = self.manager.get_current_img()

        if act.startswith("降噪"):
            dlg = DialogDenoise(self)
            if dlg.exec_():
                self.current_params = dlg.result
                self.text_log.append(f"[设置] 降噪参数：{self.current_params}")

        elif act.startswith("裁剪"):
            h, w = img.shape[:2]
            dlg = DialogCrop(self, img_width=w, img_height=h)
            if dlg.exec_():
                self.current_params = dlg.result
                self.text_log.append(f"[设置] 裁剪参数：{self.current_params}")

    def apply_action(self):
        if not self.manager.has_image():
            return

        act = self.combo_action.currentText()

        try:
            if act.startswith("降噪"):
                if not self.current_params:
                    QMessageBox.information(self, "提示", "请先设置降噪参数")
                    return
                res = self.manager.apply_denoise(**self.current_params)

            elif act.startswith("裁剪"):
                if not self.current_params:
                    QMessageBox.information(self, "提示", "请先设置裁剪参数")
                    return
                res = self.manager.apply_crop(**self.current_params)

            elif act.startswith("格式转换"):
                img = self.manager.get_current_img()
                mode = "gray" if img.ndim == 3 else "rgb"
                res = self.manager.apply_color_convert(mode=mode)

            elif act.startswith("对齐"):
                res = self.manager.apply_align()

            elif act.startswith("旋转"):
                res = self.manager.apply_rotate(angle=90)

            elif act.startswith("翻转"):
                res = self.manager.apply_flip(mode="h")

            elif act.startswith("直方图均衡"):
                res = self.manager.apply_hist_equalize()

            else:
                QMessageBox.information(self, "提示", "未选择有效预处理动作")
                return

        except Exception as e:
            QMessageBox.critical(self, "ERROR", str(e))
            self.text_log.append(f"[ERROR] 预处理失败：{e}")
            return

        img = res["img"]
        self.viewer_processed.set_pixmap(cv2_to_pixmap(img))
        self._refresh_info(img, res["display_name"])
        self._refresh_history_combo()
        self.update_middle_view()

        self.btn_undo.setEnabled(self.manager.can_undo())
        self.btn_redo.setEnabled(self.manager.can_redo())

        self.text_log.append(f"[执行] {res['tag']} → 保存：{res['saved_path']}")

        self.current_params = None

    def do_undo(self):
        info = self.manager.undo()
        if info is None:
            return

        img = info["img"]
        self.viewer_processed.set_pixmap(cv2_to_pixmap(img))
        self._refresh_info(img, info["display_name"])
        self._refresh_history_combo()
        self.update_middle_view()

        self.btn_undo.setEnabled(self.manager.can_undo())
        self.btn_redo.setEnabled(self.manager.can_redo())

        self.text_log.append("[撤销] 回到上一状态")

    def do_redo(self):
        info = self.manager.redo()
        if info is None:
            return

        img = info["img"]
        self.viewer_processed.set_pixmap(cv2_to_pixmap(img))
        self._refresh_info(img, info["display_name"])
        self._refresh_history_combo()
        self.update_middle_view()

        self.btn_undo.setEnabled(self.manager.can_undo())
        self.btn_redo.setEnabled(self.manager.can_redo())

        self.text_log.append("[恢复] 前进一步")

    def toggle_roi_mode(self, state):
        if state == Qt.Checked:
            self.viewer_processed.set_mode("roi")
            self.text_log.append("[模式] ROI 框选模式开启")
        else:
            self.viewer_processed.set_mode("view")
            self.text_log.append("[模式] 查看模式")

    def crop_by_roi(self):
        roi = self.viewer_processed.get_last_roi()
        if roi is None:
            self.text_log.append("[WARN] 请先在右侧图像框选 ROI")
            return

        x, y, w, h = roi
        try:
            res = self.manager.apply_crop(x=x, y=y, w=w, h=h)
        except Exception as e:
            QMessageBox.critical(self, "ERROR", str(e))
            self.text_log.append(f"[ERROR] ROI 裁剪失败：{e}")
            return

        img = res["img"]
        self.viewer_processed.set_pixmap(cv2_to_pixmap(img))
        self._refresh_info(img, res["display_name"])
        self._refresh_history_combo()
        self.update_middle_view()

        self.text_log.append(f"[ROI 裁剪] {res['tag']} → 保存：{res['saved_path']}")

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
        self.viewer_processed.set_annotations(ann_boxes)
        self.viewer_processed.set_ai_text(ai_res["诊断结论"])

        self.text_log.append(f"[AI] 诊断：{ai_res['诊断结论']}")
