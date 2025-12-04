import sys
import cv2
import numpy as np
from typing import List, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QLabel,
    QComboBox, QListWidget, QMessageBox, QCheckBox
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


# ------------------ CV2 → Pixmap ----------------------
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


# ------------------ 标注转换 --------------------------
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


# ============================================================
#                    主 窗 体 MainWindow
# ============================================================
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能医疗影像处理系统 - Qt GUI（高级版）")
        self.setMinimumSize(1600, 900)

        # 核心处理管理器
        self.manager = ProcessingManager()
        self.current_params = None

        # 构建 UI
        self._setup_ui()

    # ==================== UI 构建 ==========================
    def _setup_ui(self):
        central = QWidget()
        main = QVBoxLayout()

        # -----------------------------------------------------
        # 图像显示区域（3 个 viewer）
        # -----------------------------------------------------
        img_layout = QHBoxLayout()

        # 左侧 原图
        left_layout = QVBoxLayout()
        self.label_original_title = QLabel("原始影像")
        self.label_original_title.setAlignment(Qt.AlignCenter)
        self.viewer_original = ImageViewer()
        left_layout.addWidget(self.label_original_title)
        left_layout.addWidget(self.viewer_original)

        # 中间 历史选择图像
        middle_layout = QVBoxLayout()
        self.label_middle_title = QLabel("中间影像（历史步骤）")
        self.label_middle_title.setAlignment(Qt.AlignCenter)
        self.viewer_middle = ImageViewer()

        # 历史选择 ComboBox
        self.combo_history = QComboBox()
        self.combo_history.addItem("（无历史）")
        self.combo_history.currentIndexChanged.connect(self.update_middle_view)

        middle_layout.addWidget(self.label_middle_title)
        middle_layout.addWidget(self.combo_history)
        middle_layout.addWidget(self.viewer_middle)

        # 右侧 最终图像
        right_layout = QVBoxLayout()
        self.label_processed_title = QLabel("最终处理影像")
        self.label_processed_title.setAlignment(Qt.AlignCenter)
        self.viewer_processed = ImageViewer()
        right_layout.addWidget(self.label_processed_title)
        right_layout.addWidget(self.viewer_processed)

        img_layout.addLayout(left_layout, 1)
        img_layout.addLayout(middle_layout, 1)
        img_layout.addLayout(right_layout, 1)

        # -----------------------------------------------------
        # 文件名 + shape
        # -----------------------------------------------------
        info_layout = QHBoxLayout()
        self.label_filename = QLabel("文件名：无")
        self.label_shape = QLabel("shape：无")
        info_layout.addWidget(self.label_filename)
        info_layout.addWidget(self.label_shape)

        # -----------------------------------------------------
        # 预处理操作
        # -----------------------------------------------------
        op_layout = QHBoxLayout()
        self.btn_choose = QPushButton("选择影像")

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

        # -----------------------------------------------------
        # ROI、直方图、AI 诊断
        # -----------------------------------------------------
        extra_layout = QHBoxLayout()
        self.cb_roi_mode = QCheckBox("ROI 框选模式")
        self.btn_crop_roi = QPushButton("使用 ROI 裁剪")
        self.btn_hist = QPushButton("显示直方图")
        self.btn_ai = QPushButton("AI 诊断")

        self.btn_crop_roi.setEnabled(False)
        self.btn_hist.setEnabled(False)
        self.btn_ai.setEnabled(False)

        extra_layout.addWidget(self.cb_roi_mode)
        extra_layout.addWidget(self.btn_crop_roi)
        extra_layout.addWidget(self.btn_hist)
        extra_layout.addWidget(self.btn_ai)

        # -----------------------------------------------------
        # 撤销 / 恢复
        # -----------------------------------------------------
        undo_layout = QHBoxLayout()
        self.btn_undo = QPushButton("撤销（Undo）")
        self.btn_redo = QPushButton("恢复（Redo）")
        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)

        undo_layout.addWidget(self.btn_undo)
        undo_layout.addWidget(self.btn_redo)

        # -----------------------------------------------------
        # 日志
        # -----------------------------------------------------
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setMinimumHeight(180)

        # 加入到主布局
        main.addLayout(img_layout)
        main.addLayout(info_layout)
        main.addLayout(op_layout)
        main.addLayout(extra_layout)
        main.addLayout(undo_layout)
        main.addWidget(self.text_log)

        central.setLayout(main)
        self.setCentralWidget(central)

        # 信号绑定
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

    # ==========================================================
    #           辅助函数：更新 UI
    # ==========================================================
    def _refresh_info(self, img: np.ndarray, display_name: str):
        self.label_filename.setText(f"文件名：{display_name}")
        self.label_shape.setText(f"shape：{img.shape}")

    def _refresh_history_combo(self):
        self.combo_history.clear()
        tags = self.manager.get_history_descriptions()
        for i, t in enumerate(tags):
            self.combo_history.addItem(f"step{i}: {t}")

    def update_middle_view(self):
        if not self.manager.has_image():
            return
        idx = self.combo_history.currentIndex()
        img = self.manager.get_image_at_step(idx)
        if img is None:
            return
        pix = cv2_to_pixmap(img)
        self.viewer_middle.set_pixmap(pix)
        self.label_middle_title.setText(f"中间影像（step{idx}）")

    def _update_viewers_post_action(self, img, display_name):
        pix = cv2_to_pixmap(img)
        self.viewer_processed.set_pixmap(pix)
        self._refresh_info(img, display_name)

        self._refresh_history_combo()
        self.update_middle_view()

        self.btn_undo.setEnabled(self.manager.can_undo())
        self.btn_redo.setEnabled(self.manager.can_redo())

    # ==========================================================
    #                   槽函数：行为逻辑
    # ==========================================================

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

        # 原图 & 初始处理图像显示相同内容
        pix = cv2_to_pixmap(info["img"])
        self.viewer_original.set_pixmap(pix)
        self.viewer_processed.set_pixmap(pix)
        self.viewer_middle.set_pixmap(pix)

        self._refresh_info(info["img"], info["display_name"])
        self._refresh_history_combo()

        # 打开所有按钮
        self.combo_action.setEnabled(True)
        self.btn_hist.setEnabled(True)
        self.btn_ai.setEnabled(True)
        self.btn_crop_roi.setEnabled(True)

        self.text_log.append(f"[INFO] 成功加载图像：{info['display_name']}")

    def _action_changed(self):
        act = self.combo_action.currentText()
        if act.startswith("降噪") or act.startswith("裁剪"):
            self.btn_set_param.setEnabled(True)
        else:
            self.btn_set_param.setEnabled(False)

        if self.combo_action.currentIndex() == 0:
            self.btn_apply.setEnabled(False)
        else:
            self.btn_apply.setEnabled(True)

    def set_params(self):
        if not self.manager.has_image():
            return

        act = self.combo_action.currentText()
        img = self.manager.get_current_img()

        if act.startswith("降噪"):
            dlg = DialogDenoise(self)
            if dlg.exec_():
                self.current_params = dlg.result
                self.text_log.append(f"[设置参数] 降噪：{self.current_params}")

        elif act.startswith("裁剪"):
            h, w = img.shape[:2]
            dlg = DialogCrop(self, img_width=w, img_height=h)
            if dlg.exec_():
                self.current_params = dlg.result
                self.text_log.append(f"[设置参数] 裁剪：{self.current_params}")

    def apply_action(self):
        if not self.manager.has_image():
            return

        act = self.combo_action.currentText()

        try:
            if act.startswith("降噪"):
                res = self.manager.apply_denoise(**self.current_params)

            elif act.startswith("裁剪"):
                res = self.manager.apply_crop(**self.current_params)

            elif act.startswith("格式转换"):
                img = self.manager.get_current_img()
                mode = "gray" if img.ndim == 3 else "rgb"
                res = self.manager.apply_color_convert(mode=mode)

            elif act.startswith("对齐"):
                res = self.manager.apply_align(dx=5, dy=5)

            elif act.startswith("旋转"):
                res = self.manager.apply_rotate(angle=90)

            elif act.startswith("翻转"):
                res = self.manager.apply_flip(mode="h")

            elif act.startswith("直方图均衡"):
                res = self.manager.apply_hist_equalize()

            else:
                QMessageBox.information(self, "提示", "未选择有效动作")
                return

        except Exception as e:
            QMessageBox.critical(self, "ERROR", str(e))
            self.text_log.append(f"[ERROR] {e}")
            return

        self._update_viewers_post_action(res["img"], res["display_name"])
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

    # ---------------- ROI 模式 ----------------
    def toggle_roi_mode(self, state):
        if state == Qt.Checked:
            self.viewer_processed.set_mode("roi")
            self.text_log.append("[模式] ROI 框选 模式开启")
        else:
            self.viewer_processed.set_mode("view")
            self.text_log.append("[模式] 查看 模式")

    def crop_by_roi(self):
        roi = self.viewer_processed.get_last_roi()
        if roi is None:
            self.text_log.append("[WARN] 请先用鼠标框选 ROI")
            return
        x, y, w, h = roi

        try:
            res = self.manager.apply_crop(x=x, y=y, w=w, h=h)
        except Exception as e:
            QMessageBox.critical(self, "ERROR", str(e))
            return

        self._update_viewers_post_action(res["img"], res["display_name"])
        self.text_log.append(f"[ROI 裁剪] {res['tag']} → 保存：{res['saved_path']}")

    # ---------------- 直方图 ----------------
    def show_histogram(self):
        if not self.manager.has_image():
            return
        img = self.manager.get_current_img()
        dlg = DialogHistogram(img, self)
        dlg.exec_()

    # ---------------- AI 诊断 ----------------
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


# ============================================================
# 启动应用
# ============================================================
def run_qt_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_qt_app()
