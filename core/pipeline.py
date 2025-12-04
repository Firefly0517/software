import enum
from typing import Dict, Any

from .io_utils import load_image, save_processed_image
from .metadata_utils import parse_metadata
from .preprocessing import Preprocessor
from .annotation import AnnotationManager
from .ai_module import SimpleAIDiagnosis


class State(enum.Enum):
    IDLE = 0
    IMPORT = 1
    PREPROCESS = 2
    ANNOTATE = 3
    AI_DIAGNOSIS = 4
    OUTPUT = 5


class ImageProcessingPipeline:
    """
    影像处理流程管线：
    空闲 → 导入 → 预处理 → 标注 → AI 诊断 → 输出 → 空闲
    对应你文档中的行为需求和状态图。
    """

    def __init__(self):
        self.state = State.IDLE
        self.preprocessor = Preprocessor()
        self.annotation_manager = AnnotationManager()
        self.ai_model = SimpleAIDiagnosis()

    def run(self, image_path: str, save_output: bool = True) -> Dict[str, Any]:
        logs = []

        def log(msg: str):
            print(msg)
            logs.append(msg)

        # 导入阶段
        log(f"[STATE] {self.state.name} → IMPORT")
        self.state = State.IMPORT

        image = load_image(image_path)
        metadata = parse_metadata(image)

        log(f"[INFO] 影像加载完成，元数据：{metadata}")

        # 预处理阶段
        log("[STATE] IMPORT → PREPROCESS")
        self.state = State.PREPROCESS
        processed_img = self.preprocessor.run_all(image)

        # 标注阶段（这里用假标注）
        log("[STATE] PREPROCESS → ANNOTATE")
        self.state = State.ANNOTATE
        annotations = self.annotation_manager.auto_generate_dummy()
        log(f"[INFO] 生成标注数量：{len(annotations)}")

        # AI 诊断阶段
        log("[STATE] ANNOTATE → AI_DIAGNOSIS")
        self.state = State.AI_DIAGNOSIS
        ai_result = self.ai_model.predict(processed_img, metadata, annotations)
        log(f"[AI] 诊断结果：{ai_result['诊断结论']}")

        # 输出阶段
        log("[STATE] AI_DIAGNOSIS → OUTPUT")
        self.state = State.OUTPUT

        save_path = None
        if save_output:
            save_path = save_processed_image(processed_img, image_path)
            log(f"[INFO] 处理后影像已保存到：{save_path}")

        log("[STATE] OUTPUT → IDLE")
        self.state = State.IDLE

        return {
            "logs": logs,
            "metadata": metadata,
            "annotations": annotations,
            "ai_result": ai_result,
            "processed_img": processed_img,
            "save_path": save_path,
        }
