import os
from flask import Flask, render_template, request, send_from_directory

from core.pipeline import ImageProcessingPipeline
from config import RAW_IMAGE_DIR, PREPROCESSED_IMAGE_DIR

app = Flask(__name__)
pipeline = ImageProcessingPipeline()


@app.route("/raw/<path:filename>")
def raw_image(filename):
    """提供原始图像访问"""
    return send_from_directory(RAW_IMAGE_DIR, filename)


@app.route("/preprocessed/<path:filename>")
def processed_image(filename):
    """提供预处理后图像访问"""
    return send_from_directory(PREPROCESSED_IMAGE_DIR, filename)


@app.route("/", methods=["GET", "POST"])
def index():
    logs = None
    ai_result = None
    raw_filename = None
    processed_filename = None

    if request.method == "POST":
        if "image_file" not in request.files:
            return "没有上传文件", 400

        file = request.files["image_file"]
        if file.filename == "":
            return "文件名为空", 400

        # 保存原始图像
        os.makedirs(RAW_IMAGE_DIR, exist_ok=True)
        raw_filename = file.filename
        raw_save_path = os.path.join(RAW_IMAGE_DIR, raw_filename)
        file.save(raw_save_path)

        # 调用处理流程
        result = pipeline.run(raw_save_path)
        logs = result["logs"]
        ai_result = result["ai_result"]

        # 预处理后影像文件名
        processed_path = result.get("save_path")
        if processed_path:
            processed_filename = os.path.basename(processed_path)

    return render_template(
        "index.html",
        logs=logs,
        ai_result=ai_result,
        raw_filename=raw_filename,
        processed_filename=processed_filename,
    )


def run_web_app():
    app.run(host="0.0.0.0", port=5000, debug=True)
