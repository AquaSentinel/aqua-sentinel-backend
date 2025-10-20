# app.py
import io
import os
import zipfile
from typing import List

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

# Import inference modules (they are designed to lazy-load models)
from inference import ship as ship_infer
from inference import debris as debris_infer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Models are in the root directory, not in a models subdirectory
MODEL_DIR = BASE_DIR

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXT

@app.route("/", methods=["GET"])
def index():
        """
        Root route - simple service info / quick usage help.
        """
        return jsonify({
            "service": "Aqua Sentinel Backend",
            "description": "Upload two images (ship and debris) to /api/detect as multipart/form-data.",
            "endpoints": { 
                "detect": {
                    "path": "/api/detect",
                    "method": "POST",
                    "fields": ["ship", "debris"]
                }
            }
        })


@app.route("/api/detect", methods=["POST"])
def detect_route():
    """
    Accepts multipart/form-data with two files:
      - preferred form field names: 'ship' and 'debris'
      - or any two uploaded files (first -> ship, second -> debris)

    Returns a ZIP file containing:
      - ship_output.jpg
      - debris_output.jpg
    """
    # Resolve uploaded files
    if "ship" in request.files and "debris" in request.files:
        ship_file = request.files["ship"]
        debris_file = request.files["debris"]
    else:
        files = list(request.files.values())
        if len(files) < 2:
            return jsonify({"error": "Please upload two image files (ship and debris)."}), 400
        ship_file, debris_file = files[0], files[1]

    # Validate
    if ship_file.filename == "" or debris_file.filename == "":
        return jsonify({"error": "Missing filename in upload."}), 400
    if not allowed_file(ship_file.filename) or not allowed_file(debris_file.filename):
        return jsonify({"error": "Unsupported file extension."}), 400

    # Read into PIL
    try:
        ship_pil = Image.open(io.BytesIO(ship_file.read())).convert("RGB")
        debris_pil = Image.open(io.BytesIO(debris_file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to read uploaded images: {e}"}), 400

    # Ensure models are available in MODEL_DIR path (modules will look there by default)
    # (The inference modules will try to load models lazily when detect_image is called.)
    # Run ship detection
    try:
        buf_ship = io.BytesIO()
        ship_pil.save(buf_ship, format="PNG")
        ship_bytes = buf_ship.getvalue()
        out_bytes = ship_infer.run_ship(ship_bytes, model_path=os.path.join(MODEL_DIR, "ship_detection.onnx"))
        ship_out_img = Image.open(io.BytesIO(out_bytes)).convert("RGB")
    except Exception as e:
        # On failure, return original image with a diagnostic text drawn (the module also handles safe fallback)
        return jsonify({"error": f"Ship inference failed: {e}"}), 500

    # Run debris detection
    try:
        debris_out_img = debris_infer.detect_image(debris_pil, model_path=os.path.join(MODEL_DIR, "marine_debris_detector.onnx"))
    except Exception as e:
        return jsonify({"error": f"Debris inference failed: {e}"}), 500

    # Package into zip
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        b = io.BytesIO()
        ship_out_img.save(b, format="PNG")
        zf.writestr("ship-result.png", b.getvalue())

        b2 = io.BytesIO()
        debris_out_img.save(b2, format="PNG")
        zf.writestr("debris-result.png", b2.getvalue())

    memory_file.seek(0)
    return send_file(memory_file, mimetype="application/zip", as_attachment=True, download_name="detections.zip")


if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=5050, debug=True)
