# app.py
import io
import os
import zipfile
from dotenv import load_dotenv

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth, firestore
from PIL import Image
import requests

# Import inference modules
from inference import ship
from inference import debris
from utils.firebase_utils import db
from utils.email_utils import send_mail_with_attachment
from grid_processor import process_satellite_timestamp

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Models are in the root directory, not in a models subdirectory
MODEL_DIR = f"{BASE_DIR}/models"

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
    return jsonify(
        {
            "service": "Aqua Sentinel Backend",
            "description": "Upload two images (ship and debris) to /api/detect as multipart/form-data.",
            "endpoints": {
                "detect": {
                    "path": "/api/detect",
                    "method": "POST",
                    "fields": ["ship", "debris"],
                }
            },
        }
    )


@app.route("/api/signup", methods=["POST"])
def signup():
    # Check for authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return jsonify({"error": "Missing Authorization header"}), 401

    token = auth_header.split(" ")[1]  # Extract token from "Bearer <token>"
    try:
        decoded_token = auth.verify_id_token(token)
        user_email = decoded_token.get("email")
        user_name = decoded_token.get("name")
        user_picture = decoded_token.get("picture")

        # For now, just return info — later you can store this in a DB
        return jsonify(
            {
                "message": f"Welcome {user_name or user_email}!",
                "email": user_email,
                "name": user_name,
                "picture": user_picture,
            }
        )
    except Exception as e:
        print("Error verifying token:", str(e))
        return jsonify({"error": "Invalid or expired token"}), 401


@app.route("/api/login/google", methods=["POST"])
def google_login():
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return jsonify({"error": "Missing Authorization header"}), 401

    token = auth_header.split(" ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        user_email = decoded_token.get("email")
        user_name = decoded_token.get("name")
        picture = decoded_token.get("picture")

        # Optionally store user in DB if first time
        # save_user_to_db(user_email, user_name, picture)

        return jsonify(
            {
                "success": True,
                "message": f"Welcome back, {user_name or user_email}!",
                "email": user_email,
                "name": user_name,
                "picture": picture,
            }
        )
    except Exception as e:
        print("Error verifying token:", str(e))
        return jsonify({"error": "Invalid or expired token"}), 401


@app.route("/api/logout", methods=["POST"])
def logout():
    # In a stateless API, logout usually means client just discards its token.
    # But we can still return a friendly message.
    return jsonify({"success": True, "message": "Logged out successfully"})


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
            return jsonify(
                {"error": "Please upload two image files (ship and debris)."}
            ), 400
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
        _, ship_out_img = ship.detect_ships(
            ship_pil, model_path=os.path.join(MODEL_DIR, "ship_detection.onnx")
        )
    except Exception as e:
        print(f"Ship inference error: {e}")
        return jsonify({"error": f"Ship inference failed: {e}"}), 500

    # Run debris detection
    try:
        _, debris_out_img = debris.detect_debris(
            debris_pil,
            model_path=os.path.join(MODEL_DIR, "marine_debris_detector.onnx"),
        )
    except Exception as e:
        print(f"Debris inference error: {e}")
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
    return send_file(
        memory_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name="detections.zip",
    )


@app.route("/api/report", methods=["POST"])
def receive_report():
    try:
        vessel = request.form.get("vessel")
        location = request.form.get("location")
        email = request.form.get("email")
        toEmail = request.form.get("toEmail")
        notes = request.form.get("notes")
        user_id = request.form.get(
            "userId"
        )  # comes from frontend (auth.currentUser.uid)
        # If the client provided an Authorization header with an ID token, verify it and use the server-verified uid.
        auth_header = request.headers.get("Authorization")
        if auth_header:
            try:
                token = auth_header.split(" ")[1]
                decoded = auth.verify_id_token(token)
                server_uid = decoded.get("uid")
                if server_uid:
                    user_id = server_uid
                    print(f"[receive_report] authenticated request for uid={user_id}")
            except Exception as e:
                print("[receive_report] failed to verify id token:", e)

        print("=== Received Report ===")
        print(f"Vessel   : {vessel}")
        print(f"Location : {location}")
        print(f"Email    : {email}")
        print(f"toEmail  : {toEmail}")
        print(f"Notes    : {notes}")
        print(f"User ID  : {user_id}")

        #  Use uploaded files if provided (e.g. stitched image), otherwise
        #  fetch requested record (recordId) or latest record (4 image URLs) from Firestore.
        uploaded_files = list(request.files.items())

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(
            zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as z:
            # Package any uploaded files first (these are authoritative for this request)
            uploaded_map = {k: v for k, v in uploaded_files}
            for idx, (fieldname, fobj) in enumerate(uploaded_map.items()):
                try:
                    content = fobj.read()
                    # Use the client-provided filename when available; otherwise fall back to a predictable name
                    fname = fobj.filename or f"uploaded_{idx + 1}.png"
                    z.writestr(fname, content)
                except Exception as e:
                    print(f"Warning: failed to read uploaded file {fieldname}:", e)

            # If result images (shipResult/debrisResult) were NOT uploaded, but we have a verified user_id,
            # try to fetch them from the user's latest (or requested) Firestore record so older workflow still works.
            need_ship_result = (
                "shipResult" not in uploaded_map and "ship_result" not in uploaded_map
            )
            need_debris_result = (
                "debrisResult" not in uploaded_map
                and "debris_result" not in uploaded_map
            )

            if need_ship_result or need_debris_result:
                # If we don't have a server-verified user_id, we cannot fetch records.
                if not user_id:
                    # nothing more to do
                    pass
                else:
                    records_ref = (
                        db.collection("users").document(user_id).collection("records")
                    )
                    record_id = request.form.get("recordId")
                    if record_id:
                        doc = records_ref.document(record_id).get()
                        if not doc.exists:
                            print(
                                f"Requested recordId '{record_id}' not found for user {user_id}"
                            )
                            record = None
                        else:
                            record = doc.to_dict()
                    else:
                        docs = list(
                            records_ref.order_by(
                                "createdAt", direction=firestore.Query.DESCENDING
                            )
                            .limit(1)
                            .stream()
                        )
                        record = docs[0].to_dict() if docs else None

                    if record:
                        urls = record.get("images", [])
                        # Expecting [ship-original, debris-original, ship-result, debris-result]
                        if len(urls) >= 4:
                            try:
                                # Only fetch and write missing result images
                                if need_ship_result:
                                    r = requests.get(urls[2], timeout=30)
                                    r.raise_for_status()
                                    z.writestr("ship-result.png", r.content)
                                if need_debris_result:
                                    r2 = requests.get(urls[3], timeout=30)
                                    r2.raise_for_status()
                                    z.writestr("debris-result.png", r2.content)
                            except Exception as e:
                                print(
                                    "Warning: failed to download record result images:",
                                    e,
                                )

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.read()

        # Email content
        subject = f"Aqua Sentinel Report — {vessel or 'Unknown Vessel'}"
        body = f"""
        <h3>Aqua Sentinel Report</h3>
        <p><b>Vessel:</b> {vessel}<br>
           <b>Location:</b> {location}<br>
           <b>Email:</b> {email}<br>
           <b>Notes:</b> {notes}</p>
        """

        # Send email with the ZIP attachment
        send_mail_with_attachment(
            subject=subject,
            toEmail=toEmail,
            html_body=body,
            attachment_name="aqua-report.zip",
            attachment_bytes=zip_bytes,
        )

        return jsonify({"message": "Report received successfully"}), 200

    except Exception as e:
        print("Error while processing report:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/process/<timestamp>", methods=["GET"])
def process_timestamp_route(timestamp):
    """Process 16 satellite images and return detection results."""
    try:
        # Get base coordinates from query parameters
        base_lat = request.args.get("baseLat", type=float)
        base_lon = request.args.get("baseLon", type=float)

        # Check if this is the initial timestamp (defaults to false)
        is_initial = request.args.get("isInitial", "false").lower() == "true"

        result = process_satellite_timestamp(timestamp, base_lat, base_lon, is_initial)
        return jsonify(result), 200
    except FileNotFoundError as e:
        return jsonify({"error": f"Directory not found: {str(e)}"}), 404
    except Exception as e:
        print(f"Error processing timestamp {timestamp}: {str(e)}")
        return jsonify({"error": "Processing failed"}), 500

@app.route("/api/view/<timestamp>/<model_type>/<lat>/<lon>")
def serve_single_grid_image(timestamp, model_type, lat, lon):
    """Serve a single image from the 4x4 grid using lat/lon coordinates."""
    try:
        if model_type not in ["ship", "debris"]:
            return jsonify({"error": "Invalid model type"}), 400

        # Define the processed image directory path - structure: images/processed/timestamp/(ship|debris)
        processed_dir = os.path.join("images", "processed", timestamp, model_type)

        if not os.path.exists(processed_dir):
            return jsonify(
                {"error": f"Processed images not found: {processed_dir}"}
            ), 404

        # Construct filename based on lat/lon coordinates
        image_filename = f"{lat}_{lon}.jpg"
        image_path = os.path.join(processed_dir, image_filename)

        if not os.path.exists(image_path):
            return jsonify({"error": "Image file not found"}), 404

        return send_file(image_path, mimetype="image/jpeg")

    except Exception as e:
        print(f"Error serving grid image for lat={lat}, lon={lon}: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("Starting AquaSentinel Backend Server...")
    print("Models will be loaded on-demand for each request")
    app.run(host="0.0.0.0", port=os.getenv("PORT"), debug=False, threaded=True)
