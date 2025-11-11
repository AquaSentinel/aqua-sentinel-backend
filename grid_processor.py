# batch_processor.py - Refactored to reuse inference modules
import os
import gc
import glob
import time
from typing import Dict, Any
from PIL import Image
from tqdm import tqdm
from utils.email_utils import send_mail_with_attachment
import io
import zipfile

# Import the inference modules
from inference import ship
from inference import debris

# --- Configuration ---
# Satellite coordinate configuration
BASE_LAT, BASE_LON = 15.01, 85.01  # Bay of Bengal (offset to avoid .0 endings)
LAT_STEP = 0.017297  # ~1.92km spacing
LON_STEP = 0.017297  # ~1.92km spacing


def generate_satellite_coordinates(base_lat=None, base_lon=None):
    """Generate GPS coordinates for 16 satellite image patches in a 4x4 grid.

    Args:
        base_lat: Center latitude of the 4x4 grid
        base_lon: Center longitude of the 4x4 grid
    """
    # Use provided coordinates or fall back to defaults
    if base_lat is None:
        base_lat = BASE_LAT
    if base_lon is None:
        base_lon = BASE_LON

    # Calculate the top-left patch center from the base intersection point
    # Want: base coordinates at intersection between patches (1,1), (1,2), (2,1), (2,2)
    # From intersection point to patch (0,0) center: go 1.5 steps north and 1.5 steps west
    top_left_lat = base_lat + (
        1.5 * LAT_STEP
    )  # 1.5 steps north to reach patch (0,0) center
    top_left_lon = base_lon - (
        1.5 * LON_STEP
    )  # 1.5 steps west to reach patch (0,0) center

    coordinates = []
    for i in range(16):
        row, col = divmod(i, 4)
        lat = top_left_lat - (row * LAT_STEP)  # Subtract to go south as row increases
        lon = top_left_lon + (col * LON_STEP)  # Add to go east as col increases
        coordinates.append(
            {
                "patch_id": f"P{i + 1:02d}",
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
            }
        )
    return coordinates


def process_satellite_timestamp(
    timestamp: str, base_lat: float = None, base_lon: float = None
) -> Dict[str, Any]:
    """
    Process all 16 satellite images for a given timestamp using existing inference modules.
    This reuses the ship and debris inference code instead of duplicating logic.
    """

    # Define source paths - structure: images/source/timestamp/(ship|debris)
    source_timestamp_dir = os.path.join("images", "source", timestamp)
    ship_dir = os.path.join(source_timestamp_dir, "ship")
    debris_dir = os.path.join(source_timestamp_dir, "debris")

    # Validate directories exist
    for dir_path, name in [
        (source_timestamp_dir, "source timestamp"),
        (ship_dir, "ship"),
        (debris_dir, "debris"),
    ]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{name} directory not found: {dir_path}")

    # Get image files
    ship_files = sorted(
        glob.glob(os.path.join(ship_dir, "*.jpg"))
        + glob.glob(os.path.join(ship_dir, "*.png"))
    )
    debris_files = sorted(
        glob.glob(os.path.join(debris_dir, "*.jpg"))
        + glob.glob(os.path.join(debris_dir, "*.png"))
    )

    # Validate we have exactly 16 images
    if len(ship_files) != 16:
        raise ValueError(f"Expected 16 ship images, found {len(ship_files)}")
    if len(debris_files) != 16:
        raise ValueError(f"Expected 16 debris images, found {len(debris_files)}")

    print(f"Processing satellite data for timestamp: {timestamp}")

    # Start timing
    start_time = time.time()

    try:
        # Generate satellite coordinates for 4x4 grid using provided base coordinates
        coordinates = generate_satellite_coordinates(base_lat, base_lon)

        # Create output directories for processed images - structure: images/processed/timestamp/(ship|debris)
        processed_timestamp_dir = os.path.join("images", "processed", timestamp)
        processed_ship_dir = os.path.join(processed_timestamp_dir, "ship")
        processed_debris_dir = os.path.join(processed_timestamp_dir, "debris")
        os.makedirs(processed_ship_dir, exist_ok=True)
        os.makedirs(processed_debris_dir, exist_ok=True)

        patch_data = []
        alerts = []

        # Model paths - use the same paths as the main API
        ship_model_path = os.path.join("models", "ship_detection.onnx")
        debris_model_path = os.path.join("models", "marine_debris_detector.onnx")

        # Process patches with progress bar
        for i in tqdm(range(16), desc="Processing patches", unit="patch"):
            coord_info = coordinates[i]
            lat = coord_info["latitude"]
            lon = coord_info["longitude"]

            # Ship Processing using inference module
            try:
                # Load image as PIL for ship inference
                ship_original_img = Image.open(ship_files[i])

                # Use the optimized detection function that returns actual detection data
                ship_boxes, ship_annotated_img = ship.detect_ships(
                    ship_original_img, ship_model_path
                )

                # Extract detection info from actual results - match original format
                ship_detections = {
                    "has_detections": len(ship_boxes) > 0,
                    "detection_count": len(ship_boxes),
                }

            except Exception as e:
                print(f"Ship processing failed for patch {i}: {e}")
                # Fallback to original image
                ship_annotated_img = Image.open(ship_files[i])
                ship_detections = {"has_detections": False, "detection_count": 0}

            # Debris Processing using inference module
            try:
                # Load image as PIL for debris inference
                debris_original_img = Image.open(debris_files[i])

                # Use the optimized detection function that returns actual detection data
                debris_boxes, debris_annotated_img = debris.detect_debris(
                    debris_original_img, debris_model_path
                )

                # Extract detection info from actual results - match original format
                debris_detections = {
                    "has_detections": len(debris_boxes) > 0,
                    "detection_count": len(debris_boxes),
                }

            except Exception as e:
                print(f"Debris processing failed for patch {i}: {e}")
                # Fallback to original image
                debris_annotated_img = Image.open(debris_files[i])
                debris_detections = {"has_detections": False, "detection_count": 0}

            # Save processed images with lat/lon filenames
            ship_filename = f"{lat}_{lon}.jpg"
            debris_filename = f"{lat}_{lon}.jpg"

            ship_output_path = os.path.join(processed_ship_dir, ship_filename)
            debris_output_path = os.path.join(processed_debris_dir, debris_filename)

            ship_annotated_img.save(ship_output_path, "JPEG", quality=90)
            debris_annotated_img.save(debris_output_path, "JPEG", quality=90)

            # Create patch data (matching original format)
            patch_info = {
                "patch_id": coord_info["patch_id"],
                "coordinates": {
                    "latitude": coord_info["latitude"],
                    "longitude": coord_info["longitude"],
                },
                "detections": {
                    "ship": ship_detections,
                    "debris": debris_detections,
                    "is_alert": ship_detections["has_detections"]
                    and debris_detections["has_detections"],
                },
                "processed_images": {
                    "ship": ship_output_path,
                    "debris": debris_output_path,
                },
            }

            patch_data.append(patch_info)

            # Check for alerts (both ship and debris detected) - match original format
            if patch_info["detections"]["is_alert"]:
                alerts.append(
                    {
                        "patch_id": coord_info["patch_id"],
                        "coordinates": patch_info["coordinates"],
                    }
                )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Summary statistics
        total_ship_detections = sum(
            1 for p in patch_data if p["detections"]["ship"]["has_detections"]
        )
        total_debris_detections = sum(
            1 for p in patch_data if p["detections"]["debris"]["has_detections"]
        )
        total_alerts = len(alerts)

        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Ship detections: {total_ship_detections}/16 patches")
        print(f"Debris detections: {total_debris_detections}/16 patches")
        print(f"Alerts (both detected): {total_alerts}/16 patches")

        # Force garbage collection to free memory
        gc.collect()

        if alerts:
            send_detection_report_email(timestamp, patch_data, alerts)

        return {
            "timestamp": timestamp,
            "processing_time": processing_time,
            "base_coordinates": {
                "latitude": base_lat or BASE_LAT,
                "longitude": base_lon or BASE_LON,
            },
            "grid_config": {"size": "4x4", "lat_step": LAT_STEP, "lon_step": LON_STEP},
            "patches": patch_data,
            "alerts": alerts,
            "summary": {
                "total_patches": 16,
                "ship_detections": total_ship_detections,
                "debris_detections": total_debris_detections,
                "alert_patches": total_alerts,
            },
        }

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        # Force garbage collection on error too
        gc.collect()
        raise
# --- Email Reporting Function ---


def send_detection_report_email(timestamp: str, patch_data: list, alerts: list):
    """
    Sends a single email to jainprinci00@gmail.com with one ZIP file attached
    containing all ship and debris annotated images for alert patches.
    Includes HTML summary of coordinates and patch IDs.
    """
    recipient = "jainprinci00@gmail.com"

    if not alerts:
        print("No alerts found — no email sent.")
        return

    # Build HTML summary body
    html_body = f"""
    <html>
    <body>
        <h2>Marine Alert Report</h2>
        <p>Timestamp: <b>{timestamp}</b></p>
        <p>The following patches show both ship and debris detections:</p>
        <table border="1" cellspacing="0" cellpadding="5">
            <tr>
                <th>Patch ID</th>
                <th>Latitude</th>
                <th>Longitude</th>
            </tr>
    """

    for alert in alerts:
        html_body += f"""
            <tr>
                <td>{alert['patch_id']}</td>
                <td>{alert['coordinates']['latitude']}</td>
                <td>{alert['coordinates']['longitude']}</td>
            </tr>
        """

    html_body += """
        </table>
        <p>All annotated detection images are attached in the ZIP file.</p>
    </body>
    </html>
    """

    # Prepare ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for alert in alerts:
            patch_info = next(
                (p for p in patch_data if p["patch_id"] == alert["patch_id"]), None
            )
            if not patch_info:
                continue

            for label in ["ship", "debris"]:
                img_path = patch_info["processed_images"][label]
                if not os.path.exists(img_path):
                    continue
                try:
                    zip_file.write(img_path, arcname=f"{alert['patch_id']}_{label}.jpg")
                except Exception as e:
                    print(f"Failed to add {label} image for {alert['patch_id']}: {e}")

    # Finalize ZIP
    zip_buffer.seek(0)

    # Send email
    subject = f"[Marine Alert] Ship + Debris Detections ({timestamp})"
    try:
        send_mail_with_attachment(
            subject=subject,
            toEmail=recipient,
            html_body=html_body,
            attachment_name=f"marine_alerts_{timestamp}.zip",
            attachment_bytes=zip_buffer.read(),
        )
        print(f"Email with ZIP attachment sent to {recipient}")
    except Exception as e:
        print(f"Email sending failed: {e}")