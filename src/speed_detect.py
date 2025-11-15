import time
import csv
import os
import cv2
import pygame
import numpy as np
from datetime import datetime
import pytz

import supervision as sv
from inference.models.utils import get_roboflow_model

# ----------------------------
# Audio alert setup
# ----------------------------
pygame.mixer.init()

def play_alert_sound():
    pygame.mixer.music.load("output24.wav")
    pygame.mixer.music.play()

# ----------------------------
# Roboflow Inference model
# Using a pre-trained COCO YOLOv8s model
# ----------------------------
model = get_roboflow_model(
    model_id="yolov8n-640",
    api_key="your-api-here"   # <-- put your key here
)

# ----------------------------
# ByteTrack tracker (Supervision)
# ----------------------------
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.3,   # conf threshold to start a track
    lost_track_buffer=60,              # how long a track survives without detection
    minimum_matching_threshold=0.7,    # IOU threshold for matching
    frame_rate=30,                     # will be updated after we read FPS
    minimum_consecutive_frames=2       # require 2 frames before track is "real"
)

# Only keep vehicles (COCO class IDs)
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# ----------------------------
# Geo markers & speed config
# ----------------------------
geo_markers = {
    "A": (160, 174),  # Top-Left
    "B": (365, 200),  # Top-Right
    "C": (3,   280),  # Bottom-Left
    "D": (168, 355)   # Bottom-Right
}

# Real-world distance inside the box (meters)
real_distance_meters = 18.18  # 102 feet converted to meters

# Object tracking dictionary: per ByteTrack ID
object_tracking = {}

# Define EST timezone
est = pytz.timezone("US/Eastern")

def convert_to_est(unix_timestamp: float) -> str:
    """Convert unix ts to human-readable EST time with full seconds."""
    return datetime.fromtimestamp(unix_timestamp, tz=pytz.utc).astimezone(est).strftime("%Y-%m-%d %H:%M:%S")

# ----------------------------
# CSV & output setup
# ----------------------------
csv_filename = "speed_log5.csv"
os.makedirs("speeding_frames5", exist_ok=True)

if not os.path.exists(csv_filename):
    with open(csv_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Object ID",
            "Entry Time (EST)",
            "Exit Time (EST)",
            "Time Taken (s)",
            "Speed (m/s)",
            "Speed (mph)"
        ])

# ----------------------------
# Video stream
# ----------------------------
cap = cv2.VideoCapture(  
    "Paste your updated video URL here"
    
)

# Get FPS and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 13  # fallback if stream doesn't report fps

byte_tracker.frame_rate = int(fps)  # update tracker framerate

frame_time = 1 / fps

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer for output (optional)
out = cv2.VideoWriter(
    "output1.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

process_every_nth_frame = 1
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % process_every_nth_frame != 0:
        continue

    # ----------------------------
    # Roboflow Inference detection
    # conf is your "YOLO conf level" knob here
    # ----------------------------
    results = model.infer(
        image=frame,
        confidence=0.5,      # tweak 0.2–0.5 depending on how noisy it is
        iou_threshold=0.7
    )[0]

    detections = sv.Detections.from_inference(results)

    # Filter to vehicle classes only
    if len(detections) > 0:
        mask = np.isin(detections.class_id, VEHICLE_CLASS_IDS)
        detections = detections[mask]

    if len(detections) == 0:
        # No vehicle detections this frame
        cv2.imshow("Speed Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # ----------------------------
    # ByteTrack tracking
    # ----------------------------
    tracked = byte_tracker.update_with_detections(detections)

    boxes = tracked.xyxy
    track_ids = tracked.tracker_id

    current_time = time.time()

    # ----------------------------
    # Draw and compute speed
    # ----------------------------
    for xyxy, track_id in zip(boxes, track_ids):
        if track_id is None:
            continue

        track_id = int(track_id)

        x_min, y_min, x_max, y_max = map(int, xyxy)
        centroid = ((x_min + x_max) // 2, (y_min + y_max) // 2)

        if track_id not in object_tracking:
            object_tracking[track_id] = {
                "entry_time": None,
                "exit_time": None,
                "speed": None
            }

        # Draw bounding box and ID overlay
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw markers
        for marker, pos in geo_markers.items():
            cv2.circle(frame, pos, 5, (0, 0, 255), -1)

        # -----------------------------------
        # Entry line: crossing bottom segment (C-D)
        # -----------------------------------
        if geo_markers["C"][1] - 15 <= centroid[1] <= geo_markers["D"][1] + 15:
            if object_tracking[track_id]["entry_time"] is None:
                object_tracking[track_id]["entry_time"] = current_time
                print(f"🚗 Object {track_id} entered at {convert_to_est(current_time)} (EST)")

        # -----------------------------------
        # Exit line: crossing top segment (A-B)
        # -----------------------------------
        if (
            object_tracking[track_id]["entry_time"] is not None
            and object_tracking[track_id]["exit_time"] is None
        ):
            if geo_markers["A"][1] - 15 <= centroid[1] <= geo_markers["B"][1] + 15:
                entry_time = object_tracking[track_id]["entry_time"]
                exit_time = current_time
                time_taken = exit_time - entry_time

                if time_taken > 0:
                    speed_mps = real_distance_meters / time_taken
                    speed_mph = speed_mps * 2.23694
                else:
                    speed_mps = 0.0
                    speed_mph = 0.0

                object_tracking[track_id]["exit_time"] = exit_time
                object_tracking[track_id]["speed"] = speed_mph

                entry_time_est = convert_to_est(entry_time)
                exit_time_est = convert_to_est(exit_time)

                print(
                    f"✅ Object {track_id} exited at {exit_time_est} (EST), "
                    f"Speed: {speed_mph:.2f} mph"
                )

                # Save to CSV
                with open(csv_filename, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        track_id,
                        entry_time_est,
                        exit_time_est,
                        round(time_taken, 2),
                        round(speed_mps, 2),
                        round(speed_mph, 2)
                    ])

                # Save speeding frame + play sound
                if speed_mph > 35:
                    frame_filename = f"speeding_frames5/speed_{track_id}_{int(exit_time)}.jpg"
                    cv2.imwrite(frame_filename, frame)
                    print(f"📸 Speeding frame saved: {frame_filename}")
                    play_alert_sound()

        # ----------------------------
        # Speed overlay on box
        # ----------------------------
        if object_tracking[track_id]["speed"] is not None:
            speed_mph = object_tracking[track_id]["speed"]
            color = (0, 255, 0) if speed_mph <= 35 else (0, 0, 255)
            cv2.putText(frame, f"Speed: {speed_mph:.2f} mph",
                        (x_min, y_min - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2)

    # ----------------------------
    # Show + write frame
    # ----------------------------
    cv2.imshow("Speed Detection", frame)
    out.write(frame)

    # Adjust playback speed
    elapsed_time = time.time() - start_time
    sleep_time = max(0, frame_time - elapsed_time)
    time.sleep(sleep_time)
    start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
