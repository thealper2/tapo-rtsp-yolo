import cv2
import argparse
from ultralytics import YOLO
import os
from datetime import datetime
import logging

os.makedirs("records", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")
record_folder = os.path.join("records", today)
output_folder = os.path.join("outputs", today)
os.makedirs(record_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

log_file = os.path.join(output_folder, "log.txt")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("Program started.")

parser = argparse.ArgumentParser(description="RTSP Stream Viewer")
parser.add_argument('--ip', type=str, required=True, help="IP Address")
parser.add_argument('--port', type=str, default='554', help='RTSP Port (default=554)')
parser.add_argument('--username', type=str, help='Username')
parser.add_argument('--password', type=str, help='Password')
parser.add_argument('--resolution', type=str, choices=['640x480', '1080p'], default='1080p', help='Video Resolution')

args = parser.parse_args()

if args.resolution == "640x480":
    rtsp_url = f"rtsp://{args.username}:{args.password}@{args.ip}:{args.port}/stream2"
else:
    rtsp_url = f"rtsp://{args.username}:{args.password}@{args.ip}:{args.port}/stream1"

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    logging.error("Failed to open RTSP stream.")
    exit()

video_filename = os.path.join(record_folder, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

logging.info(f"Video recording started: {video_filename}")

while True:
    ret, frame = cap.read()

    if not ret:
        logging.error("Failed to read frame.")
        break

    out.write(frame)

    results = model(frame, verbose=False)

    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > 0.5:
            class_name = model.names[int(class_id)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            screenshot_filename = os.path.join(output_folder, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{class_name}.jpg")
            cv2.imwrite(screenshot_filename, frame)
            logging.info(f"Detected: {class_name} with confidence {score:.2f}. Screenshot saved: {screenshot_filename}")

    cv2.imshow("RTSP Stream with YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info("Program terminated by user.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
logging.info("Program ended.")
