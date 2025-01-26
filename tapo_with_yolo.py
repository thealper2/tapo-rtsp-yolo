import cv2
import argparse
from ultralytics import YOLO

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
    print("Failed to open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    results = model(frame, verbose=False)

    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > 0.5:
            class_name = model.names[int(class_id)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("RTSP Stream with YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()