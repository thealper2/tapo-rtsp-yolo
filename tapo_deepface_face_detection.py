import cv2
import argparse
from deepface import DeepFace

parser = argparse.ArgumentParser(description="RTSP Stream Viewer with Face Detection")
parser.add_argument("--ip", type=str, required=True, help="IP Address")
parser.add_argument("--port", type=str, default="554", help="RTSP Port (Default: 554)")
parser.add_argument("--username", type=str, help="Username")
parser.add_argument("--password", type=str, help="Password")
parser.add_argument("--resolution", type=str, choices=["640x480", "1080p"], default="1080p", help="Video Resolution")

args = parser.parse_args()

if args.resolution == "640x480":
    rtsp_url = f"rtsp://{args.username}:{args.password}@{args.ip}:{args.port}/stream2"
else:
    rtsp_url = f"rtsp://{args.username}:{args.password}@{args.ip}:{args.port}/stream1"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Failed to open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    try:
        detected_faces = DeepFace.extract_faces(frame, detector_backend="opencv")

        for face in detected_faces:
            x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    except Exception as e:
        print(f"Error in face detection: {e}")

    cv2.imshow("RTSP Stream with Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()