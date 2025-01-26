import os
import cv2
import argparse

parser = argparse.ArgumentParser(description="RTSP Stream Viewer")
parser.add_argument('--ip', type=str, required=True, help="IP Address")
parser.add_argument('--port', type=str, default='554', help='RTSP Port (default=554)')
parser.add_argument('--username', type=str, help='Username')
parser.add_argument('--password', type=str, help='Password')
parser.add_argument('--resolution', type=str, choices=['640x480', '1080p'], default='1080p', help='Video Resolution')

args = parser.parse_args()

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
cascade_path = os.path.join(cv2_base_dir, "data/haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
