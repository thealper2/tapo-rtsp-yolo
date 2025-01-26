import cv2
import argparse

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

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
	print("Failed to open RTSP stream.")
	exit()

while True:
	ret, frame = cap.read()

	if not ret:
		print("Failed to read frame.")
		break

	cv2.imshow("RTSP Stream", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
