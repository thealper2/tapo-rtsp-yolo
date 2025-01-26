import cv2
import argparse
import numpy as np

def stabilize_frame(prev_frame, current_frame, prev_points):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, prev_points, None)
    good_prev = prev_points[status == 1]
    good_curr = current_points[status == 1]

    if len(good_prev) >= 2 and len(good_curr) >= 2:
        transform_matrix, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
        if transform_matrix is not None:
            stabilize_frame = cv2.warpAffine(current_frame, transform_matrix, (current_frame.shape[1], current_frame.shape[0]))
            return stabilize_frame, current_points
        
    return current_frame, current_points

parser = argparse.ArgumentParser(description="RTSP Stream Viewer")
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

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    stabilized_frame, prev_points = stabilize_frame(prev_frame, frame, prev_points)

    cv2.imshow("Original RTSP Stream", frame)
    cv2.imshow("Stabilized RTSP Stream", stabilized_frame)

    prev_frame = frame
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()