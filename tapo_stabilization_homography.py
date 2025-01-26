import cv2
import argparse
import numpy as np

def stabilize_frame_with_homography(prev_frame, current_frame, prev_keypoints, prev_descriptors):
    orb = cv2.ORB_create()
    current_keypoints, current_descriptors = orb.detectAndCompute(current_frame, None)
    if prev_descriptors is not None and current_descriptors is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_descriptors, current_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        if len(src_pts) >= 4 and len(dst_pts) >= 4:
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                stabilized_frame = cv2.warpPerspective(current_frame, H, (current_frame.shape[1], current_frame.shape[0]))
                return stabilized_frame, current_keypoints, current_descriptors
            
    return current_frame, current_keypoints, current_descriptors

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
orb = cv2.ORB_create()
prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_frame, None)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    stabilized_frame, prev_keypoints, prev_descriptors = stabilize_frame_with_homography(
        prev_frame, frame, prev_keypoints, prev_descriptors
    )

    cv2.imshow("Original RTSP Stream", frame)
    cv2.imshow("Stabilized RTSP Stream", stabilized_frame)

    prev_frame = frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()