import cv2
import argparse
import numpy as np

def apply_fourier_transform(frame, cutoff_frequency=30):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray_frame)
    f_transform_shifted = np.fft.fftshift(f_transform)
    rows, cols = gray_frame.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 1
    f_transform_shifted_filtered = f_transform_shifted * mask
    f_transform_filtered = np.fft.ifftshift(f_transform_shifted_filtered)
    denoised_frame = np.fft.ifft2(f_transform_filtered)
    denoised_frame = np.abs(denoised_frame)
    denoised_frame = np.uint8(cv2.normalize(denoised_frame, None, 0, 255, cv2.NORM_MINMAX))
    return denoised_frame

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

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    cv2.imshow("Original RTSP Stream", frame)

    denoised_frame = apply_fourier_transform(frame)

    cv2.imshow("Denoised RTSP Stream", denoised_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()