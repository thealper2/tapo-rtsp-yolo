import cv2
from ultralytics import YOLO
from flask import Flask, Response, render_template

app = Flask(__name__)

ip_address = "X.X.X.X"
port = "554"
username = ""
password = ""
rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}/stream1"

model = YOLO('yolov8n.pt')

def generate_frames():
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Failed to open RTSP stream.")
        return
    
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

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    cap.release()

@app.route("/")
def index():
    return render_template('./index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)