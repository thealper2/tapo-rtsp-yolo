from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/event', methods=['POST'])
def receive_event():
    try:
        event_data = request.get_json()
        print(f"Received event data: {event_data}")
        return jsonify({
            "status": "success",
            "message": "Event received successfully"
        }), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to receive event"
        }), 400
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)