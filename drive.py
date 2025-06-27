import socketio
import eventlet
from flask import Flask
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import base64

# Create SocketIO server and Flask app
sio = socketio.Server()
app = Flask(__name__)

# Load model
model = load_model('model.h5', compile=False)

# Speed limit
speed_limit = 25

# Preprocessing function to match training
def preprocess(image):
    image = image / 255.0 - 0.5  # Normalize
    return image

@sio.on('telemetry')
def telemetry(sid, data):
    try:
        if data:
            img_string = data["image"]
            image = Image.open(BytesIO(base64.b64decode(img_string)))
            image_array = np.asarray(image)
            image_array = preprocess(image_array)
            image_array = np.expand_dims(image_array, axis=0)

            steering_angle = float(model.predict(image_array, batch_size=1, verbose=0))
            speed = float(data["speed"])
            throttle = 0.25 if speed < speed_limit else 0.1

            print(f"Steering: {steering_angle:.4f}, Throttle: {throttle:.2f}, Speed: {speed:.2f}")
            send_control(steering_angle, throttle)
    except Exception as e:
        print(f"❌ Error in telemetry: {e}")

@sio.on('connect')
def connect(sid, environ):
    print("✅ Connected to simulator")
    send_control(0.0, 0.0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle)
        },
        skip_sid=True
    )

# Run server
if __name__ == "__main__":
    app = socketio.WSGIApp(sio, app)
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 4567)), app)
