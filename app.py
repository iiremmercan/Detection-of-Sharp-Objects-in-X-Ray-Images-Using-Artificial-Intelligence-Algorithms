from flask import request, Response, Flask, render_template
from PIL import Image
import json
import sqlite3
import threading
import matplotlib.pyplot as plt
import io
import base64
from ultralytics import YOLO

app = Flask(__name__)

conn = sqlite3.connect('object_detection.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS Detections
             ( INTEGER PRIMARY KEY AUTOINCREMENT, object_name TEXT, detection_count INTEGER)''')

@app.route("/")
def root():
    with open("templates/index.html") as file:
        return file.read()

@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    boxes, detected_objects = detect_objects_on_image(Image.open(buf.stream))
    return Response(
        json.dumps(boxes),
        mimetype='application/json'
    )

@app.route("/statistics", methods=["GET"])
def statistics():
    c.execute("SELECT object_name, SUM(detection_count) FROM Detections GROUP BY object_name")
    data = c.fetchall()

    objects = [item[0] for item in data]
    counts = [item[1] for item in data]

    # Matplotlib kütüphanesi ile grafik oluşturma
    fig, ax = plt.subplots()
    ax.bar(objects, counts)
    ax.set_xticklabels(objects, fontsize=8)
    ax.set_xlabel('Sınıf İsmi')
    ax.set_ylabel('Tespit Sayısı')
    ax.set_title('Tespit İstatistikleri')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template("statistics.html", img_data=img_b64)

model = YOLO("best2.pt")
lock = threading.Lock()

def detect_objects_on_image(buf):
    results = model.predict(buf)
    result = results[0]
    output = []
    detected_objects = {}
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        class_name = result.names[class_id]
        prob = round(box.conf[0].item(), 2)

        output.append([x1, y1, x2, y2, class_name, prob])
        if class_name not in detected_objects:
            detected_objects[class_name] = 0
        detected_objects[class_name] += 1

    with lock:
        for obj, count in detected_objects.items():
            c.execute("INSERT INTO Detections (object_name, detection_count) VALUES (?, ?)", (obj, count))
    conn.commit()

    return output, detected_objects

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)