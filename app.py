import os
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import csv

app = Flask(__name__)
UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["video"]
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    # 调用你现成的处理函数
    from video_new import process_one_video
    csv_path = os.path.join(OUTPUT_DIR, "web_result.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["video_name", "Squat", "Pushup", "PlankTime"])
    process_one_video(save_path, csv_path)

    # 读结果
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    last = rows[-1]
    return jsonify({
        "squat": last[1], "pushup": last[2], "plank": last[3],
        "video_url": f"/output/out_{filename}"
    })

@app.route("/output/<path:fn>")
def serve_output(fn):
    return send_file(os.path.join(OUTPUT_DIR, fn))

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)