from flask import Flask, request, render_template
from utils import transform_image, get_prediction
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            label = get_prediction(image_path)
            return render_template("result.html", label=label, image_path=image_path)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
