from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "model", "rice_model_fixed")
IMG_SIZE = 224

CLASSES = ["Arborio", "Basmati", "Brown Rice", "Jasmine", "Karacadag"]

model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully")
# ---------------------------------------


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == "POST":

        # 1️⃣ Get file safely
        file = request.files.get("file")

        if file and file.filename != "":
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            # 2️⃣ Preprocess image
            img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # 3️⃣ Predict
            preds = model.predict(img_array)[0]
            class_index = np.argmax(preds)

            prediction = CLASSES[class_index]
            confidence = round(float(preds[class_index]) * 100, 2)
            image_url = f"uploads/{file.filename}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url
    )


if __name__ == "__main__":
    app.run(debug=True)
