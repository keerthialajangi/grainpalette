# 🌾 GrainPalette – Rice Classification System

An end-to-end **Deep Learning–powered rice grain classification system** that identifies rice types from images using **CNN and MobileNetV4**, deployed as a **Flask web application**.

This project demonstrates practical skills in **computer vision, deep learning, model training, and full-stack ML deployment**, making it suitable for academic projects and industry portfolios.

---

## 🚀 Features

* 📸 Upload a rice grain image and get instant predictions
* 🧠 Deep Learning model trained on custom rice dataset
* 📊 Confidence scores for each rice class
* 🎨 Clean and user-friendly Flask UI
* 🔁 End-to-end pipeline: data → training → deployment

---

## 🧠 Tech Stack

* **Programming Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Model Architecture:** CNN, MobileNetV4 (Transfer Learning)
* **Web Framework:** Flask
* **Frontend:** HTML, CSS, Bootstrap
* **Others:** NumPy, Pillow, OpenCV

---

## 📁 Project Structure

```
GrainPalette-Flask/
│── app.py                  # Flask application
│── retrain_model.py        # Model training script
│── requirements.txt        # Dependencies
│── README.md               # Project documentation
│
├── model/
│   └── rice_model.keras    # Trained model
│
├── rice_data/
│   ├── train/
│   └── test/
│
├── static/
│   ├── css/
│   └── uploads/
│
├── templates/
│   └── index.html
│
└── venv/   (ignored)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/keerthialajangi/grainpalette.git
cd GrainPalette-Flask
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training (Optional)

If you want to retrain the model:

```bash
python retrain_model.py
```

The trained model will be saved inside the `model/` folder.

---

## ▶️ Run the Flask App

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## 📊 Output

* Predicted rice type
* Confidence score
* Uploaded image preview

---

## 🎯 Use Cases

* Smart agriculture systems
* Quality inspection in food industry
* Educational & academic projects
* AI/ML portfolio demonstration

---

## 📌 Resume Description (Short)

> Developed an AI-powered rice classification system using CNN and MobileNetV4 with a Flask-based web interface for real-time image prediction.

---

## 👤 Author

**Keerthi Alajangi**
B.Tech Computer Science Engineering
GitHub: [https://github.com/keerthialajangi](https://github.com/keerthialajangi)

---

## ⭐ Acknowledgements

* TensorFlow & Keras
* Flask Framework
* Open-source ML community

---

⭐ *If you like this project, don’t forget to star the repository!*
