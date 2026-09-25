# Face Detection & Recognition

A computer-vision project for **face detection and recognition** with an interactive **Streamlit** web app.

## ✨ Features

- Face detection using **OpenCV's DNN** (SSD face detector)
- Face recognition model (TensorFlow + scikit-learn)
- Data preprocessing pipeline
- Model training pipeline (`trainer.py`, `model.py`)
- Deployment-ready Streamlit UI (`streamlit_app.py`)

## 🚀 Getting started

```bash
# 1. Clone
git clone https://github.com/FR720/face_detection.git
cd face_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run src/streamlit_app.py
```

## 📁 Structure

```
src/
├── data_preprocessing.py   # data loading & prep
├── face_detector.py        # OpenCV DNN face detection
├── face_recognizer.py      # recognition logic
├── model.py / trainer.py   # model training
└── streamlit_app.py        # web UI
```

## 🛠️ Tech stack

Python · OpenCV · TensorFlow · scikit-learn · NumPy · Streamlit

## 📄 License

MIT
