# Deepfake Detection Project

A Django-based web application for detecting deepfake images and videos, with support for webcam-based real-time detection and a machine learning pipeline for training and evaluating detection models.

## Features

- 🎭 **Image-based detection** — upload an image and check whether it's real or fake
- 🎥 **Webcam detection** — real-time deepfake detection using your webcam
- 🧠 **ML training pipeline** — scripts to clean data, extract frames, merge datasets, and train a detection model
- 🌐 **Django web interface** — simple UI to interact with the detector

## Project Structure

```
deepfake_detection_project/
├── detector/                  # Django app for the detection interface
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   ├── models.py
│   ├── templates/
│   │   ├── home.html
│   │   ├── image.html
│   │   └── webcam.html
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── src/                        # ML pipeline scripts
│   ├── dataclean.py            # Cleans and preprocesses raw data
│   ├── extract_frames.py       # Extracts frames from video files
│   ├── merge.py                # Merges datasets
│   ├── split.py                # Splits data into train/test sets
│   ├── train_model.py          # Trains the deepfake detection model
│   ├── predict.py              # Runs predictions on images
│   └── video_predict.py        # Runs predictions on videos
├── webcam.py                   # Standalone webcam detection script
├── manage.py                   # Django management script
└── requirements.txt            # Python dependencies (add this — see below)
```

> **Note:** Large folders like `venv/`, `dataset_split/`, `extracted_frames/`, `merged_dataset/`, and `media/` are excluded from version control via `.gitignore` since they contain the virtual environment, datasets, and generated files.

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- A webcam (for real-time detection features)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/venkateswari-06/deepfake_detection_project.git
   cd deepfake_detection_project
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run database migrations
   ```bash
   python manage.py migrate
   ```

5. Start the development server
   ```bash
   python manage.py runserver
   ```

6. Open your browser at `http://127.0.0.1:8000/`

### Training the Model

To train the deepfake detection model from scratch:

```bash
python src/extract_frames.py     # Extract frames from raw videos
python src/dataclean.py          # Clean and preprocess the data
python src/split.py              # Split into train/test sets
python src/merge.py              # Merge datasets if needed
python src/train_model.py        # Train the model
```

### Running Predictions

```bash
python src/predict.py --input path/to/image.jpg
python src/video_predict.py --input path/to/video.mp4
```

Or use the webcam script for real-time detection:

```bash
python webcam.py
```

## Tech Stack

- **Backend:** Django
- **ML/Computer Vision:** (add your libraries here — e.g. TensorFlow/PyTorch, OpenCV)
- **Frontend:** HTML/CSS/JS via Django templates

## License

This project is currently unlicensed. Add a license (e.g. MIT) if you plan to share or open-source this work.

## Author

**Venkateswari** — [GitHub](https://github.com/venkateswari-06)
