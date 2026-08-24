import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

# Load deepfake model
model = load_model(
r"C:\Users\Lenovo\OneDrive\Desktop\deepfake_detection_project\models\deepfake_model.h5"
)

video_path = r"C:\Users\Lenovo\OneDrive\Desktop\deepfake_detection_project\realdownload.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize MediaPipe face detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

predictions = []
frame_count = 0

print("Processing video...")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process every 5th frame
    if frame_count % 5 != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)

    if results.detections:

        h, w, _ = frame.shape

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            face = frame[y:y+bh, x:x+bw]

            if face.size == 0:
                continue

            face = cv2.resize(face,(224,224))
            face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

            face = np.array(face,dtype=np.float32)
            face = preprocess_input(face)

            face = np.expand_dims(face,axis=0)

            prediction = model.predict(face,verbose=0)[0][0]

            predictions.append(prediction)

cap.release()

if len(predictions) == 0:
    print("No faces detected")
    exit()

fake_frames = sum(p > 0.5 for p in predictions)
real_frames = len(predictions) - fake_frames

print("Frames analyzed:", len(predictions))
print("Fake frames:", fake_frames)
print("Real frames:", real_frames)

if fake_frames > real_frames:
    print("FAKE VIDEO")
    confidence = fake_frames / len(predictions)
else:
    print("REAL VIDEO")
    confidence = real_frames / len(predictions)

print("Confidence:", round(confidence*100,2), "%")