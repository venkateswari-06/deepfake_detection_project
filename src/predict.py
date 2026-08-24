import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model(r"C:\Users\Lenovo\OneDrive\Desktop\deepfake_detection_project\models\deepfake_model.h5")

# Image path (change this to your test image)
image_path = r"C:\Users\Lenovo\OneDrive\Desktop\deepfake_detection_project\dataset_split\test\real\adham7elym_0z7Dejtv.jpg"

# Read image
img = cv2.imread(image_path)

# Resize to model input size
img = cv2.resize(img, (224,224))

# Normalize
img = img / 255.0

# Expand dimensions
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)[0][0]

# Print result
if prediction > 0.5:
    print("REAL IMAGE")
    print("Confidence:", round(prediction*100,2), "%")
else:
    print("FAKE IMAGE")
    print("Confidence:", round((1-prediction)*100,2), "%")