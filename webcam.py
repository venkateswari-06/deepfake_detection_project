import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("detector/models/deepfake_model.h5")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

print("Press Q to exit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face,(224,224))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

        face = face / 255.0
        face = np.expand_dims(face,axis=0)

        prediction = model.predict(face,verbose=0)[0][0]

        if prediction > 0.5:
            label = "REAL"
            color = (0,255,0)
            confidence = prediction*100
        else:
            label = "FAKE"
            color = (0,0,255)
            confidence = (1-prediction)*100

        text = f"{label} ({confidence:.2f}%)"

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,text,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.imshow("Live Deepfake Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()