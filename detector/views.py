import numpy as np
import os
import cv2
import base64
from django.shortcuts import render
from tensorflow.keras.models import load_model

model = load_model("detector/models/deepfake_model.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# HOME PAGE
def home(request):
    return render(request,"home.html")


# IMAGE DETECTION
def image_detection(request):

    result=None
    confidence=None
    face_image=None

    if request.method == "POST" and request.FILES.get('image'):

        uploaded_image = request.FILES['image']
        file_path = os.path.join("media", uploaded_image.name)

        with open(file_path,'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        if len(faces)==0:
            result="No Face Detected"
            return render(request,"image.html",{"result":result})

        x,y,w,h = faces[0]
        face = img[y:y+h,x:x+w]

        face_filename="face_"+uploaded_image.name
        face_path=os.path.join("media",face_filename)

        cv2.imwrite(face_path,face)

        face_image=face_filename

        face=cv2.resize(face,(224,224))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

        face=np.array(face)/255.0
        face=np.expand_dims(face,axis=0)

        prediction=model.predict(face)

        score=prediction[0][0]

        if score>0.5:
            result="REAL IMAGE"
            confidence=round(score*100,2)
        else:
            result="FAKE IMAGE"
            confidence=round((1-score)*100,2)

        return render(request,"image.html",{
            "result":result,
            "confidence":confidence,
            "image":uploaded_image.name,
            "face":face_image
        })

    return render(request,"image.html")


# WEBCAM DETECTION
def webcam_detection(request):

    result=None
    confidence=None

    if request.method=="POST":

        image_data=request.POST.get("image")

        if image_data:

            format,imgstr=image_data.split(';base64,')
            img_bytes=base64.b64decode(imgstr)

            file_path=os.path.join("media","webcam.jpg")

            with open(file_path,"wb") as f:
                f.write(img_bytes)

            img=cv2.imread(file_path)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces=face_cascade.detectMultiScale(gray,1.3,5)

            if len(faces)==0:
                result="No Face Detected"
                return render(request,"webcam.html",{"result":result})

            x,y,w,h=faces[0]
            face=img[y:y+h,x:x+w]

            face=cv2.resize(face,(224,224))
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

            face=np.array(face)/255.0
            face=np.expand_dims(face,axis=0)

            prediction=model.predict(face)

            score=prediction[0][0]

            if score>0.5:
                result="REAL FACE"
                confidence=round(score*100,2)
            else:
                result="FAKE FACE"
                confidence=round((1-score)*100,2)

    return render(request,"webcam.html",{
        "result":result,
        "confidence":confidence
    })