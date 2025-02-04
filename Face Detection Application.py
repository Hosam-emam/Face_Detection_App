import cv2
import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp

st.title('Face Detection Application')

upload = st.file_uploader('Please, choose an image:', type=['png','jpg','webp','jpeg'])

option = st.selectbox('Choose the face detection method: ',options=('MediaPipe','OpenCV'))

def detect_face_OpencCV(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for x,y,width,height in faces:
        cv2.rectangle(img,pt1=(x,y),pt2=(x+width,y+height),color=(255,0,0),thickness=3)
        return img
    
def detect_face_MediaPipe(img):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection() as face_detection:
        result = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if result.detections:
            for detection in result.detections:
                mp_drawing.draw_detection(img,detection)
    return img

if(upload is not None):
    image = Image.open(upload)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if(option == 'OpenCV'):
        result =detect_face_OpencCV(image)
    elif(option == 'MediaPipe'):
        result = detect_face_MediaPipe(image)
    else:
        st.write('Invalid Choice Try Again')

    st.image(result,caption='Result Image',channels='BGR',use_container_width=True)