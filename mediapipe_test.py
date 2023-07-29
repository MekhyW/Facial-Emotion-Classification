import numpy as np
import cv2
import mediapipe as mp
mp_face_mes = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mes.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

image = cv2.imread('eunavida.jpg')
image = cv2.resize(image, (0, 0), fx=2, fy=2)
empty = np.zeros(image.shape, dtype=np.uint8)

results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
        mp_drawing.draw_landmarks(image=empty, landmark_list=face_landmarks, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)

cv2.imshow('image with facemesh', image)
cv2.imshow('facemesh only', empty)
cv2.waitKey(0)