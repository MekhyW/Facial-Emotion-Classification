from deepface import DeepFace
import numpy as np
import cv2
import mediapipe as mp
mp_face_mes = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mes.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

def predict_emotion_deepface(cropped):
    try:
        emotion = DeepFace.analyze(cropped, actions=['emotion'], detector_backend='mediapipe', enforce_detection=False)
        emotion = emotion[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print(e)
        return None

while True:
    ret, frame = cap.read()
    frame_facemesh = frame.copy()
    rgb_image = cv2.cvtColor(frame_facemesh, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(rgb_image)
    if results_mesh.multi_face_landmarks:
        mesh_points=np.array([np.multiply([p.x, p.y], [frame.shape[1], frame.shape[0]]).astype(int) for p in results_mesh.multi_face_landmarks[0].landmark])
        for faceLms in results_mesh.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=frame_facemesh, landmark_list=faceLms, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
        x1, y1 = np.min(mesh_points, axis=0)
        x2, y2 = np.max(mesh_points, axis=0)
        cropped = frame[y1:y2, x1:x2]
        emotion = predict_emotion_deepface(cropped)
        if emotion:
            cv2.putText(frame_facemesh, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame_facemesh)
        cv2.imshow('cropped', cropped)
    else:
        cv2.imshow('frame', frame)
    cv2.waitKey(1)