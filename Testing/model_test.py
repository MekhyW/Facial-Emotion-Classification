import numpy as np
import cv2
import mediapipe as mp
import pickle
mp_face_mes = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mes.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mesh_points = None
emotion_model = pickle.load(open('emotion_model.pkl', 'rb'))

cap = cv2.VideoCapture(0)

def predict_emotion():
    global mesh_points, emotion_model
    if mesh_points is None:
        return None
    nose_tip = mesh_points[4] 
    forehead = mesh_points[151]
    mesh_norm = mesh_points - nose_tip
    scale_factor = np.linalg.norm(forehead - nose_tip)
    if np.isclose(scale_factor, 0):
        scale_factor = 1e-6
    mesh_norm = np.divide(mesh_norm, scale_factor)
    landmarks_flat = mesh_norm.flatten()
    pred = emotion_model.predict([landmarks_flat])
    return pred[0].capitalize()

while True:
    ret, frame = cap.read()
    frame_facemesh = frame.copy()
    empty = np.zeros(frame.shape, dtype=np.uint8)
    H, W, _ = frame_facemesh.shape
    rgb_image = cv2.cvtColor(frame_facemesh, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(rgb_image)
    if results_mesh.multi_face_landmarks:
        mesh_points=np.array([np.multiply([p.x, p.y], [W, H]).astype(int) for p in results_mesh.multi_face_landmarks[0].landmark])
        emotion = predict_emotion()
        if emotion:
            cv2.putText(frame_facemesh, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(empty, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        for faceLms in results_mesh.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=frame_facemesh, landmark_list=faceLms, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
            mp_drawing.draw_landmarks(image=empty, landmark_list=faceLms, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
    cv2.imshow('frame', frame_facemesh)
    cv2.imshow('facemesh-only', empty)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break