import numpy as np
from scipy.special import expit
import cv2
import mediapipe as mp
import pickle
import joblib
mp_face_mes = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mes.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
emotion_model = pickle.load(open('Testing/emotion_model.pkl', 'rb'))
pca_model = joblib.load('Testing/pca_model.pkl')
cap = cv2.VideoCapture(0)

mesh_points = None
emotion_labels = ['angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']
emotion_scores = [0]*6

def transform_to_zero_one_numpy(arr):
    if len(arr) == 0:
        return arr
    min_val = np.min(arr)
    max_val = np.max(arr)
    if min_val == max_val:
        return np.zeros_like(arr)
    value_range = max_val - min_val
    transformed_arr = (arr - min_val) / value_range
    return transformed_arr

def predict_emotion():
    global mesh_points, emotion_labels, emotion_scores
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
    landmarks_transformed = pca_model.transform([landmarks_flat])
    pred = emotion_model.predict_proba(landmarks_transformed)[0]
    emotion_scores_noisy = transform_to_zero_one_numpy(pred)
    for score in range(len(emotion_scores)):
        emotion_scores_noisy[score] = expit(10 * (emotion_scores_noisy[score] - 0.5))
        emotion_scores[score] = emotion_scores[score]*0.9 + emotion_scores_noisy[score]*0.1
    pred_index = np.argmax(emotion_scores)
    return emotion_labels[pred_index]

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
    emotion_scores_rounded = [round(score, 2) for score in emotion_scores]
    print(emotion_scores_rounded)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break