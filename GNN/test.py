import numpy as np
import cv2
import mediapipe as mp
import json

with open('GNN/edges.json') as f:
    FACEMESH_TESSELATION = json.load(f)
mp_face_mes = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mes.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

frame = cv2.imread('GNN/a.jpg')
H, W, _ = frame.shape
rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results_mesh = face_mesh.process(rgb_image)

vertices = []
edges = []

for edge in FACEMESH_TESSELATION:
    pointA_index = edge[0]
    pointB_index = edge[1]
    pointC_index = edge[2]
    edges.append([pointA_index, pointB_index])
    edges.append([pointB_index, pointC_index])
    edges.append([pointC_index, pointA_index])
    if pointA_index not in vertices:
        vertices.append(pointA_index)
    if pointB_index not in vertices:
        vertices.append(pointB_index)
    if pointC_index not in vertices:
        vertices.append(pointC_index)

print(len(vertices))
print(len(edges))