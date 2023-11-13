import numpy as np
import cv2
import mediapipe as mp
import json
import networkx as nx
import matplotlib.pyplot as plt

with open('GNN/edges.json') as f:
    FACEMESH_TESSELATION = json.load(f)
mp_face_mes = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mes.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

frame = cv2.imread('GNN/angry.jpeg')
H, W, _ = frame.shape
rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results_mesh = face_mesh.process(rgb_image)

graph = nx.Graph()
vertices = []
edges = []

for edge in FACEMESH_TESSELATION:
    # pega os pontos do triangulo
    pointA_index = edge[0]
    pointB_index = edge[1]
    pointC_index = edge[2]
    graph.add_node(pointA_index)
    graph.add_node(pointB_index)
    graph.add_node(pointC_index)
    # conecta os pontos
    edges.append([pointA_index, pointB_index])
    edges.append([pointB_index, pointC_index])
    edges.append([pointC_index, pointA_index])
    graph.add_edge(pointA_index, pointB_index)
    graph.add_edge(pointB_index, pointC_index)
    graph.add_edge(pointC_index, pointA_index)
    if pointA_index not in vertices:
        vertices.append(pointA_index)
    if pointB_index not in vertices:
        vertices.append(pointB_index)
    if pointC_index not in vertices:
        vertices.append(pointC_index)

# Visualizando o grafo (moved outside the loop)
pos = nx.spring_layout(graph)  # You can use different layout algorithms
nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=8, font_color='black', font_weight='bold', edge_color='gray', linewidths=1, alpha=0.7)
plt.show()

print(len(vertices))
print(len(edges))