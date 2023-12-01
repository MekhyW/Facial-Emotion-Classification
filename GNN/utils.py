import pathlib

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION, FACEMESH_LEFT_IRIS, FACEMESH_RIGHT_IRIS

_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

def get_mediapipe_adjacency_matrix():
    FULL_FACEMESH_TESSELATION = FACEMESH_TESSELATION | FACEMESH_LEFT_IRIS | FACEMESH_RIGHT_IRIS
    adjacency_matrix = np.zeros((478, 478), dtype=np.int8)
    for i, j in FULL_FACEMESH_TESSELATION:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
    return adjacency_matrix


def preprocess_frame(rgb_image):
    H, W, _ = rgb_image.shape
    results_mesh = _face_mesh.process(rgb_image)
    mesh_points = np.array([
        np.multiply([p.x, p.y], [W, H]).astype(int) 
        for p in results_mesh.multi_face_landmarks[0].landmark
    ])
    nose_tip = mesh_points[4]
    forehead = mesh_points[151]
    scale_factor = np.linalg.norm(forehead - nose_tip)
    if np.isclose(scale_factor, 0):
        scale_factor = 1e-6
    return results_mesh, mesh_points, scale_factor


def read_image(path_img):
    frame = cv2.imread(str(path_img))
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_meshpoints(mesh_points, filename):
    path_name = pathlib.Path(filename).parent.absolute()
    emotion = path_name.name.split('_')[1]
    path_name = path_name.parent.absolute()
    path_name = path_name.joinpath(f'{emotion}_meshpoints')
    # separete the filename from the path
    filename = pathlib.Path(filename).absolute()
    # create te folder if it doesn't exist
    path_name.mkdir(parents=True, exist_ok=True)
    # save the npz file in the path_name
    file_out = path_name.joinpath(f'{filename.name}.json')
    with open(file_out, 'w') as outfile:
        json.dump(mesh_points.tolist(), outfile)