import cv2
import numpy as np
import os

def stretch_image(image_path, output_path):
    axis = np.random.choice(['x', 'y'])
    if axis == 'x':
        stretch_factor_x = np.random.uniform(1.2, 1.5)
        stretch_factor_y = 1.0
    else:
        stretch_factor_x = 1.0
        stretch_factor_y = np.random.uniform(1.2, 1.5)
    original_image = cv2.imread(image_path)
    height, width = original_image.shape[:2]
    new_height = int(height * stretch_factor_y)
    new_width = int(width * stretch_factor_x)
    stretched_image = cv2.resize(original_image, (new_width, new_height))
    cv2.imwrite(output_path, stretched_image)

folder = 'face_sad'
limit = 500
images = os.listdir(folder)
np.random.shuffle(images)
images = images[:limit]
for image in images:
    image_path = os.path.join(folder, image)
    output_path = os.path.join(folder, f"stretched_{image}")
    try:
        stretch_image(image_path, output_path)
    except Exception as e:
        print(e)
        continue