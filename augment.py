import albumentations
import cv2
import numpy as np
import os

folder = 'face_surprised'
limit = 3717

transform_pipeline = albumentations.Compose([
    albumentations.GridDistortion(p=0.5, num_steps=5, distort_limit=0.2),
    albumentations.OpticalDistortion(p=0.5, distort_limit=0.2, shift_limit=0.05),
    albumentations.HorizontalFlip(p=0.5),
])

images = os.listdir(folder)
np.random.shuffle(images)
images = images[:limit]
for image in images:
    image_path = os.path.join(folder, image)
    output_path = os.path.join(folder, f"augmented_{image}")
    try:
        image = cv2.imread(image_path)
        transformed = transform_pipeline(image=image)
        transformed_image = transformed["image"]
        cv2.imwrite(output_path, transformed_image)
    except Exception as e:
        print(e)
        continue