# Facial-Emotion-Recognition
Classification model for facial expressions, using Mediapipe

## Project Description

### Project 1 - Facial Emotion Recognition with Mediapipe and SVM

In this project, we will use the mediapipe library to detect facial landmarks and use them to train a SVM model to classify facial expressions. The dataset used is on the Kaggle websit (link: https://kaggle.com/datasets/669f5ba44ea30e40a7a42fb066bfa0cb89ca843deee526d633b4803014a49912). The dataset contains pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (Angry, Disgusted, Fear, Happy, Sad, Surprised, Neutral).

### Project 2 - Facial Emotion Recognition with Mediapipe and CNN (For the machine learning course)

In this project, we will use the mediapipe library to detect facial landmarks and use them to train a CNN model to classify facial expressions. The dataset used is on the Kaggle websit (link: https://kaggle.com/datasets/669f5ba44ea30e40a7a42fb066bfa0cb89ca843deee526d633b4803014a49912). The dataset contains pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (Angry, Disgusted, Fear, Happy, Sad, Surprised, Neutral).

### Installation for both projects

1. Clone the repo
   ```sh
   git clone
    ```
2. Install Python packages
    ```sh
    pip install -r requirements.txt
    ```
3. Download the dataset from Kaggle (https://kaggle.com/datasets/669f5ba44ea30e40a7a42fb066bfa0cb89ca843deee526d633b4803014a49912) and and extract its folders to the dataset folder, so it will look like this:
    ```sh
    dataset
    ├── face_angry
    ├── face_disgusted
    ├── face_fear
    ├── face_happy
    ├── face_neutral
    ├── face_sad
    └── face_surprised

    ```

## Usage for project 1

1. Run landmarks.ipynb to generate the landmarks of the faces
2. Run pca.ipynb to generate the pca of the landmarks
3. Run model.ipynb to train the model and test it

## Usage for project 2

1. Run the landmarks_graph.ipynb to generate the meshs of the faces
2. Run model_test_gnn.ipynb to train the model and test it




