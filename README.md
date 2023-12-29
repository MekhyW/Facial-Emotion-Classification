# Facial-Emotion-Recognition
Classification model for facial expressions, using Mediapipe

## Project Description

This project solves the problem of robust, real-time classification of facial expressions in single-face images, with fast predictions for low-power devices.

Using the state-of-the-art Google Mediapipe facial landmark detection model, we can use face meshes (undirected graphs) to train the model, instead of using the raw images. This not only reduces the feature space, but also leaves the classification step immune to changes in lighting, noise, rotation/translation/scale, and other factors.

The dataset used is a combination of the FER-2013 dataset and other public materials. It can be found at https://kaggle.com/datasets/669f5ba44ea30e40a7a42fb066bfa0cb89ca843deee526d633b4803014a49912

### Approach 1 - PCA and SVM

The distances from each point in each face mesh to a fixed point are obtained, normalized and saved to intermediate files. Then, Principal Component Analysis is used to reduce the dimensionality of the data, and the resulting data is used to train a the final model.

According to the results, the best model tested is a Support Vector Machine with a RBF kernel, using 50 components in the PCA step, StandardScaler for normalization, and C=5 for the SVM.

### Approach 2 - GNN

The normalized X,Y coordinates of each point in each face mesh are obtained and saved to intermediate files, along with an adjacency matrix for each face mesh. Then, a Graph Neural Network built using Keras and Spektral is used to train the final model.

### Installation for both projects

1. Clone the repo
   ```sh
   git clone
    ```
2. Install Python packages
    ```sh
    pip install -r requirements.txt
    ```
3. Download the dataset from Kaggle and and extract its folders to the repository root folder, so it will look like this:
    ```sh
    root
    ├── face_angry
    ├── face_disgusted
    ├── face_fear
    ├── face_happy
    ├── face_neutral
    ├── face_sad
    └── face_surprised

    ```

## Usage for approach 1

1. Run landmarks.ipynb to generate the landmarks of the faces
2. Run pca.ipynb to generate the pca of the landmarks
3. Run model.ipynb to train the model and test it

## Usage for approach 2

1. Run the landmarks_graph.ipynb to generate the meshes of the faces
2. Run model_test_gnn.ipynb to train the model and test it




