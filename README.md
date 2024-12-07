# Bone Fracture Detection
This project detects bone fractures from X-ray images using EfficientNetB3 for binary classification. It incorporates data augmentation, model evaluation and real-time prediction to accurately identify fractures, enhancing the reliability of diagnostic tools in medical imaging.

## Execution Guide:
1. Run the following command line in your terminal:
   ```
   pip install tensorflow numpy matplotlib opencv-python pillow scikit-learn imbalanced-learn
   ```

2. Download the dataset (link to the dataset: **https://www.kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-xrays?rvi=1**)

3. Copy the path of the dataset folder and paste it into the code

4. After running all the cells, it will create an additional file called `best_model.keras` (this file stores the model, you can download the model from the repositry and directly use it)

5. Enter the path of the image you want in the last cell to check if it has the presence of fracture or not

6. This is how the final output will look like:

   ![image](https://github.com/user-attachments/assets/dbfb756a-20df-4dbf-8b2a-b6328f047636)

   ![image](https://github.com/user-attachments/assets/030ddbb3-f425-4d0b-85f5-211a3de0247e)

## Overview:
The provided code is for building and training a **bone fracture detection** model using deep learning techniques, specifically using a pre-trained EfficientNetB3 model. Here's an overview of the steps and functionalities in the code:

1. **Import Libraries:** 
The code imports several libraries for image processing, data handling, deep learning, and visualization, such as `NumPy`, `Matplotlib`, `OpenCV`, `Seaborn`, `TensorFlow`, and `scikit-learn`.

2. **Dataset Download and Unzipping**
   - The dataset (`bone-fracture-detection-using-xrays.zip`) is downloaded from Kaggle using the `kaggle` API.
   - The dataset is unzipped and the file contents are extracted for further use.

3. **Data Preprocessing**
   - **ImageDataGenerator** is used to apply real-time data augmentation to the images, such as rotation, zoom, and horizontal flipping. These techniques help in preventing overfitting and improving generalization.
   - The data is split into training and validation sets, with the `train_path` and `test_path` variables pointing to the respective directories containing the images.
   - The `flow_from_directory` method is used to load images from the directories for both training and validation, resizing the images to 224x224 pixels and batching them.

4. **Model Building**
   - **EfficientNetB3**, a pre-trained model from the Keras applications module, is used as the base model. The weights are loaded from ImageNet, but the top layers are excluded.
   - The base model layers are frozen to prevent retraining, and additional custom layers are added on top:
     - A `GaussianNoise` layer for regularization.
     - A `GlobalAveragePooling2D` layer to reduce the spatial dimensions.
     - A dense fully-connected layer with 512 units, followed by batch normalization and another `GaussianNoise` layer.
     - A `Dropout` layer for regularization.
     - A final output layer with a sigmoid activation function to classify the images as either "fractured" or "not fractured" (binary classification).
   - The model summary is displayed to show the architecture and number of parameters.

5. **Model Compilation:** The model is compiled using the **binary cross-entropy loss function**, **Adam optimizer**, and metrics like **accuracy**, **precision**, **recall**, and **AUC** (Area Under the Curve).

6. **Model Training**
   - The model is trained for 10 epochs using the training data (`train_generator`) and validation data (`validation_generator`).
   - **ModelCheckpoint** saves the best model based on validation accuracy.
   - **ReduceLROnPlateau** reduces the learning rate if the validation loss plateaus.
   - The training process is carried out with feedback provided via metrics like AUC, precision, recall, and loss values.

7. **Model Evaluation**
   - After training, the model is evaluated using the `accuracy_score` and `classification_report` from `scikit-learn` to assess the performance on the validation dataset.
   - The performance is evaluated in terms of precision, recall, F1-score, and accuracy.

8. **Visualization:** **Accuracy and Loss** plots are generated to visualize the model's performance over the epochs for both training and validation sets. This helps in understanding whether the model is overfitting or underfitting.

9. **Model Prediction**
   - A function `predict_bone_fracture` is defined to predict whether an X-ray image shows a fracture or not.
   - The image is preprocessed (resized and converted into an array), and the model is used to make a prediction.
   - The model outputs a confidence score, which is then displayed along with the predicted label ("Fracture" or "Normal").
   - The predicted label and confidence are visualized on the X-ray image itself using `Matplotlib`.

10. **Example Prediction:** The function is called twice to predict bone fractures in two different images. One image contains a fractured bone, and the other contains a normal bone X-ray.

### Key Points:
- The code uses a **transfer learning** approach by utilizing a pre-trained EfficientNetB3 model as the base model.
- Data augmentation techniques are applied to the training images to improve model generalization.
- The final model is evaluated using a variety of metrics, and its performance is visualized over epochs.
- Predictions on new images are made and displayed with confidence scores.

### Next Steps:
- You can further fine-tune the model by unfreezing some of the layers of the EfficientNetB3 model.
- Hyperparameters like learning rate, batch size, and the number of epochs can be tuned for better performance.
- Additional performance metrics such as the confusion matrix and ROC curves could be visualized for more insights.
