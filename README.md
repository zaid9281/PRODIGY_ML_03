# PRODIGY_ML_03

# PRODIGY_ML_04  
## Hand Gesture Recognition using CNN on LeapGestRecog Dataset

This project implements a Convolutional Neural Network (CNN) to classify hand gestures from images. The model is trained on the LeapGestRecog dataset, which contains images of 10 different hand signs.The goal is to build an accurate model for real-time hand gesture recognition.


---

## 📁 Files in this Repository
- `task4_code.ipynb` – Jupyter notebook containing:
  - Data loading and preprocessing
  - CNN model building and training
  - Visualization of training history
  - Model evaluation with classification report and confusion matrix
- `hand_gesture_recognition.h5` – Trained CNN model saved after training.
- `/images` – Folder containing screenshots of:
  - Training and validation accuracy & loss plots
  - Classification report and confusion matrix
  - Combined summary screenshot of all results

---

## Dataset

The dataset used is **LeapGestRecog**, which contains images of 10 distinct hand gestures. The folder structure is as follows:

- `leapGestRecog/` (main dataset folder)  
  - `01_palm`  
  - `02_l`  
  - `03_fist`  
  - `04_fist_moved`  
  - `05_thumb`  
  - `06_index`  
  - `07_ok`  
  - `08_palm_moved`  
  - `09_c`  
  - `10_down`  

Each subfolder contains images representing that specific gesture class.

### Dataset Source

The dataset can be downloaded from Kaggle here:  
[https://www.kaggle.com/gti-upm/leapgestrecog](https://www.kaggle.com/gti-upm/leapgestrecog)

⚠️ **Note:** The dataset is not included here due to file size limits. Please download it manually and place the folder as `leapGestRecog/`.
---

## Project Overview

- Load and preprocess images with data augmentation.
- Build and train a CNN model with 4 convolutional layers and batch normalization.
- Use EarlyStopping and ReduceLROnPlateau callbacks for efficient training.
- Evaluate the model using accuracy, classification report, and confusion matrix.
- Save the trained model for future use.
- Include an example function for single image prediction.

---
## Methodology

1. **Data Preparation**  
   - Load images from the 10 gesture folders  
   - Resize images to 128x128 pixels  
   - Normalize pixel values by scaling between 0 and 1  
   - Apply data augmentation (rotation, shifts, zoom, brightness, flips)  
   - Split data into training (80%) and validation (20%) sets  

2. **Model Architecture**  
   - CNN with 4 convolutional blocks followed by Batch Normalization and MaxPooling  
   - Fully connected dense layers with Dropout for regularization  
   - Softmax output layer for multi-class classification  

3. **Training**  
   - Optimizer: Adam with learning rate 0.001  
   - Loss function: Categorical crossentropy  
   - Early stopping and learning rate reduction callbacks to optimize training  

4. **Evaluation**  
   - Plot training/validation accuracy and loss curves  
   - Generate classification report and confusion matrix on validation data  


## Results

- Training and validation accuracy and loss curves show model convergence.
- Classification report provides precision, recall, and F1-score for each gesture class.
- Confusion matrix highlights classification performance and misclassifications.

---

## ▶️ How to Use

1. Download the LeapGestRecog dataset from Kaggle.
2. Place the dataset folder (`leapGestRecog`) in the project directory or update the path in the code.
3. Run the training script or notebook to train the model.
4. Use the saved model `hand_gesture_recognition.h5` for inference on new images.


To use the trained model for prediction on a new image, use the provided `predict_gesture` function:

```python
predicted_class, confidence = predict_gesture('path_to_image.jpg', model)
print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")





