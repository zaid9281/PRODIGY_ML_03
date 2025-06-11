# PRODIGY_ML_03

# 🐾 Cat vs Dog Image Classification using SVM

This project implements a **Support Vector Machine (SVM)** model to classify images of **cats and dogs** using handcrafted **HOG (Histogram of Oriented Gradients)** features. It is part of my internship tasks at **Prodigy Infotech** under the **Machine Learning domain**.

---

## 📌 Objective

To build a machine learning model that:
- Processes image data (cats and dogs)
- Extracts visual features using HOG
- Classifies images using an SVM with hyperparameter tuning
- Evaluates the performance using precision, recall, and F1-score

---

## 🧰 Tech Stack

- Python  
- NumPy  
- `scikit-image` (`hog`, `resize`, `imread`)  
- `scikit-learn` (`SVC`, `GridSearchCV`, `train_test_split`, `StandardScaler`)  
- `tqdm` – for progress visualization  
- `joblib` – for model saving

---

## 🧠 Methodology

1. **Image Loading**  
   - Loaded and resized cat and dog images from the Kaggle PetImages dataset.

2. **Feature Extraction**  
   - Used HOG (Histogram of Oriented Gradients) to extract shape and edge features.

3. **Data Preprocessing**  
   - Scaled features using `StandardScaler`.

4. **Model Training**  
   - Trained an SVM classifier using RBF kernel.
   - Tuned hyperparameters with `GridSearchCV`.

5. **Evaluation**  
   - Evaluated the model using classification metrics like confusion matrix and classification report.

6. **Model Saving**  
   - Saved the best model using `joblib`.

---

## 📂 Directory Structure

📁 PetImages/
├── 🐱 Cat/
└── 🐶 Dog/

📄 svm_cat_dog_classifier.py
📄 optimized_svm_model.pkl
📄 README.md


---

## 🚀 How to Run

1. Download and unzip the [Kaggle PetImages dataset](https://www.kaggle.com/c/dogs-vs-cats/data).
2. Update the image path in the script:
   ```python
   X, y = load_images(r"C:\Path\To\PetImages")
3. Run the script:
   python svm_cat_dog_classifier.py

📦 Output
- The trained model is saved as optimized_svm_model.pkl, which includes:

- The best SVM model

- Fitted scaler for preprocessing

- HOG configuration used during training
