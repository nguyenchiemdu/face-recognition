import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os 
import joblib
from lib.ml_function import get_hog_feature

def svm_train_model(path = './cropped_image',save_model = True):
    # Load and preprocess images
    data = []
    labels = []
    
    # Iterate over your data folder
    for student_folder in os.listdir(path):
        for image_name in os.listdir(os.path.join(path, student_folder)):
            image_path = os.path.join(path, student_folder, image_name)
            image = cv2.imread(image_path)
            hog_features = get_hog_feature(image,gray_convert=True)
            
            data.append(hog_features)
            labels.append(student_folder)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train SVM
    svm = SVC(kernel='poly')
    svm.fit(X_train, y_train)

    # Predict
    predictions = svm.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    # Assuming 'svm' is your trained SVM model
    joblib.dump(svm, './models/svm_model.sav')
    return svm,accuracy
