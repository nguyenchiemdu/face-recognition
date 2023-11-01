import cv2
import joblib
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_hog_feature(image, gray_convert = False):
    image = cv2.resize(image, (64, 128))  # Resize image if needed
    if gray_convert:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Extract features using HOG
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image)
    return hog_features.flatten()

def load_svm_model(path = './models/svm_model.sav'):
    loaded_svm = joblib.load(path)
    return loaded_svm