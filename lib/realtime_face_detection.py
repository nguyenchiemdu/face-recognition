import cv2
import joblib
from lib.ml_function import get_hog_feature,face_cascade,load_svm_model

def realtime_face_detection():
    # Open the default camera (0)
    cap = cv2.VideoCapture(0)

    # Load the saved SVM model
    loaded_svm = load_svm_model()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        
        # load saved svm model and predice the face
        gray_image = gray[y:y+h, x:x+w]

        hog_features = get_hog_feature(gray_image)

        # Perform prediction using the loaded SVM model
        predicted_student = loaded_svm.predict([hog_features])
        confidence = loaded_svm.decision_function([hog_features])[0]
        # get max value of confidence list
        max_confidence = max(confidence)
        
        print("Predicted student:", predicted_student[0])
        # Draw the text under the face
        cv2.putText(frame,  f'{predicted_student[0]}', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 
        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
