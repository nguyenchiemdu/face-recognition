import os
import cv2
from lib.ml_function import face_cascade
from lib.custom_os import list_student_dir, make_dir
def crop_faces_in_directory(input_dir = './data', output_dir = './cropped_image'):
    # Loop through all folders (each student's folder)
    for student_folder in list_student_dir(input_dir):
        student_path = os.path.join(input_dir, student_folder)
        output_student_path = os.path.join(output_dir, student_folder)
        make_dir(output_student_path)

        # Loop through all images in the student's folder
        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)

            # Read the image
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize = (250,250))

            # Crop and save each detected face into the respective student's output folder
            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y+h, x:x+w]
                output_file_path = os.path.join(output_student_path, f"face_{img_name}")
                cv2.imwrite(output_file_path, face)

                
def crop_one_face(name,input_path = './data', output_path ='./cropped_image'):
    student_path = f'{input_path}/{name}'
    output_student_path = f'{output_path}/{name}'
    print(output_student_path)
    make_dir(output_student_path)
    # Loop through all images in the student's folder
    for img_name in os.listdir(student_path):
        img_path = os.path.join(student_path, img_name)
        print(img_path)
        # Read the image
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize = (250,250))

        # Crop and save each detected face into the respective student's output folder
        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            output_file_path = os.path.join(output_student_path, f"face_{img_name}")
            print(output_file_path)
            cv2.imwrite(output_file_path, face)

