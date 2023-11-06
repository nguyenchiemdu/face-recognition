# Face Recognition Project

This project is designed to facilitate facial recognition tasks using Python and associated libraries like OpenCV, Pillow, and tkinter. The project includes functionalities for collecting data, cropping faces, training a support vector machine (SVM) model, and real-time face detection.

## Abstract

The Face Recognition Project is aimed at providing a user-friendly interface for facial recognition tasks. It includes the following key features:

- **Collect Data**: Allows users to capture images from a webcam, associate them with student names and IDs, and save the images for training.
- **Crop All Faces**: Provides functionality to automatically detect and crop faces from captured images and saves them for training purposes.
- **Train Model**: Trains a Support Vector Machine (SVM) model using the collected and cropped face images for recognition purposes.
- **Real-time Face Detection**: Enables real-time face detection through webcam interaction.

## Installation

To run this project, follow these steps:

1. **Clone the Repository**:

   ```
   git clone https://github.com/nguyenchiemdu/face-recognition.git
   ```

2. **Install Required Packages**:
   Ensure you have Python installed. Then, install the required libraries using pip:

   ```
   pip3 install -r requirements.txt
   ```

3. **Run the Application**:
   Execute the following command in the project directory:
   ```
   python3 index.py
   ```

## Instructions to Run

1. Upon running the application (`index.py`), the GUI window will appear with various options:

   - Use the "Collect Data" button to capture images associated with student names and IDs.
   - "Crop All Faces" enables the automatic detection and cropping of faces from captured images.
   - "Train Model" initiates the training of an SVM model using the collected images.
   - "Face Recognition" performs real-time face detection using the webcam.

2. Interact with the different functionalities provided and follow the instructions displayed in the GUI window to collect data, train the model, and perform face recognition.

## Note

This project relies on proper directory structures and the presence of necessary resources. Ensure that the required folders and files exist before executing specific functions.
See our report [here](./Report.md)
