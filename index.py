import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from lib.custom_os import make_dir
from lib.svm_train_model import svm_train_model
from lib.realtime_face_detection import realtime_face_detection
from lib.crop_save_face import crop_faces_in_directory,crop_one_face
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.eval('tk::PlaceWindow . center')
        self.root.geometry("400x400")

        self.collect_data = ttk.Button(root, text="Collect Data", command=self.open_collect_data)
        self.collect_data.pack()

        self.crop_all_faces_btn = ttk.Button(root, text="Crop All Faces", command=self.crop_all_faces)
        self.crop_all_faces_btn.pack()

        self.train_model_btn = ttk.Button(root, text="Train Model", command=self.train_model)
        self.train_model_btn.pack()

        self.face_recognition = ttk.Button(root, text="Face Recognition",command=realtime_face_detection)
        self.face_recognition.pack()

    def open_collect_data(self):
        self.window = tk.Toplevel(self.root)
        self.vid = cv2.VideoCapture(0)
        video_width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.window.geometry(f"{video_width}x{video_height+150}")

        self.canvas = tk.Canvas(self.window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        self.user_name_title = ttk.Label(self.window, text="Student Name")
        self.user_name_title.pack()
        self.user_name = ttk.Entry(self.window)
        self.user_name.pack()

        self.id_title = ttk.Label(self.window, text="Student ID")
        self.id_title.pack()
        self.id= ttk.Entry(self.window)
        self.id.pack()

        self.capture = ttk.Button(self.window, text="Capture", command=self.capture_image)
        self.capture.pack()

        self.captured_frames_text = ttk.Label(self.window, text="Captured Frames: 0")
        self.captured_frames_text.pack()
        self.captured_frames = 0

        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)  # Call close_window when the window is closed
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)

    def capture_image(self):
        user_name = self.user_name.get()
        user_id = self.id.get()
        make_dir(f"./data/{user_name}-{user_id}")
        self.loop_capture(user_name,user_id)

    def loop_capture(self, user_name,user_id):
        self.captured_frames += 1
        self.captured_frames_text.config(text=f"Captured Frames: {self.captured_frames}")
        ret, frame = self.vid.read()
        cv2.imwrite(f"./data/{user_name}-{user_id}/{user_name}-{user_id}-{self.captured_frames}.jpg", frame)
        if self.captured_frames < 3:
            self.window.after(1000, self.loop_capture, user_name,user_id)
        else:
            self.captured_frames = 0
            self.captured_frames_text.config(text="Cropping captured images")
            crop_one_face(f"{user_name}-{user_id}")
            self.captured_frames_text.config(text="Done capturing and cropping images")

    def close_window(self):
        if self.vid.isOpened():
            self.vid.release()
        self.root.destroy()
    def train_model(self):
        model, accuracy = svm_train_model()
        message = "Model trained successfully! Accuracy: %s" % accuracy
        self.show_alert_box(message)
    def show_alert_box(self,message = 'Ok'):
        #show alert box
        self.alert_box = tk.Toplevel(self.root)
        self.alert_box.geometry("400x200")
        self.alert_box.title("Alert")
        self.alert_box_label = ttk.Label(self.alert_box, text=message)
        self.alert_box_label.pack()
        self.alert_box_button = ttk.Button(self.alert_box, text="OK", command=self.close_alert_box)
        self.alert_box_button.pack()
    def close_alert_box(self):
        self.alert_box.destroy()
    def crop_all_faces(self):
        crop_faces_in_directory()
        self.show_alert_box("Done cropping all faces")



def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
