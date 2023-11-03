import requests
import time
from datetime import datetime
# URL of the server endpoint to which you want to send the data
url = 'https://face-attendance-gnl9.onrender.com/attendance'




def send_attendance_data(predict_result):
    try:
        [student_name,student_id] = str(predict_result).split('-')
        # get current time to millisecond

        current_time = datetime.now().timestamp()
        data_to_send = {
            'name': student_name,
            'id': student_id,
            'time': current_time
        }
        print(data_to_send)
        response = requests.post(url, json=data_to_send)
        if response.status_code == 200:  # Successful response code (e.g., 200 OK)
            print("Data sent successfully!")
            print("Server Response:", response.text)
        else:
            print("Failed to send data. Status code:", response.status_code)
    except:
        print("Server error")


