import os 
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_student_dir(path):
    list_dir = os.listdir(path)
    # remove all dir not start with character
    list_dir = [dir for dir in list_dir if dir[0].isalpha()]
    return list_dir