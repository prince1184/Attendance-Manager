import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Constants
PATH_TO_IMAGES = 'Images'
ATTENDANCE_FILE = 'Attendence.csv'

# Initialize lists
images = []
person_names = []

# Load images and extract person names
def load_images_from_path(path):
    image_files = os.listdir(path)
    for file in image_files:
        img = cv2.imread(f'{path}/{file}')
        images.append(img)
        person_names.append(os.path.splitext(file)[0])
    print(f"Loaded images: {image_files}")
    print(f"Person names: {person_names}")

# Encode faces from images
def encode_faces(image_list):
    encodings = []
    for img in image_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings.append(face_recognition.face_encodings(img)[0])
    return encodings

# Mark attendance in the CSV file
def mark_attendance(name):
    with open(ATTENDANCE_FILE, 'r+') as file:
        data_lines = file.readlines()
        recorded_names = [line.split(',')[0] for line in data_lines]
        
        if name not in recorded_names:
            now = datetime.now()
            time_str = now.strftime('%H:%M:%S')
            date_str = now.strftime('%d/%m/%Y')
            file.writelines(f'\n{name},{time_str},{date_str}')
            print(f"Attendance marked for {name}")

# Initialize the system
def initialize_system():
    load_images_from_path(PATH_TO_IMAGES)
    return encode_faces(images)

# Face recognition and attendance loop
def run_face_recognition(known_encodings, known_names):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Resize and convert the frame
        resized_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Detect and encode faces in the current frame
        face_locations = face_recognition.face_locations(resized_frame)
        face_encodings = face_recognition.face_encodings(resized_frame, face_locations)

        for encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(known_encodings, encoding)
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            match_index = np.argmin(face_distances)

            if matches[match_index]:
                name = known_names[match_index].upper()

                # Draw a rectangle and label on the detected face
                y1, x2, y2, x1 = [coord * 4 for coord in face_location]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Mark attendance
                mark_attendance(name)

        # Display the video feed
        cv2.imshow("Face Recognition", frame)

        # Exit on pressing 'Enter' (keycode 13)
        if cv2.waitKey(10) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    print("Initializing system...")
    known_encodings = initialize_system()
    print("Encodings complete. Starting face recognition...")
    run_face_recognition(known_encodings, person_names)
