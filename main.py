import face_recognition
import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
import os

#   fast1. Process each video frame at 1/5 resolution (though still display it at full resolution)
#   fast2. Only detect faces in every other frame of video.

video_capture = cv2.VideoCapture(0)  # get video from default webcam

IMAGE_FILE = 'images/'
def include_faces():
    known_face_encodings, known_face_names = [], []
    for filename in os.listdir(IMAGE_FILE):
        if os.path.splitext(filename)[1].lower() in ['.jpg','.jpeg','.png']:
            filepath = IMAGE_FILE + filename
            image = face_recognition.load_image_file(filepath)
            image_face = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(image_face)  # known_face_encodings
            known_face_names.append(os.path.splitext(filename)[0])  # known_face_names
            # os.remove(filepath)
    return known_face_encodings, known_face_names

def find_main_face(faces_locations):
    if not faces_locations:
        return -1
    main_face_idx = 0
    max_area = abs((faces_locations[0][0]-faces_locations[0][2])*(faces_locations[0][1]-faces_locations[0][3]))
    for i in range(len(faces_locations)):
        area = abs((faces_locations[i][0]-faces_locations[i][2])*(faces_locations[i][1]-faces_locations[i][3]))
        if area > max_area:
            main_face_idx = i
            max_area = area
    return main_face_idx

WINK_TIME_LIMIT = 3
def main():
    face_locations, face_encodings, face_names = [], [], []
    process_this_frame = True
    known_face_encodings, known_face_names = include_faces()

    count_total, count_known, count_unknown = 0, 0, 0
    wink_time = 0

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)  #   fast1
        rgb_small_frame = small_frame[:, :, ::-1]  # BGR (OpenCV) to RGB (face_recognition uses)

        if process_this_frame:  #   fast2
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                '''
                if True in matches:  # find correct face
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                '''
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)  # use best matching face
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

            main_face_idx = find_main_face(face_locations)
            if main_face_idx >= 0:
                face_landmark = face_landmarks_list[main_face_idx]
                face_location = face_locations[main_face_idx]
                if abs((face_location[0]-face_location[2])*(face_location[1]-face_location[3])) > 3844:
                    count_total += 1
                left_eye, right_eye = face_landmark['left_eye'], face_landmark['right_eye']
                ear_left, ear_right = get_ear(left_eye), get_ear(right_eye)
                if (ear_left < 0.19 and ear_right > 0.2) or (ear_left > 0.2 and ear_right < 0.19):
                    wink_time += 1
                else:
                    wink_time = 0
                if wink_time >= WINK_TIME_LIMIT and face_names[main_face_idx] == "Unknown":
                    known_face_encodings.append(face_encodings[main_face_idx])
                    reg_face_name = input("Please enter your name to register")
                    known_face_names.append(reg_face_name)
                    print("Registered as " + reg_face_name)
                    wink_time = 0

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):  # Display in video
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # box around faces
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)  # label below faces
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # q to quit
            break

    video_capture.release()  # release webcam handling
    cv2.destroyAllWindows()


def get_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

if __name__ == "__main__":
    main()