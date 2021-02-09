import face_recognition
import cv2
import numpy as np
from scipy.spatial import distance as dist
import os
import schedule
from datetime import datetime
import time

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
    if max_area < 1200:
        return -1
    return main_face_idx

def get_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


WINK_TIME_LIMIT = 3
def main():
    process_this_frame = True
    face_locations, face_names = [], []
    known_face_encodings, known_face_names = include_faces()
    unknown_face_encodings = []
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
            main_face_idx = find_main_face(face_locations)
            for i in range(len(face_locations)):
                if abs((face_locations[i][0]-face_locations[i][2])*(face_locations[i][1]-face_locations[i][3])) < 1200:
                    face_names.append("_")
                    continue
                face_encoding = face_encodings[i]
                known_face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                known_best_match_index = np.argmin(known_face_distances)  # use best matching known face
                known_best_match_distance = known_face_distances[known_best_match_index]
                if known_best_match_distance < 0.3:
                    face_names.append(known_face_names[known_best_match_index])
                    known_face_encodings[known_best_match_index] = (known_face_encodings[known_best_match_index] + face_encoding) / 2

                elif len(unknown_face_encodings):
                    unknown_face_distances = face_recognition.face_distance(unknown_face_encodings, face_encoding)
                    unknown_best_match_index = np.argmin(unknown_face_distances)  # use best matching unknown face
                    unknown_best_match_distance = unknown_face_distances[unknown_best_match_index]
                    if unknown_best_match_distance < 0.3:
                        face_names.append("Unknown")
                        unknown_face_encodings[unknown_best_match_index] = (unknown_face_encodings[unknown_best_match_index] + face_encoding) / 2
                    else:
                        face_names.append("Unknown")
                        unknown_face_encodings.append(face_encoding)
                else:
                    face_names.append("Unknown")
                    unknown_face_encodings.append(face_encoding)
                if i == main_face_idx:
                    face_landmark = face_landmarks_list[i]
                    left_eye, right_eye = face_landmark['left_eye'], face_landmark['right_eye']
                    ear_left, ear_right = get_ear(left_eye), get_ear(right_eye)
                    if (ear_left < 0.19 and ear_right > 0.2) or (ear_left > 0.2 and ear_right < 0.19):
                        wink_time += 1
                    else:
                        wink_time = 0
                    if wink_time >= WINK_TIME_LIMIT or (cv2.waitKey(1) & 0xFF == ord('s')):
                        reg_face_name = input("Please enter your name to register")
                        print("Registered as " + reg_face_name)
                        if face_names[i] != "Unknown":
                            face_names[i] = reg_face_name
                            
                        elif len(unknown_face_encodings) and unknown_best_match_distance < 0.3:
                            unknown_face_encodings.pop(i)
                            known_face_encodings.append(face_encodings[i])
                            known_face_names.append(reg_face_name)
                        wink_time = 0

        process_this_frame = not process_this_frame

        for i in range(len(face_names)):  # Display in video
            print(i, len(face_locations), len(face_names))
            name = face_names[i]
            top, right, bottom, left = face_locations[i]
            if abs((top-bottom)*(left-right)) < 1200:
                continue
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5
            if i == main_face_idx:
                cv2.rectangle(frame, (left, top + 35), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            else:
                cv2.rectangle(frame, (left, top + 35), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or datetime.now().hour == 20:  # q to quit
            print(len(known_face_encodings))
            print(len(unknown_face_encodings))
            break

    video_capture.release()  # release webcam handling
    cv2.destroyAllWindows()


schedule.every().day.at("08:00").do(main)
'''
while True:
    schedule.run_pending()
    time.sleep(1)
'''
main()