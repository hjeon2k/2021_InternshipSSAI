import face_recognition
import cv2
from scipy.spatial import distance as dist
import numpy as np
import os
import schedule
import time
import csv
from datetime import datetime

from snu_visitor_bot import *
from multiprocessing import Process

#   fast1. Process each video frame at 1/5 resolution (though still display it at full resolution)
#   fast2. Only detect faces in every other frame of video.

video_capture = cv2.VideoCapture(0)  # get video from default webcam
FACE_DATA = "face_data/"

def include_faces():
    known_face_encodings, known_face_names = [], []
    for filename in os.listdir(FACE_DATA):
        if os.path.splitext(filename)[1].lower() in ['.jpg','.jpeg','.png']:
            filepath = FACE_DATA + filename
            image = face_recognition.load_image_file(filepath)
            image_face = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(image_face)  # known_face_encodings
            known_face_names.append(os.path.splitext(filename)[0])  # known_face_names
            os.remove(filepath)
    if os.path.isfile(FACE_DATA + "face_encodings.csv"):
        with open(FACE_DATA + "face_encodings.csv", 'r') as f:
            frd = csv.reader(f)
            for line in frd:
                known_face_encodings.append(np.array(line[1:]).astype('float'))
                known_face_names.append(line[0])
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


def main_cam():
    process_this_frame = True
    face_locations, face_names = [], []
    known_face_encodings, known_face_names = include_faces()
    known_face_pass, known_face_inframe = [[] for i in range(len(known_face_encodings))], [[0,0,0,0,0,0,0,0,0,0] for i in range(len(known_face_encodings))]
    unknown_face_encodings, unknown_face_pass, unknown_face_inframe = [], [], []
    fn = 0
    hour = datetime.now().hour
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)  #   fast1
        rgb_small_frame = small_frame[:, :, ::-1]  # BGR (OpenCV) to RGB (face_recognition uses)
        if process_this_frame:  #   fast2
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for i in range(len(known_face_inframe)):
                known_face_inframe[i][fn] = 0
            if len(unknown_face_encodings):
                for i in range(len(unknown_face_inframe)):
                    unknown_face_inframe[i][fn] = 0
            main_face_idx = find_main_face(face_locations)
            for i in range(len(face_locations)):
                if abs((face_locations[i][0]-face_locations[i][2])*(face_locations[i][1]-face_locations[i][3])) < 1200:
                    face_names.append("_")
                    continue
                face_encoding = face_encodings[i]
                known_face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                known_best_match_index = np.argmin(known_face_distances)  # use best matching known face
                known_best_match_distance = known_face_distances[known_best_match_index]
                if known_best_match_distance < 0.35:
                    face_names.append(known_face_names[known_best_match_index])
                    known_face_encodings[known_best_match_index] = (known_face_encodings[known_best_match_index] + face_encoding) / 2
                    if abs((face_locations[i][0] - face_locations[i][2]) * (face_locations[i][1] - face_locations[i][3])) > 3600:
                        known_face_inframe[known_best_match_index][fn] = 1

                elif len(unknown_face_encodings):
                    unknown_face_distances = face_recognition.face_distance(unknown_face_encodings, face_encoding)
                    unknown_best_match_index = np.argmin(unknown_face_distances)  # use best matching unknown face
                    unknown_best_match_distance = unknown_face_distances[unknown_best_match_index]
                    if unknown_best_match_distance < 0.35:
                        face_names.append("Unknown")
                        unknown_face_encodings[unknown_best_match_index] = (unknown_face_encodings[unknown_best_match_index] + face_encoding) / 2
                        if abs((face_locations[i][0] - face_locations[i][2]) * (face_locations[i][1] - face_locations[i][3])) > 3600:
                            unknown_face_inframe[unknown_best_match_index][fn] = 1
                    else:
                        face_names.append("Unknown")
                        unknown_face_encodings.append(face_encoding)
                        unknown_face_inframe.append([0,0,0,0,0,0,0,0,0,0])
                        unknown_face_pass.append([])
                        if abs((face_locations[i][0] - face_locations[i][2]) * (face_locations[i][1] - face_locations[i][3])) > 3600:
                            unknown_face_inframe[-1][fn] = 1
                else:
                    face_names.append("Unknown")
                    unknown_face_encodings.append(face_encoding)
                    unknown_face_inframe.append([0,0,0,0,0,0,0,0,0,0])
                    unknown_face_pass.append([])
                    if abs((face_locations[i][0] - face_locations[i][2]) * (face_locations[i][1] - face_locations[i][3])) > 3600:
                        unknown_face_inframe[-1][fn] = 1

                if i == main_face_idx:
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        reg_face_name = input("Please enter your name to register")
                        print("Registered as " + reg_face_name)
                        if face_names[i] != "Unknown":
                            face_names[i] = reg_face_name
                            known_face_names[known_best_match_index] = reg_face_name
                        elif len(unknown_face_encodings) and unknown_best_match_distance < 0.35:
                            known_face_encodings.append(unknown_face_encodings.pop(unknown_best_match_index))
                            known_face_names.append(reg_face_name)
                            known_face_inframe.append(unknown_face_inframe.pop(unknown_best_match_index))
                            known_face_pass.append(unknown_face_pass.pop(unknown_best_match_index))

            fn = (fn + 1) % 10
            for idx in range(len(known_face_encodings)):
                if known_face_inframe[idx][fn] == 1 and sum(known_face_inframe[idx]) == 1:
                    known_face_pass[idx].append(datetime.now())
            for idx in range(len(unknown_face_encodings)):
                if unknown_face_inframe[idx][fn] == 1 and sum(unknown_face_inframe[idx]) == 1:
                    unknown_face_pass[idx].append(datetime.now())

        process_this_frame = not process_this_frame
        for i in range(len(face_names)):  # Display in video
            name = face_names[i]
            top, right, bottom, left = face_locations[i]
            if abs((top-bottom)*(left-right)) < 1200:
                continue
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5
            if i == main_face_idx:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)

        if datetime.now().hour != hour:
            c_time = datetime.now()
            k_ppl, uk_ppl, k_pass, uk_pass = 0, 0, 0, 0
            for passing in known_face_pass:
                k_pass += len(passing)
                if len(passing):
                    k_ppl += 1
            for passing in unknown_face_pass:
                uk_pass += len(passing)
                if len(passing):
                    uk_ppl += 1
            with open("pass.csv", "a", newline='\n') as fp:
                wr = csv.writer(fp, delimiter=',')
                wr.writerow([c_time.year, c_time.month, c_time.day, c_time.hour, k_ppl, uk_ppl, k_pass, uk_pass])
            known_face_pass = [[] for i in range(len(known_face_encodings))]
            unknown_face_pass = [[] for i in range(len(unknown_face_encodings))]
            hour = datetime.now().hour

        if cv2.waitKey(1) & 0xFF == ord('q') or datetime.now().hour == 19:  # q to quit
            c_time = datetime.now()
            k_ppl, uk_ppl, k_pass, uk_pass = 0, 0, 0, 0
            for passing in known_face_pass:
                k_pass += len(passing)
                if len(passing):
                    k_ppl += 1
            for passing in unknown_face_pass:
                uk_pass += len(passing)
                if len(passing):
                    uk_ppl += 1
            with open("pass.csv", "a", newline='\n') as fp:
                wr = csv.writer(fp, delimiter=',')
                wr.writerow([c_time.year, c_time.month, c_time.day, c_time.hour, k_ppl, uk_ppl, k_pass, uk_pass])

            with open(FACE_DATA + "face_encodings.csv", 'w', newline='') as f:
                fwr = csv.writer(f, delimiter=',')
                fwr.writerows(np.concatenate((np.array(known_face_names).reshape(len(known_face_names), 1), known_face_encodings), axis=1))
            break

    video_capture.release()  # release webcam handling
    cv2.destroyAllWindows()


if not os.path.isfile("pass.csv"):
    with open("pass.csv", "a", newline='\n') as fp:
        wr = csv.writer(fp, delimiter=',')
        wr.writerow(["year", "month", "day", "hour", "known_ppl", "unknown_ppl", "known_pass", "unknown_pass"])

#schedule.every().day.at("08:00").do(main_cam)
def main_cam_daily():
    main_cam()
    #while True:
        #schedule.run_pending()
        #time.sleep(1)


if __name__ == '__main__':
    p1 = Process(target=main_cam_daily) #함수 1을 위한 프로세스
    p2 = Process(target=main_bot)
    p1.start()
    p2.start()
    p1.join()
    p2.join()