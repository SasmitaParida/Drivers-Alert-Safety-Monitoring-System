# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:36:04 2024

@author: hp
"""

from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import pandas as pd

print('start')
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])    
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

#Calculate lip distance for Yawn
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

def internal_cam_fun(argument1):
    thresh_Drowsiness = 0.25
    frame_check = 20
    thresh_Yawn = 22
    frame_Yawn = 5
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("D:/hp/Desktop/project/shape_predictor_68_face_landmarks.dat")
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    video_Input = argument1
    video_Output = "D:/hp/Desktop/project/Output/TheOutput.mp4"
    
    cap = cv2.VideoCapture(video_Input)
    writer = None
    
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] {} :Total frames in Video".format(total))
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1
     
    count_Drowsiness=0
    count_Yawn = 0
    indicator_Drowsiness = 0
    indicator_Yawn = 0
    alert_Video = 0
    indicator_center_road_look = 0 
    count_NOT_center_road_look=0
    DROWSINESS_alert_Video = 0
    YAWN_alert_Video = 0
    DISTRACTION_alert_Video = 0
    
    mycolumns = ['Alert_Event', 'Alert_Event_Start_Time','Alert_Event_Stop_Time','Duration_Seconds','Video_Snippet_path']
    df = pd.DataFrame(columns=mycolumns)
    idx = 0
    Alert_Event = ''
    start_time_stamp = ''
    end_time_stamp = ''
    datetimeFormat = '%Y-%B-%d %H:%M:%S'
    
    i=0
    while cap.isOpened():
        i = i+1
        print('while loop=',i)
        ret, frame=cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=450)
        use_frame= 0 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
    
        for subject in subjects:
            
            indicator_center_road_look = 1 
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
    
            distance = lip_distance(shape)
            
            if ear < thresh_Drowsiness:                   
                count_Drowsiness += 1
                if count_Drowsiness >= frame_check:
                    indicator_Drowsiness = 1 
            else:
                count_Drowsiness = 0
                indicator_Drowsiness = 0
    
            if distance > thresh_Yawn:
                count_Yawn += 1
                if count_Yawn >= frame_Yawn:
                    indicator_Yawn = 1
            else:
                count_Yawn = 0
                indicator_Yawn = 0
    
            if (indicator_Drowsiness == 1):
                
                alert_Video = 1
                use_frame = 1
                DROWSINESS_alert_Video += 1
                print('drowsy alert')
                
                cv2.putText(frame, "*****DROWSY ALERT!*****", (20, 100),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif (indicator_Yawn == 1):
                    
                    alert_Video = 1
                    use_frame = 1
                    YAWN_alert_Video += 1
                    print('fatigue alert')
                    cv2.putText(frame, "*****FATIGUE ALERT!*****", (20, 100),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
            else:
                alert_Video = 0 
                     
            if indicator_center_road_look == 0:
                count_NOT_center_road_look += 1        
                if count_NOT_center_road_look >= frame_check:
                    alert_Video = 1
                    use_frame = 1
                    print('distraction alert')
                    cv2.putText(frame, "***DISTRACTION ALERT!***", (20, 100),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    DISTRACTION_alert_Video +=  1     
            else:
                count_NOT_center_road_look = 0
                indicator_center_road_look = 0  
       
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_Output, fourcc, 20, (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)   
    
    print("[INFO] Your video file is ready... Go to the desired directory!...")
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

aa ='D:/hp/Desktop/project/Input/input1.mp4'

internal_cam_fun(aa)
