# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:50:35 2020

@author: Admin
"""


# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import pandas as pd
import datetime
import sys
#import pytesseract


print('start')
#Used for capture timestamp
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#Calculate Eye aspect ratio for drowsiness
def eye_aspect_ratio(eye):
    
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
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

    #Threshold value for Drowsiness
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold for to set off the
    # alarm
    
    thresh_Drowsiness = 0.25
    frame_check = 20
    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
    #COUNTER = 0
    #ALARM_ON = False
    
    #Threshhold value for Yawn
    thresh_Yawn = 22
    frame_Yawn = 5
    
    #For Drowsiness and Yawn
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detect = dlib.get_frontal_face_detector()
   # predict = dlib.shape_predictor("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    predict = dlib.shape_predictor("D:/hp/Desktop/project/shape_predictor_68_face_landmarks.dat")
    
    #Find left eye and right eye
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    #Input and output video
    video_Input = argument1
    video_Output = "D:/hp/Desktop/project/Output/Output.mp4"
    video_Output_Alert = "D:/hp/Desktop/project/Output/Output_Alert.mp4"
    
    # let's Load our video-file from disk
    cap = cv2.VideoCapture(video_Input)
    writer = None
    writer_only_alert = None
    
    #Find number of frames from video
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] {} :Total frames in Video".format(total))
    
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1
     
    #Local variable
    print('test1')
    count_Drowsiness=0
    count_Yawn = 0
    indicator_Drowsiness = 0
    indicator_Yawn = 0
    alert_Video = 0
    indicator_center_road_look = 0 # 0 mean not looking at center of road
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
    
    # looping over the frames from Video
    
    print('test2')
    i=0
    while cap.isOpened():
        i = i+1
        print('while loop=',i)
        ret, frame=cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=450)
        use_frame= 0 # new frame generated but not yet used for alert
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
    
        for subject in subjects:
            
            indicator_center_road_look = 1 #looking at center of road
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)#converting to NumPy Array
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            #leftEyeHull = cv2.convexHull(leftEye)
            #rightEyeHull = cv2.convexHull(rightEye)
            #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    
            distance = lip_distance(shape)
            #lip = shape[48:60]
            #cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
    
            #Check for Drowsiness
            if ear < thresh_Drowsiness:                   
                count_Drowsiness += 1
                if count_Drowsiness >= frame_check:
                    indicator_Drowsiness = 1 
            else:
                count_Drowsiness = 0
                indicator_Drowsiness = 0
    
            #Check for Yawn
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
                
                frame1 = frame[770:800,110:355] 
                time_stamp = tess.image_to_string(frame1)
                #time_stamp = pytesseract.image_to_string(frame1)
                if DROWSINESS_alert_Video == 1:
                    start_time_stamp = time_stamp
                    end_time_stamp = time_stamp 
                    Alert_Event = 'Drowsy'
                    
                    writer_sleep_alert = None
                    #Remove space and : from string
                    video_name = time_stamp.replace(" ","-" ).replace(":","-" ) 
                    video_Output_Alert_Sleep = video_name+"_Drowsy_Alert.mp4"
                else:
                    end_time_stamp = time_stamp
                
                if writer_sleep_alert is None:
                    fourcc1 = cv2.VideoWriter_fourcc(*"mp4v")
                    writer_sleep_alert = cv2.VideoWriter(video_Output_Alert_Sleep, fourcc1, 20, (frame.shape[1], frame.shape[0]), True)
                writer_sleep_alert.write(frame) 
    
            
            elif (indicator_Yawn == 1):
                
                alert_Video = 1
                use_frame = 1
                YAWN_alert_Video += 1
                print('fatigue alert')
                cv2.putText(frame, "*****FATIGUE ALERT!*****", (20, 100),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                frame1 = frame[770:800,110:355]
                time_stamp = tess.image_to_string(frame1)
                #time_stamp = pytesseract.image_to_string(frame1)
                if YAWN_alert_Video == 1:
                    start_time_stamp = time_stamp
                    end_time_stamp = time_stamp 
                    Alert_Event = 'Fatigue'
                    
                    writer_yawn_alert = None
                    video_name = time_stamp.replace(" ","-" ).replace(":","-" )
                    video_Output_Alert_Yawn = video_name +"_Fatigue_Alert.mp4"                
                else:
                    end_time_stamp = time_stamp   
                    
                if writer_yawn_alert is None:
                    fourcc1 = cv2.VideoWriter_fourcc(*"mp4v")
                    writer_yawn_alert = cv2.VideoWriter(video_Output_Alert_Yawn, fourcc1, 20, (frame.shape[1], frame.shape[0]), True)
                writer_yawn_alert.write(frame)                 
            else:
                alert_Video = 0 
             
       
        #Check for NOT looking at center of road
        if indicator_center_road_look == 0:
            count_NOT_center_road_look += 1        
            if count_NOT_center_road_look >= frame_check:
                alert_Video = 1
                use_frame = 1
                print('distraction alert')
                cv2.putText(frame, "***DISTRACTION ALERT!***", (20, 100),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                DISTRACTION_alert_Video += 1
                
                frame1 = frame[770:800,110:355] 
                time_stamp = tess.image_to_string(frame1) 
                #time_stamp = pytesseract.image_to_string(frame1)
    
                if DISTRACTION_alert_Video == 1:
                    start_time_stamp = time_stamp
                    end_time_stamp = time_stamp 
                    Alert_Event = 'Distraction'
                    
                    writer_distract_alert = None
                    video_name = time_stamp.replace(" ","-" ).replace(":","-" )                
                    video_Output_Alert_Distract = video_name +"_Distract_Alert.mp4"                 
                else:
                    end_time_stamp = time_stamp
                    
                if writer_distract_alert is None:
                    fourcc1 = cv2.VideoWriter_fourcc(*"mp4v")
                    writer_distract_alert = cv2.VideoWriter(video_Output_Alert_Distract, fourcc1, 20, (frame.shape[1], frame.shape[0]), True)
                writer_distract_alert.write(frame) 
                
            else:
                alert_Video = 0
    
        else:
            count_NOT_center_road_look = 0
            indicator_center_road_look = 0
    
        if ((use_frame == 0) and (DROWSINESS_alert_Video !=0)):
            start_time_stamp = pd.to_datetime(start_time_stamp).strftime('%Y-%B-%d %H:%M:%S')
            end_time_stamp = pd.to_datetime(end_time_stamp).strftime('%Y-%B-%d %H:%M:%S')
            duration = (datetime.datetime.strptime(end_time_stamp, datetimeFormat)\
                     - datetime.datetime.strptime(start_time_stamp, datetimeFormat)).seconds    
               
            df.loc[idx] =  [Alert_Event, start_time_stamp,end_time_stamp,duration,video_Output_Alert_Sleep]
            idx += 1
            DROWSINESS_alert_Video = 0
            writer_sleep_alert.release()
           
            
        if ((use_frame == 0) and (YAWN_alert_Video !=0)):
            
            start_time_stamp = pd.to_datetime(start_time_stamp).strftime('%Y-%B-%d %H:%M:%S')
            end_time_stamp = pd.to_datetime(end_time_stamp).strftime('%Y-%B-%d %H:%M:%S')
            duration = (datetime.datetime.strptime(end_time_stamp, datetimeFormat)\
                     - datetime.datetime.strptime(start_time_stamp, datetimeFormat)).seconds    
    
            df.loc[idx] =  [Alert_Event, start_time_stamp,end_time_stamp,duration,video_Output_Alert_Yawn]
            idx += 1
            YAWN_alert_Video = 0  
            writer_yawn_alert.release()        
    
        if ((use_frame == 0) and (DISTRACTION_alert_Video !=0)):
            
            start_time_stamp = pd.to_datetime(start_time_stamp).strftime('%Y-%B-%d %H:%M:%S')
            end_time_stamp = pd.to_datetime(end_time_stamp).strftime('%Y-%B-%d %H:%M:%S')
            duration = (datetime.datetime.strptime(end_time_stamp, datetimeFormat)\
                     - datetime.datetime.strptime(start_time_stamp, datetimeFormat)).seconds    
    
            df.loc[idx] =  [Alert_Event, start_time_stamp,end_time_stamp,duration,video_Output_Alert_Distract]
            idx += 1
            DISTRACTION_alert_Video = 0  
            writer_distract_alert.release()        
           
        # write the video frames to our disk...
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            #fourcc = cv2.VideoWriter_fourcc(*"DIVX")
            writer = cv2.VideoWriter(video_Output, fourcc, 20, (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)   
    
        #Writing only Alert message in different video 
        if alert_Video == 1:
            if writer_only_alert is None:
                fourcc1 = cv2.VideoWriter_fourcc(*"mp4v")
                writer_only_alert = cv2.VideoWriter(video_Output_Alert, fourcc1, 20, (frame.shape[1], frame.shape[0]), True)
            writer_only_alert.write(frame) 
    
    df.to_csv('D:/hp/Desktop/project/Output/Summary_Alert_Message.csv')
    print("[INFO] Your video file and summary csv file are ready ... Go to the desired directory!...")
    
    
    # let's cleanup
    cap.release()
    writer.release()
    writer_only_alert.release()
    cv2.destroyAllWindows()
    
#internal_cam_fun(sys.argv[1])
aa ='D:/hp/Desktop/project/Input/Input1.mp4'

internal_cam_fun(aa)
