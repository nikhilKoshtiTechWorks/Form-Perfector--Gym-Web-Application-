import cv2
import mediapipe as mp
import numpy as np
from pygame import mixer
import time

mixer.init()                                #Initialzing pyamge mixer
mixer.music.load('buzzer.mp3')              #Loading Music File

mp_drawing = mp.solutions.drawing_utils    
mp_pose = mp.solutions.pose

def playsound():
    mixer.music.play() #Playing Music with Pygame
    time.sleep(0.2)
    mixer.music.stop() 


def calculate_angle(a,b,c):
    a = np.array(a)   # first i.e. shoulder
    b = np.array(b)   # first i.e. elbow
    c = np.array(c)   # first i.e. wrist
    
    radian = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])  # calculating angke
    angle = np.abs(radian * 180.0 / np.pi)                                       # returning absolute degree value
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def biceps_processing():
    cap = cv2.VideoCapture(0)

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Recoloring image again
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                # Calculate angle
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_hip_angle = calculate_angle(left_knee, left_hip, left_shoulder)
                right_hip_angle = calculate_angle(right_knee, right_hip, right_shoulder)
                
                # Visualizing the angle for reference only
                # cv2.putText(
                #     image, str(left_elbow_angle),
                #     tuple(np.multiply(left_elbow, [640, 480]).astype(int)),            # actual coordinates on display window; 640x480 are the dimensions of video display window
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA
                # )
                                
                # Rendering (showing image with landmarks and connections)
                mp_drawing.draw_landmarks(
                    image,                       # = processed image
                    results.pose_landmarks,      # = gives the co-ordinated of points that are on the joints
                    mp_pose.POSE_CONNECTIONS,    # = gives the combination of landmark points between which we are creating the connections(line)
                    mp_drawing.DrawingSpec(color=(255,0,66), thickness=2, circle_radius=2),     # specifications of our drawing
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                )
                
                # Logic for threshold value

                # for hip
                if 160 < left_hip_angle and right_hip_angle < 175:
                    # Putting box and text for reference only
                    # cv2.rectangle(image, (0,40), (270,80), (0,255,0), -1)
                    # cv2.putText(image, 'Jip is Right', (10, 70),
                    #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)     # Draw landmarks with default specifications      
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)
                    cv2.circle(image, (rhx,rhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)
                    cv2.circle(image, (rhx,rhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)
                
                # for left elbow
                if left_elbow_angle > 30:
                    lex = int(left_elbow[0] * 640)
                    ley = int(left_elbow[1] * 480)

                    cv2.circle(image, (lex,ley), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)


                else:
                    playsound()
                    lex = int(left_elbow[0] * 640)
                    ley = int(left_elbow[1] * 480)

                    cv2.circle(image, (lex,ley), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # for right elbow
                if right_elbow_angle > 30:
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)
                else:
                    playsound()
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)
                    
                    
            except:
                pass
            
            ret, jpeg = cv2.imencode('.jpg', image)
            data = []
            data.append(jpeg.tobytes())

            frames=data[0]
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n\r\n')
            
        cap.release()

def triceps_processing():
    cap = cv2.VideoCapture(0)

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Recoloring image again
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                
                # cv2.circle(image,(ex,ey),5,(0,0,255),-1)

                # Calculate angle
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_hip_angle = calculate_angle(left_knee, left_hip, left_shoulder)
                right_hip_angle = calculate_angle(right_knee, right_hip, right_shoulder)
                
                # Rendering (showing image with landmarks and connections)
                mp_drawing.draw_landmarks(
                    image,                       # = processed image
                    results.pose_landmarks,      # = gives the co-ordinated of points that are on the joints
                    mp_pose.POSE_CONNECTIONS,    # = gives the combination of landmark points between which we are creating the connections(line)
                    mp_drawing.DrawingSpec(color=(255,0,66), thickness=2, circle_radius=2),     # specifications of our drawing
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                )
                
                # Logic for threshold value

                # for hip
                if 160 < left_hip_angle and right_hip_angle < 175:
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)
                    cv2.circle(image, (rhx,rhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)
                    cv2.circle(image, (rhx,rhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)
                
                # for left elbow
                if left_elbow_angle > 85:
                    lex = int(left_elbow[0] * 640)
                    ley = int(left_elbow[1] * 480)

                    cv2.circle(image, (lex,ley), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    playsound()
                    lex = int(left_elbow[0] * 640)
                    ley = int(left_elbow[1] * 480)

                    cv2.circle(image, (lex,ley), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # for right elbow
                if right_elbow_angle > 85:
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    playsound()                  
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)
                    
                    
            except:
                pass
            
            ret, jpeg = cv2.imencode('.jpg', image)
            data = []
            data.append(jpeg.tobytes())

            frames=data[0]
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n\r\n')
            
        cap.release()

def plank_processing():
    cap = cv2.VideoCapture(0)

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        while cap.isOpened():
            counter = 0
            ret, frame = cap.read()
            
            # Recolor image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Recoloring image again
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                
                # cv2.circle(image,(ex,ey),5,(0,0,255),-1)

                # Calculate angle
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_hip_angle = calculate_angle(left_knee, left_hip, left_shoulder)
                right_hip_angle = calculate_angle(right_knee, right_hip, right_shoulder)
                left_elbow_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)
                right_elbow_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)
                
                # Rendering (showing image with landmarks and connections)
                mp_drawing.draw_landmarks(
                    image,                       # = processed image
                    results.pose_landmarks,      # = gives the co-ordinated of points that are on the joints
                    mp_pose.POSE_CONNECTIONS,    # = gives the combination of landmark points between which we are creating the connections(line)
                    mp_drawing.DrawingSpec(color=(255,0,66), thickness=2, circle_radius=2),     # specifications of our drawing
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                )
                
                # Logic for threshold value

                # right hip
                if 140 < right_hip_angle < 160:
                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (rhx,rhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (rhx,rhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # left hip
                if 140 < left_hip_angle < 160:
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1           
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # right knee
                if 155 < right_knee_angle < 175:
                    rkx = int(right_knee[0] * 640)
                    rky = int(right_knee[1] * 480)

                    cv2.circle(image, (rkx,rky), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rkx = int(right_knee[0] * 640)
                    rky = int(right_knee[1] * 480)

                    cv2.circle(image, (rkx,rky), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # leff knee
                if 155 < left_knee_angle < 175:
                    lkx = int(left_knee[0] * 640)
                    lky = int(left_knee[1] * 480)

                    cv2.circle(image, (lkx,lky), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1               
                    lkx = int(left_knee[0] * 640)
                    lky = int(left_knee[1] * 480)

                    cv2.circle(image, (lkx,lky), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # right elbow
                if 70 < right_elbow_angle <90:
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # leff elbow
                if 70 < left_elbow_angle < 90:
                    lex = int(left_elbow[0] * 640)
                    ley = int(left_elbow[1] * 480)

                    cv2.circle(image, (lex,ley), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1           
                    lex = int(left_elbow[0] * 640)
                    ley = int(left_elbow[1] * 480)

                    cv2.circle(image, (lex,ley), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)
                    
                if (counter > 3):
                    playsound()

            except:
                pass
            
            ret, jpeg = cv2.imencode('.jpg', image)
            data = []
            data.append(jpeg.tobytes())

            frames=data[0]
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n\r\n')
            
        cap.release()