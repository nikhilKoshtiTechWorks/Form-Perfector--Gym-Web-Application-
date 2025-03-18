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


# For all combined exercises
def exercise_processing(arr):
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
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
                right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]

                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                # Calculate angle
                left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
                right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_index)
                right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_index)

                left_hip_angle = calculate_angle(left_knee, left_hip, left_shoulder)
                right_hip_angle = calculate_angle(right_knee, right_hip, right_shoulder)

                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot_index)
                right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot_index)
                
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

                # for left_shoulder_angle
                if arr[0][2] < left_shoulder_angle < arr[0][3]:    
                    lsx = int(left_shoulder[0] * 640)
                    lsy = int(left_shoulder[1] * 480)

                    cv2.circle(image, (lsx,lsy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    lsx = int(left_shoulder[0] * 640)
                    lsy = int(left_shoulder[1] * 480)

                    cv2.circle(image, (lsx,lsy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)
                
                # for right_shoulder_angle
                if arr[0][4] < right_shoulder_angle < arr[0][5]:
                    rsx = int(right_shoulder[0] * 640)
                    rsy = int(right_shoulder[1] * 480)

                    cv2.circle(image, (rsx,rsy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    rsx = int(right_shoulder[0] * 640)
                    rsy = int(right_shoulder[1] * 480)

                    cv2.circle(image, (rsx,rsy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # for left_elbow_angle
                if arr[0][6] < left_elbow_angle < arr[0][7]:     
                    lex = int(left_elbow[0] * 640)
                    ley = int(left_elbow[1] * 480)

                    cv2.circle(image, (lex,ley), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    lex = int(left_elbow[0] * 640)
                    ley = int(left_elbow[1] * 480)

                    cv2.circle(image, (lex,ley), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # for right_elbow_angle
                if arr[0][8] < right_elbow_angle < arr[0][9]:    
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)    

                # for left_wrist_angle
                if arr[0][10] < left_wrist_angle < arr[0][11]:     
                    lwx = int(left_wrist[0] * 640)
                    lwy = int(left_wrist[1] * 480)


                    cv2.circle(image, (lwx,lwy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    lwx = int(left_wrist[0] * 640)
                    lwy = int(left_wrist[1] * 480)

                    cv2.circle(image, (lwx,lwy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # for right_wrist_angle
                if arr[0][12] < right_wrist_angle < arr[0][13]:
                    rwx = int(right_wrist[0] * 640)
                    rwy = int(right_wrist[1] * 480)

                    cv2.circle(image, (rwx,rwy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    rwx = int(right_wrist[0] * 640)
                    rwy = int(right_wrist[1] * 480)

                    cv2.circle(image, (rwx,rwy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)    

                # for left_hip_angle
                if arr[0][14] < left_hip_angle < arr[0][15]:     
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # for right_hip_angle
                if arr[0][16] < right_hip_angle < arr[0][17]:
                    # Putting box and text for reference only
                    # cv2.rectangle(image, (0,40), (270,80), (0,255,0), -1)
                    # cv2.putText(image, 'Jip is Right', (10, 70),
                    #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)     # Draw landmarks with default specifications      

                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (rhx,rhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (rhx,rhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)    

                # for left_knee_angle
                if arr[0][18] < left_knee_angle < arr[0][19]:    
                    lkx = int(left_knee[0] * 640)
                    lky = int(left_knee[1] * 480)

                    cv2.circle(image, (lkx,lky), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    lkx = int(left_knee[0] * 640)
                    lky = int(left_knee[1] * 480)

                    cv2.circle(image, (lkx,lky), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)  

                # for right_knee_angle
                if arr[0][20] < right_knee_angle < arr[0][21]:
                    rkx = int(right_knee[0] * 640)
                    rky = int(right_knee[1] * 480)

                    cv2.circle(image, (rkx,rky), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    rkx = int(right_knee[0] * 640)
                    rky = int(right_knee[1] * 480)

                    cv2.circle(image, (rkx,rky), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)    

                # for left_ankle_angle
                if arr[0][22] < left_ankle_angle < arr[0][23]:     
                    lax = int(left_ankle[0] * 640)
                    lay = int(left_ankle[1] * 480)

                    cv2.circle(image, (lax,lay), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    lax = int(left_ankle[0] * 640)
                    lay = int(left_ankle[1] * 480)

                    cv2.circle(image, (lax,lay), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # for right_ankle_angle
                if arr[0][24] < right_ankle_angle < arr[0][25]:
                    rax = int(right_ankle[0] * 640)
                    ray = int(right_ankle[1] * 480)

                    cv2.circle(image, (rax,ray), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    rax = int(right_ankle[0] * 640)
                    ray = int(right_ankle[1] * 480)

                    cv2.circle(image, (rax,ray), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)    
                
            except:
                pass
            
            ret, jpeg = cv2.imencode('.jpg', image)
            data = []
            data.append(jpeg.tobytes())

            frames=data[0]
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n\r\n')
            
        cap.release()


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

def lat_pull_down_processing():
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

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                
                # cv2.circle(image,(ex,ey),5,(0,0,255),-1)

                # Calculate angle
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

                # right elbow
                if 90 < right_elbow_angle < 160:
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)
                    playsound()
                    cv2.circle(image, (rex,rey), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # leff elbow
                if 90 < left_elbow_angle < 160:
                    lex = int(left_elbow[0] * 640)
                    ley = int(left_elbow[1] * 480)
        
                    cv2.circle(image, (lex,ley), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:          
                    lex = int(left_elbow[0] * 640)
                    ley = int(left_elbow[1] * 480)
                    playsound()
                    cv2.circle(image, (lex,ley), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

            except:
                pass
            
            ret, jpeg = cv2.imencode('.jpg', image)
            data = []
            data.append(jpeg.tobytes())

            frames=data[0]
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n\r\n')
            
        cap.release()

def overhead_press_processing():
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

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                
                # cv2.circle(image,(ex,ey),5,(0,0,255),-1)

                # Calculate angle
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

                # right elbow
                if 90 < right_elbow_angle < 160:
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # leff elbow
                if 90 < left_elbow_angle < 160:
                    lex = int(left_elbow[0] * 640)
                    ley = int(left_elbow[1] * 480)
                    playsound()
                    cv2.circle(image, (lex,ley), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:          
                    lex = int(left_elbow[0] * 640)
                    ley = int(left_elbow[1] * 480)
                    playsound()
                    cv2.circle(image, (lex,ley), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

            except:
                pass
            
            ret, jpeg = cv2.imencode('.jpg', image)
            data = []
            data.append(jpeg.tobytes())

            frames=data[0]
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n\r\n')
            
        cap.release()

def skullcrusher_processing():
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

def leg_press_processing():

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
                if 80 < right_hip_angle < 100:
                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (rhx,rhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (rhx,rhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # left hip
                if 80 < left_hip_angle < 100:
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1           
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # right knee
                if 80 < right_knee_angle < 170:
                    rkx = int(right_knee[0] * 640)
                    rky = int(right_knee[1] * 480)

                    cv2.circle(image, (rkx,rky), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rkx = int(right_knee[0] * 640)
                    rky = int(right_knee[1] * 480)

                    cv2.circle(image, (rkx,rky), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # leff knee
                if 80 < left_knee_angle < 170:
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

def squats_processing():
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
                if 60 < right_hip_angle < 175:
                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (rhx,rhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (rhx,rhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # left hip
                if 60 < left_hip_angle < 175:
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1           
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # right knee
                if 60 < right_knee_angle < 175:
                    rkx = int(right_knee[0] * 640)
                    rky = int(right_knee[1] * 480)

                    cv2.circle(image, (rkx,rky), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rkx = int(right_knee[0] * 640)
                    rky = int(right_knee[1] * 480)

                    cv2.circle(image, (rkx,rky), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # leff knee
                if 60 < left_knee_angle < 175:
                    lkx = int(left_knee[0] * 640)
                    lky = int(left_knee[1] * 480)

                    cv2.circle(image, (lkx,lky), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1               
                    lkx = int(left_knee[0] * 640)
                    lky = int(left_knee[1] * 480)

                    cv2.circle(image, (lkx,lky), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # right elbow
                if 70 < right_elbow_angle < 100:
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # leff elbow
                if 70 < left_elbow_angle < 100:
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

def deadlift_processing():
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
                if 50 < right_hip_angle < 175:
                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (rhx,rhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (rhx,rhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # left hip
                if 50 < left_hip_angle < 175:
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1           
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # right knee
                if 50 < right_knee_angle < 175:
                    rkx = int(right_knee[0] * 640)
                    rky = int(right_knee[1] * 480)

                    cv2.circle(image, (rkx,rky), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rkx = int(right_knee[0] * 640)
                    rky = int(right_knee[1] * 480)

                    cv2.circle(image, (rkx,rky), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # leff knee
                if 50 < left_knee_angle < 175:
                    lkx = int(left_knee[0] * 640)
                    lky = int(left_knee[1] * 480)

                    cv2.circle(image, (lkx,lky), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1               
                    lkx = int(left_knee[0] * 640)
                    lky = int(left_knee[1] * 480)

                    cv2.circle(image, (lkx,lky), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # right elbow
                if 160 < right_elbow_angle < 180:
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # leff elbow
                if 160 < left_elbow_angle < 180:
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

def seated_rows_processing():
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
                if 70 < right_hip_angle < 100:
                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (rhx,rhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rhx = int(right_hip[0] * 640)
                    rhy = int(right_hip[1] * 480)

                    cv2.circle(image, (rhx,rhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # left hip
                if 70 < left_hip_angle < 100:
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1           
                    lhx = int(left_hip[0] * 640)
                    lhy = int(left_hip[1] * 480)

                    cv2.circle(image, (lhx,lhy), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # right knee
                if 150 < right_knee_angle < 170:
                    rkx = int(right_knee[0] * 640)
                    rky = int(right_knee[1] * 480)

                    cv2.circle(image, (rkx,rky), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rkx = int(right_knee[0] * 640)
                    rky = int(right_knee[1] * 480)

                    cv2.circle(image, (rkx,rky), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # leff knee
                if 150 < left_knee_angle < 170:
                    lkx = int(left_knee[0] * 640)
                    lky = int(left_knee[1] * 480)

                    cv2.circle(image, (lkx,lky), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1               
                    lkx = int(left_knee[0] * 640)
                    lky = int(left_knee[1] * 480)

                    cv2.circle(image, (lkx,lky), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # right elbow
                if 30 < right_elbow_angle < 175:
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,255,0), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                else:
                    counter = counter + 1
                    rex = int(right_elbow[0] * 640)
                    rey = int(right_elbow[1] * 480)

                    cv2.circle(image, (rex,rey), 10, (0,0,255), 5)      # cv2.circle(image, center_coordinates, radius, color, thickness)

                # leff elbow
                if 30 < left_elbow_angle < 175:
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