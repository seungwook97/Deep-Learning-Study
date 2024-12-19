import mediapipe as mp
import cv2
import numpy as np 

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
push_count = 0
flag = 0  

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

with mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) as pose:

    while True:
        _, frame = cap.read()
        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)  # 좌우 반전
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)

        try: 
            # Get Landmarks_coordinates
            shoulder_left = [results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y]
            elbow_left = [results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y]
            wrist_left = [results.pose_landmarks.landmark[15].x, results.pose_landmarks.landmark[15].y]
            shoulder_right = [results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y]
            elbow_right = [results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[14].y]
            wrist_right = [results.pose_landmarks.landmark[16].x, results.pose_landmarks.landmark[16].y]
    
            # Calculate Angle
            angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
            angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)

            # Draw Circle and Line
            nose_x =results.pose_landmarks.landmark[0].x
            nose_y =results.pose_landmarks.landmark[0].y
            shoulder_x =results.pose_landmarks.landmark[11].x
            shoulder_y =results.pose_landmarks.landmark[11].y
            shoulder2_x =results.pose_landmarks.landmark[12].x
            shoulder2_y =results.pose_landmarks.landmark[12].y
            elbow_x =results.pose_landmarks.landmark[13].x
            elbow_y =results.pose_landmarks.landmark[13].y
            wrist_x =results.pose_landmarks.landmark[15].x
            wrist_y =results.pose_landmarks.landmark[15].y
            elbow2_x =results.pose_landmarks.landmark[14].x
            elbow2_y =results.pose_landmarks.landmark[14].y
            wrist2_x =results.pose_landmarks.landmark[16].x
            wrist2_y =results.pose_landmarks.landmark[16].y

            cv2.circle(frame, (int(nose_x*w), int(nose_y*h)), 90, (255, 255, 255), -1)  
            cv2.circle(frame, (int(shoulder_x*w), int(shoulder_y*h)), 10, (0, 255, 0), -1)
            cv2.circle(frame, (int(shoulder2_x*w), int(shoulder2_y*h)), 10, (0, 255, 0), -1)
            cv2.circle(frame, (int(elbow_x*w), int(elbow_y*h)), 10, (255, 0, 0), -1)
            cv2.circle(frame, (int(elbow2_x*w), int(elbow2_y*h)), 10, (255, 0, 0), -1)
            cv2.circle(frame, (int(wrist_x*w), int(wrist_y*h)), 10, (0, 0, 0), -1)
            cv2.circle(frame, (int(wrist2_x*w), int(wrist2_y*h)), 10, (0, 0, 0), -1)

            cv2.line(frame, (int(shoulder_x*w), int(shoulder_y*h)), (int(elbow_x*w), int(elbow_y*h)), (0, 0, 0), 5)
            cv2.line(frame, (int(elbow_x*w), int(elbow_y*h)), (int(wrist_x*w), int(wrist_y*h)), (0, 0, 0), 5)
            cv2.line(frame, (int(shoulder2_x*w), int(shoulder2_y*h)), (int(elbow2_x*w), int(elbow2_y*h)), (0, 0, 0), 5)
            cv2.line(frame, (int(elbow2_x*w), int(elbow2_y*h)), (int(wrist2_x*w), int(wrist2_y*h)), (0, 0, 0), 5)

            # Push up count
            if angle_left < 60 and flag == 0:  
                push_count += 1
                flag = 1

            elif angle_left> 140 and flag == 1:  
                flag = 0

            cv2.putText(frame, f'KG_Push Up!: {push_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        except:
            pass

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

        cv2.imshow("Motion Tracking", frame)

cap.release()
cv2.destroyAllWindows()
