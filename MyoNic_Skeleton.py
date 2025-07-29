import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

squat_count = 0
squat_state = "up"

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

camera_id = 2  # เปลี่ยนเป็น 1, 2 หรืออื่นถ้ากล้องหลักไม่ทำงาน
cap = cv2.VideoCapture(camera_id)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    overlay = frame.copy()

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        angle = calculate_angle(hip, knee, ankle)

        knee_point = tuple(np.multiply(knee, [frame.shape[1], frame.shape[0]]).astype(int))
        cv2.circle(overlay, knee_point, 18, (0, 255, 255), -1)
        cv2.putText(overlay, str(int(angle)), (knee_point[0]-15, knee_point[1]+10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (50, 50, 50), 2, cv2.LINE_AA)

        if angle < 90 and squat_state == "up":
            squat_state = "down"
        if angle > 160 and squat_state == "down":
            squat_state = "up"
            squat_count += 1

    mp_drawing.draw_landmarks(overlay, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

    cv2.rectangle(overlay, (20, 20), (260, 110), (40, 40, 40), -1)
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, 'SQUATS', (35, 55),
                cv2.FONT_HERSHEY_PLAIN, 2.5, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(frame, str(squat_count), (40, 100),
                cv2.FONT_HERSHEY_PLAIN, 3.5, (0,255,255), 4, cv2.LINE_AA)

    cv2.imshow('Squat Counter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
