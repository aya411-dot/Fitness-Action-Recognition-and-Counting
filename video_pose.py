import cv2
import mediapipe as mp

# 初始化mediapipe姿态识别
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def camera_pose_detect():
    cap = cv2.VideoCapture(0)
    win_name = "摄像头姿态识别（带人体骨架线）"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1000, 700)

    # 姿态检测模型
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 颜色空间转换
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_img.flags.writeable = False
        results = pose.process(rgb_img)

        rgb_img.flags.writeable = True
        draw_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # 绘制全身骨架线条（关键代码，没有这段就看不到线条）
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                draw_img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        
        cv2.imshow(win_name, draw_img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_pose_detect()