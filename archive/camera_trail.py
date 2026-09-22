import cv2
import mediapipe as mp
import math

# 修正：mp.solutions.pose是模块，实例化需要 .Pose()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

squat_count = 0
move_stage = "up"

def calculate_angle(point_a, point_b, point_c):
    x1, y1 = point_a
    x2, y2 = point_b
    x3, y3 = point_c
    radian_1 = math.atan2(y1 - y2, x1 - x2)
    radian_2 = math.atan2(y3 - y2, x3 - x2)
    angle = abs((radian_2 - radian_1) * 180 / math.pi)
    return 360 - angle if angle > 180 else angle

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("摄像头打开失败，请关闭微信/相机等占用软件！")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("squat_video.mp4", fourcc, 20, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("画面读取中断")
        break
    height, width = frame.shape[:2]
    # 完整标准写法，不会报 cv2.BGR2RGB 错误
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        hip_point = (int(landmarks[23].x * width), int(landmarks[23].y * height))
        knee_point = (int(landmarks[25].x * width), int(landmarks[25].y * height))
        ankle_point = (int(landmarks[27].x * width), int(landmarks[27].y * height))
        leg_angle = calculate_angle(hip_point, knee_point, ankle_point)

        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(0, 210, 0), thickness=2)
        )

        if leg_angle < 90 and move_stage == "up":
            move_stage = "down"
        if leg_angle > 160 and move_stage == "down":
            move_stage = "up"
            squat_count = squat_count + 1

        cv2.rectangle(frame, (0, 0), (240, 90), (30, 30, 30), -1)
        cv2.putText(frame, f"Squat Total: {squat_count}", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
        cv2.putText(frame, f"Posture: {move_stage}", (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    out.write(frame)
    cv2.imshow("深蹲录像 | 必须按 q 保存退出", frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        cv2.imwrite("squat_capture.jpg", frame)
        print("录像、截图保存完成！")
        break

cap.release()
out.release()
cv2.destroyAllWindows()