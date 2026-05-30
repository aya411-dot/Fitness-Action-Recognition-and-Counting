import cv2
import mediapipe as mp

# 标准兼容导入写法
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_draw = mp.solutions.drawing_utils

# 读取图片
img = cv2.imread("test_img/test.jpg")
if img is None:
    print("找不到图片，请检查test_img/test.jpg")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 窗口可缩放，完整显示图片
    cv2.namedWindow("人体姿态识别", cv2.WINDOW_NORMAL)
    cv2.imshow("人体姿态识别", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()