import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# 初始化MediaPipe人体识别
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_draw = mp.solutions.drawing_utils

# 读取测试图片
img = cv2.imread("test_img/test.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 识别人体关键点
result = pose.process(img_rgb)

# 绘制骨骼关键点
if result.pose_landmarks:
    mp_draw.draw_landmarks(img_rgb, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# 展示识别效果图
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()