import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 修改路径，读取images文件夹内图片
img = cv2.imread("./images/test.jpg")
if img is None:
    print("找不到图片，请检查 ./images/test.jpg 是否存在！")
else:
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_img)
    if result.pose_landmarks:
        mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # 输出结果图片保存到images文件夹
    cv2.imwrite("./images/result.jpg", img)
    cv2.imshow("图片人体关键点识别结果", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
