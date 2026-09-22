import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"
import cv2
import mediapipe as mp
import sys
from utils.pose_utils import extract_landmarks_px, get_body_features, classify_pose

mp_pose = mp.solutions.pose

if len(sys.argv) < 2:
    print("用法: python debug_angles.py <视频路径>")
    print("例如: python debug_angles.py ./videos/squat_01.mp4")
    sys.exit(1)

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ 打不开 {video_path}")
    sys.exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
print(f"视频: {video_path}")
print(f"分辨率: {w}x{h}, FPS: {fps:.1f}")
print("=" * 100)

pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

records = {"knee": [], "elbow": [], "hip": [], "plank": [],
           "torso": [], "leg": []}

frame_idx = 0
while cap.isOpened():
    try:
        ret, frame = cap.read()
    except cv2.error:
        break
    if not ret or frame is None or frame.size == 0:
        break
    frame_idx += 1
    if frame_idx % 10 != 0:          # 每 10 帧采样一次
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = pose.process(rgb)
    if not r.pose_landmarks:
        continue
    pts = extract_landmarks_px(r.pose_landmarks.landmark, w, h, vis_th=0.4)
    f = get_body_features(pts)
    p = classify_pose(f)

    def fmt(v):
        return f"{v:6.1f}" if isinstance(v, (int, float)) else "  --  "

    print(f"[{frame_idx:5d}] pose={p:7s} "
          f"knee={fmt(f.get('knee_angle'))} "
          f"elbow={fmt(f.get('elbow_angle'))} "
          f"hip={fmt(f.get('hip_angle'))} "
          f"plank={fmt(f.get('plank_angle'))} "
          f"torso={fmt(f.get('torso_vert_angle'))} "
          f"leg={fmt(f.get('leg_vert_angle'))}")

    for k in records:
        v = f.get(k + "_angle" if k != "torso" and k != "leg"
                  else k + "_vert_angle")
        if isinstance(v, (int, float)):
            records[k].append(v)

cap.release()

print("\n" + "=" * 100)
print("统计结果（每列代表该动作的关键指标范围）")
print("=" * 100)
for k, vs in records.items():
    if vs:
        print(f"{k:6s}:  min={min(vs):6.1f}  max={max(vs):6.1f}  "
              f"mean={sum(vs)/len(vs):6.1f}  样本数={len(vs)}")
    else:
        print(f"{k:6s}:  无有效数据（关键点未被检出）")
        