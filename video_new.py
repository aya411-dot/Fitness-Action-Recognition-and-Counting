import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"
import cv2
import mediapipe as mp
import csv
from collections import deque

from utils.pose_utils import (
    extract_landmarks_px,
    get_body_features,
    classify_pose,
    is_prone_orientation,
    RepCounter,
    PlankTimer,
)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


# ==================== 阈值参数 ====================
# 深蹲：膝角 < DOWN 蹲下，> UP 站起
SQUAT_DOWN_TH = 105
SQUAT_UP_TH = 150

# 俯卧撑：肘角 < DOWN 下压，> UP 撑起
PUSHUP_DOWN_TH = 115
PUSHUP_UP_TH = 150

# 平板支撑判据
PLANK_HIP_MIN = 130          # 肩-髋-膝至少 130（身体不折叠）
PLANK_STABILITY_WIN = 25     # 滑动窗口帧数
PLANK_STABILITY_TH = 25      # 窗口内 elbow 极差 < 25° 认为稳定
PLANK_MIN_FRAMES = 10        # 稳定持续 10 帧才确认平板

# 状态机连续帧确认
MIN_FRAMES = 4
# ===================================================


def process_one_video(video_path, csv_path):
    basename = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开 {basename}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join("output", f"out_{basename}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # ---------- 状态机 ----------
    squat_counter = RepCounter(SQUAT_DOWN_TH, SQUAT_UP_TH, MIN_FRAMES)
    pushup_counter = RepCounter(PUSHUP_DOWN_TH, PUSHUP_UP_TH, MIN_FRAMES)

    plank_timer = PlankTimer(min_frames=MIN_FRAMES)

    plank_consecutive = 0
    elbow_history = deque(maxlen=PLANK_STABILITY_WIN)

    last_lying_type = "unknown"

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    frame_idx = 0
    debug = True

    while cap.isOpened():
        try:
            ret, frame = cap.read()
        except cv2.error as e:
            print(f"⚠️ {basename} 解码异常，提前结束: {e}")
            break
        if not ret or frame is None or frame.size == 0:
            break
        frame_idx += 1

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        result = pose.process(img_rgb)
        img_rgb.flags.writeable = True

        pose_label = "unknown"
        action_label = "none"

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            pts = extract_landmarks_px(lm, width, height, vis_th=0.25)
            feats = get_body_features(pts)

            pose_label = classify_pose(feats)

            if pose_label == "lying":
                lying_type = is_prone_orientation(feats)
                if lying_type != "unknown":
                    last_lying_type = lying_type

            # ========== 站立：深蹲 ==========
            if pose_label == "stand" and feats.get('knee_angle') is not None:
                squat_counter.update(feats['knee_angle'])
                if feats['knee_angle'] < SQUAT_UP_TH + 10:
                    action_label = "Squat"

            # ========== 俯卧：俯卧撑 或 平板支撑 ==========
            elif pose_label == "lying" and last_lying_type == "prone":
                elbow = feats.get('elbow_angle')
                hip = feats.get('hip_angle')

                if elbow is not None:
                    elbow_history.append(elbow)

                if len(elbow_history) >= PLANK_STABILITY_WIN // 2:
                    elbow_range = max(elbow_history) - min(elbow_history)
                    is_stable = elbow_range < PLANK_STABILITY_TH
                else:
                    is_stable = False

                plank_pose_ok = (
                    is_stable
                    and hip is not None and hip > PLANK_HIP_MIN
                )

                if plank_pose_ok:
                    plank_consecutive += 1
                else:
                    plank_consecutive = 0

                if plank_consecutive >= PLANK_MIN_FRAMES:
                    plank_timer.update(True, frame_idx)
                    action_label = "Plank"
                else:
                    plank_timer.update(False, frame_idx)
                    if elbow is not None:
                        pushup_counter.update(elbow)
                        if elbow < PUSHUP_UP_TH + 5:
                            action_label = "Pushup"

            # 绘制骨架
            mp_draw.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 180, 255), thickness=2),
            )

            if debug:
                def fmt(v):
                    return f"{v:.0f}" if isinstance(v, (int, float)) else "--"
                cv2.putText(frame, f"knee:{fmt(feats.get('knee_angle'))} "
                                   f"elbow:{fmt(feats.get('elbow_angle'))} "
                                   f"hip:{fmt(feats.get('hip_angle'))} "
                                   f"plank:{fmt(feats.get('plank_angle'))}",
                            (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (200, 200, 200), 1)
                cv2.putText(frame, f"torso:{fmt(feats.get('torso_vert_angle'))} "
                                   f"leg:{fmt(feats.get('leg_vert_angle'))}",
                            (20, height - 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (200, 200, 200), 1)
                if len(elbow_history) >= PLANK_STABILITY_WIN // 2:
                    elbow_range = max(elbow_history) - min(elbow_history)
                else:
                    elbow_range = -1
                cv2.putText(frame, f"elbowRange:{elbow_range:.0f} "
                                   f"plankCnt:{plank_consecutive}",
                            (20, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (200, 200, 200), 1)

        # ---------- 画面文字 ----------
        squat_count = squat_counter.count
        pushup_count = pushup_counter.count
        plank_time = plank_timer.get_time(fps, mode="max")

        cv2.putText(frame, f"Pose:{pose_label} Action:{action_label}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Squat:{squat_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Pushup:{pushup_count}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Plank:{plank_time:.1f}s", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        out.write(frame)

    # ---------- 收尾 ----------
    plank_timer.finalize(frame_idx)
    plank_time = plank_timer.get_time(fps, mode="max")

    cap.release()
    out.release()

    squat_count = squat_counter.count
    pushup_count = pushup_counter.count

    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([basename, squat_count, pushup_count,
                         round(plank_time, 1)])

    print(f"✅ {basename} | Squat:{squat_count} Pushup:{pushup_count} "
          f"PlankTime:{plank_time:.1f}s")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    csv_path = os.path.join("output", "result.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "Squat", "Pushup", "PlankTime"])

    video_dir = "./videos/"
    for filename in os.listdir(video_dir):
        if filename.lower().endswith(".mp4"):
            full_path = os.path.join(video_dir, filename)
            process_one_video(full_path, csv_path)