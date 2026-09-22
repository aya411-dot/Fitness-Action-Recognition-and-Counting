import cv2
import mediapipe as mp
import math
import os

def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    ba_x = ax - bx
    ba_y = ay - by
    bc_x = cx - bx
    bc_y = cy - by
    dot = ba_x * bc_x + ba_y * bc_y
    mag_ba = math.hypot(ba_x, ba_y)
    mag_bc = math.hypot(bc_x, bc_y)
    if mag_ba == 0 or mag_bc == 0:
        return 0
    cos_ang = dot / (mag_ba * mag_bc)
    cos_ang = max(min(cos_ang, 1), -1)
    angle = math.degrees(math.acos(cos_ang))
    return angle


def process_video(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"无法打开视频：{video_file_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    basename = os.path.basename(video_file_path)
    out_name = f"out_{basename}"
    out = cv2.VideoWriter(out_name, fourcc, fps, (w, h))

    # 计数与状态
    squat_count = 0
    squat_stage = "up"

    pushup_count = 0
    pushup_stage = "up"

    situp_count = 0
    situp_stage = "down"

    plank_time = 0.0
    current_action = "None"

    mp_draw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        res = pose.process(img_rgb)
        img_rgb.flags.writeable = True
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        valid = False
        knee_angle = 0
        elbow_angle = 0
        torso_angle = 0
        sh_avg_y = 0
        hp_avg_y = 0
        foot_avg_y = 0

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            if lm[11].visibility>0.5 and lm[12].visibility>0.5 and lm[23].visibility>0.5 and lm[24].visibility>0.5 and lm[25].visibility>0.5:
                valid = True
            def get_pt(idx):
                return [lm[idx].x * w, lm[idx].y * h]
            shl = get_pt(11)
            shr = get_pt(12)
            hpl = get_pt(23)
            hpr = get_pt(24)
            kl = get_pt(25)
            el = get_pt(13)
            wl = get_pt(15)
            ankle1 = get_pt(27)
            ankle2 = get_pt(28)

            sh_avg_y = (shl[1] + shr[1]) / 2
            hp_avg_y = (hpl[1] + hpr[1]) / 2
            foot_avg_y = (ankle1[1] + ankle2[1]) / 2

            knee_angle = calculate_angle(hpl, kl, ankle1)
            elbow_angle = calculate_angle(shl, el, wl)
            torso_angle = calculate_angle(shl, hpl, kl)

        # 动作场景分类（优先级判断）
        current_action = "None"
        if valid:
            # 场景1：站立 → 深蹲
            is_stand = hp_avg_y > sh_avg_y + 0.1 * h and foot_avg_y > hp_avg_y
            # 场景2：俯卧 → 俯卧撑 / 平板
            is_prone = abs(hp_avg_y - sh_avg_y) < 0.22*h and hp_avg_y > 0.55*h
            # 场景3：平躺 → 仰卧起坐
            is_supine = abs(hp_avg_y - sh_avg_y) < 0.25*h and hp_avg_y > 0.4 * h and foot_avg_y > hp_avg_y

            if is_stand:
                current_action = "Squat"
                # 深蹲状态机
                if knee_angle < 120 and squat_stage == "up":
                    squat_stage = "down"
                if knee_angle > 140 and squat_stage == "down":
                    squat_count += 1
                    squat_stage = "up"

            elif is_prone:
                # 判断是平板还是俯卧撑
                if 150 < torso_angle < 180 and elbow_angle > 140:
                    current_action = "Plank"
                    plank_time += 1.0 / fps
                else:
                    current_action = "Pushup"
                    if elbow_angle < 100 and pushup_stage == "up":
                        pushup_stage = "down"
                    if elbow_angle > 135 and pushup_stage == "down":
                        pushup_count += 1
                        pushup_stage = "up"

            elif is_supine:
                current_action = "Situp"
                if torso_angle > 110 and situp_stage == "down":
                    situp_stage = "up"
                if torso_angle < 80 and situp_stage == "up":
                    situp_count += 1
                    situp_stage = "down"

        # 画面文字展示
        cv2.putText(img, f"Action:{current_action}", (20,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
        cv2.putText(img, f"Squat:{squat_count}", (20,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(img, f"Pushup:{pushup_count}", (20,90), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(img, f"Situp:{situp_count}", (20,120), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(img, f"Plank:{plank_time:.1f}s", (20,150), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        mp_draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        out.write(img)

    cap.release()
    out.release()
    print(f"\n==== {basename} 识别结果 ====")
    print(f"识别动作：{current_action}")
    print(f"深蹲数量: {squat_count}")
    print(f"俯卧撑数量: {pushup_count}")
    print(f"仰卧起坐数量: {situp_count}")
    print(f"平板时长: {plank_time:.1f} s\n")

if __name__ == "__main__":
    for f in os.listdir("."):
        if f.lower().endswith(".mp4") and not f.startswith("out_"):
            process_video(f)
