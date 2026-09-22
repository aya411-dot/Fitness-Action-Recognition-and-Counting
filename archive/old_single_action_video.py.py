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

def process_video(video_path, action_type):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    basename = os.path.basename(video_path)
    out_name = f"out_{action_type}_{basename}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_name, fourcc, fps, (w, h))

    mp_draw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    count = 0
    if action_type in ["squat","pushup","situp"]:
        stage = "up"
    else:
        stage = ""
    time_sec = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        res = pose.process(img_rgb)
        img_rgb.flags.writeable = True
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        show_text1 = f"Action: {action_type}"
        show_text2 = ""
        valid = False
        angle_val = 0

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            def get_pt(idx):
                return [lm[idx].x * w, lm[idx].y * h]

            if action_type == "squat":
                hip, knee, ankle = get_pt(23), get_pt(25), get_pt(27)
                angle_val = calculate_angle(hip, knee, ankle)
                valid = lm[23].visibility>0.5 and lm[25].visibility>0.5 and lm[27].visibility>0.5
                if valid:
                    if angle_val <135 and stage == "up":
                        stage = "down"
                    if angle_val >140 and stage == "down":
                        count +=1
                        stage = "up"
                    show_text2 = f"Count: {count}"
                    print(f"Knee angle:{angle_val:.1f}", end="\r")

            elif action_type == "pushup":
                shoulder, elbow, wrist = get_pt(11), get_pt(13), get_pt(15)
                angle_val = calculate_angle(shoulder, elbow, wrist)
                valid = lm[11].visibility>0.5 and lm[13].visibility>0.5 and lm[15].visibility>0.5
                if valid:
                    if angle_val <110 and stage == "up":
                        stage = "down"
                    if angle_val >140 and stage == "down":
                        count +=1
                        stage = "up"
                    show_text2 = f"Count: {count}"
                    print(f"Elbow angle:{angle_val:.1f}", end="\r")

            elif action_type == "situp":
                shoulder, hip, knee = get_pt(11), get_pt(23), get_pt(25)
                angle_val = calculate_angle(shoulder, hip, knee)
                valid = lm[11].visibility>0.5 and lm[23].visibility>0.5 and lm[25].visibility>0.5
                if valid:
                    if angle_val >100 and stage == "down":
                        stage = "up"
                    if angle_val <95 and stage == "up":
                        count +=1
                        stage = "down"
                    show_text2 = f"Count: {count}"
                    print(f"Torso angle:{angle_val:.1f}", end="\r")

            elif action_type == "plank":
                shoulder, hip, knee = get_pt(11), get_pt(23), get_pt(25)
                s, e, wr = get_pt(11), get_pt(13), get_pt(15)
                torso_angle = calculate_angle(shoulder, hip, knee)
                elbow_angle = calculate_angle(s,e,wr)
                valid = lm[11].visibility>0.5 and lm[23].visibility>0.5 and lm[25].visibility>0.5
                if valid:
                    if 150 < torso_angle <180 and elbow_angle>100:
                        time_sec += 1.0/fps
                    show_text2 = f"Duration: {time_sec:.1f}s"
                    print(f"Torso angle:{torso_angle:.1f}", end="\r")

            elif action_type == "idle":
                show_text2 = "Count: 0"

        cv2.rectangle(img, (10,10), (260,90), (0,0,0), -1)
        cv2.putText(img, show_text1, (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(img, show_text2, (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        mp_draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        out.write(img)

    cap.release()
    out.release()
    print(f"\n{basename} | Finished, {show_text2}")

if __name__ == "__main__":
    import os
    video_root = "./videos/"
    task_list = [
        ("squat", ["squat_01.mp4", "squat_02.mp4", "squat_03.mp4"]),
        ("pushup", ["pushup_01.mp4", "pushup_02.mp4", "pushup_03.mp4"]),
        ("situp", ["situp_01.mp4", "situp_02.mp4", "situp_03.mp4"]),
        ("plank", ["plank_01.mp4", "plank_02.mp4", "plank_03.mp4"]),
        ("idle", ["idle_01.mp4", "idle_02.mp4", "idle_03.mp4"]),
    ]
    for action_name, file_list in task_list:
        print(f"\n========== Process action {action_name} ==========")
        for fname in file_list:
            full_path = os.path.join(video_root, fname)
            if os.path.exists(full_path):
                print(f"Processing: {full_path}")
                process_video(full_path, action_name)
            else:
                print(f"Skip, file not found：{full_path}")
    print("\n===== All tasks finished =====")
