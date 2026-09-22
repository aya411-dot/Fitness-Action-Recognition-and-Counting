import os
import cv2

def check_video_files():
    video_dir = "./videos/"
    video_list = []
    for fname in os.listdir(video_dir):
        if fname.lower().endswith(".mp4"):
            full_path = os.path.join(video_dir, fname)
            cap = cv2.VideoCapture(full_path)
            if cap.isOpened():
                video_list.append(fname)
                print(f"✅ {fname} 可正常读取")
            else:
                print(f"❌ {fname} 视频损坏，无法打开")
            cap.release()
    print(f"\n总有效视频数量: {len(video_list)}")
    print("待推理列表：", video_list)

if __name__ == "__main__":
    check_video_files()
