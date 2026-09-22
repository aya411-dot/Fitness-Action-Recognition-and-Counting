"""
姿态工具模块：角度计算、关键点提取、姿态分类、特征提取、动作状态机
"""
import numpy as np


# ---------- 基础几何 ----------

def calculate_angle(a, b, c):
    """
    计算三点夹角 ∠ABC（单位：度，范围 [0, 180]）
    a, b, c 均为 (x, y) 元组
    """
    a = np.array(a[:2], dtype=np.float32)
    b = np.array(b[:2], dtype=np.float32)
    c = np.array(c[:2], dtype=np.float32)
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-6:
        return 180.0
    cos_ang = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def angle_between(v1, v2):
    """两个二维向量的夹角（度）"""
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return 0.0
    c = np.clip(np.dot(v1, v2) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


# ---------- 关键点提取 ----------

# MediaPipe Pose 关键点索引
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HP, R_HP = 23, 24
L_KN, R_KN = 25, 26
L_AK, R_AK = 27, 28

NEEDED_IDX = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR,
              L_HP, R_HP, L_KN, R_KN, L_AK, R_AK]


def extract_landmarks_px(landmarks, w, h, vis_th=0.25):
    """
    将归一化关键点转为像素坐标，并检查 visibility。
    返回 dict: idx -> (x, y) 或 None（关键点不可见）
    """
    pts = {}
    for i in NEEDED_IDX:
        lm = landmarks[i]
        if lm.visibility < vis_th:
            pts[i] = None
        else:
            pts[i] = (lm.x * w, lm.y * h)
    return pts


def _mid(pts, i, j):
    """两点中点，任一点缺失返回 None"""
    if pts.get(i) is None or pts.get(j) is None:
        return None
    return midpoint(pts[i], pts[j])


def _avg_angle(a1, b1, c1, a2, b2, c2):
    """左右两侧关节角的平均值（有一侧缺失时只用另一侧）"""
    angles = []
    if all(x is not None for x in (a1, b1, c1)):
        angles.append(calculate_angle(a1, b1, c1))
    if all(x is not None for x in (a2, b2, c2)):
        angles.append(calculate_angle(a2, b2, c2))
    if not angles:
        return None
    return sum(angles) / len(angles)


# ---------- 姿态与特征 ----------

def get_body_features(pts):
    """
    从像素关键点中提取动作识别所需的特征。
    返回 dict，缺失的项为 None。
    """
    f = {}

    sh_mid = _mid(pts, L_SH, R_SH)
    hp_mid = _mid(pts, L_HP, R_HP)
    ak_mid = _mid(pts, L_AK, R_AK)

    # 深蹲：膝角（髋-膝-踝），左右取平均
    f['knee_angle'] = _avg_angle(
        pts.get(L_HP), pts.get(L_KN), pts.get(L_AK),
        pts.get(R_HP), pts.get(R_KN), pts.get(R_AK),
    )

    # 俯卧撑：肘角（肩-肘-腕），左右取平均
    f['elbow_angle'] = _avg_angle(
        pts.get(L_SH), pts.get(L_EL), pts.get(L_WR),
        pts.get(R_SH), pts.get(R_EL), pts.get(R_WR),
    )

    # 仰卧起坐：躯干角（肩-髋-膝），左右取平均
    f['hip_angle'] = _avg_angle(
        pts.get(L_SH), pts.get(L_HP), pts.get(L_KN),
        pts.get(R_SH), pts.get(R_HP), pts.get(R_KN),
    )

    # 平板：肩-髋-踝是否接近直线
    f['plank_angle'] = _avg_angle(
        pts.get(L_SH), pts.get(L_HP), pts.get(L_AK),
        pts.get(R_SH), pts.get(R_HP), pts.get(R_AK),
    )

    # 躯干相对竖直方向（图像 y 向下为正）
    if sh_mid is not None and hp_mid is not None:
        torso_vec = (hp_mid[0] - sh_mid[0], hp_mid[1] - sh_mid[1])
        # 竖直向下 (0, 1)
        f['torso_vert_angle'] = angle_between(torso_vec, (0.0, 1.0))
        # 肩髋 y 差（正：髋在肩下方 → 身体基本竖直）
        f['sh_hp_dy'] = hp_mid[1] - sh_mid[1]
        f['sh_hp_dx'] = abs(hp_mid[0] - sh_mid[0])
    else:
        f['torso_vert_angle'] = None
        f['sh_hp_dy'] = None
        f['sh_hp_dx'] = None

    # 髋-踝 相对竖直方向
    if hp_mid is not None and ak_mid is not None:
        leg_vec = (ak_mid[0] - hp_mid[0], ak_mid[1] - hp_mid[1])
        f['leg_vert_angle'] = angle_between(leg_vec, (0.0, 1.0))
    else:
        f['leg_vert_angle'] = None

    # 肩-髋-踝 平均角度用于平板判定，需要躯干也接近水平
    # 记录像素坐标供调试
    f['_sh_mid'] = sh_mid
    f['_hp_mid'] = hp_mid
    f['_ak_mid'] = ak_mid

    return f


def classify_pose(feats):
    """
    粗分类：stand / lying / unknown
    只用躯干角度，不依赖腿（腿容易识别失败）。
    """
    tv = feats.get('torso_vert_angle')
    if tv is None:
        return "unknown"

    if tv < 50:
        return "stand"
    if tv > 55:
        return "lying"
    return "unknown"



def is_prone_orientation(feats):
    """
    在 lying 状态中判断是俯卧（prone）还是仰卧（supine）。
    判据：肩在髋上方（图像坐标 y 小）→ prone；反之为 supine。
    注意：2D 情况下区分有限，仅作为动作优先级辅助。
    """
    dy = feats.get('sh_hp_dy')
    if dy is None:
        return "unknown"
    if dy > 0:      # 髋 y 大于肩 y → 肩在上方 → 俯卧
        return "prone"
    else:
        return "supine"


# ---------- 动作状态机 ----------

class RepCounter:
    """
    通用重复计数状态机（用于深蹲、俯卧撑、仰卧起坐）。
    down_th：进入"下"相位的阈值
    up_th：  进入"上"相位的阈值（up_th > down_th）
    要求连续 min_frames 帧满足阈值才切换状态，抗抖。
    """

    def __init__(self, down_th, up_th, min_frames=4):
        self.down_th = down_th
        self.up_th = up_th
        self.min_frames = min_frames
        self.state = "up"     # up / down
        self.down_cnt = 0
        self.up_cnt = 0
        self.count = 0

    def update(self, value):
        """
        value: 当前帧的关节角度（值越小 → 越接近"下"相位）
        返回：本帧是否计数 +1
        """
        if value is None:
            # 关键点缺失，不更新
            return False

        incremented = False
        if self.state == "up":
            if value < self.down_th:
                self.down_cnt += 1
                if self.down_cnt >= self.min_frames:
                    self.state = "down"
                    self.down_cnt = 0
            else:
                self.down_cnt = 0
        elif self.state == "down":
            if value > self.up_th:
                self.up_cnt += 1
                if self.up_cnt >= self.min_frames:
                    self.count += 1
                    self.state = "up"
                    self.up_cnt = 0
                    incremented = True
            else:
                self.up_cnt = 0
        return incremented


class PlankTimer:
    """
    平板支撑计时器：条件连续满足 min_frames 帧才认为进入平板状态。
    使用帧号计算，避免 POS_MSEC 返回 0 的问题。
    """

    def __init__(self, min_frames=4):
        self.min_frames = min_frames
        self.valid = False
        self.cnt = 0
        self.start_frame = 0
        self.total_frames = 0       # 累计帧数
        self.max_frames = 0         # 单次最长帧数
        self.cur_frames = 0         # 当前这一次的持续帧数

    def update(self, condition, frame_idx):
        """condition: bool，本帧是否满足平板姿态"""
        if condition:
            self.cnt += 1
            if self.cnt >= self.min_frames:
                if not self.valid:
                    self.valid = True
                    self.start_frame = frame_idx - self.cnt + 1
                    self.cur_frames = 0
                self.cur_frames = frame_idx - self.start_frame + 1
        else:
            self.cnt = 0
            if self.valid:
                # 结束一次保持
                self.total_frames += self.cur_frames
                if self.cur_frames > self.max_frames:
                    self.max_frames = self.cur_frames
                self.valid = False
                self.cur_frames = 0

    def finalize(self, frame_idx):
        """视频结束时收尾"""
        if self.valid:
            self.cur_frames = frame_idx - self.start_frame + 1
            self.total_frames += self.cur_frames
            if self.cur_frames > self.max_frames:
                self.max_frames = self.cur_frames
            self.valid = False

    def get_time(self, fps, mode="max"):
        """
        mode: 'max' 取单次最长；'total' 取累计
        """
        if fps <= 0:
            return 0.0
        frames = self.max_frames if mode == "max" else self.total_frames
        return frames / fps