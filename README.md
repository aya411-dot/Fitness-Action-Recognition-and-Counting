# Fitness Action Recognition and Counting

## 项目概述

本项目实现一套基于 MediaPipe Pose 的健身动作识别与计数系统，支持三类常见动作的自动识别与计数，并同时提供批量视频处理与 Web 端在线推理两种使用方式。

**核心任务**：给定一段健身视频，自动识别视频中人体正在进行的动作类型，输出对应的重复次数或保持时长。

**技术方案**：基于 MediaPipe Pose 预训练模型提取人体 33 个 2D 骨骼关键点，通过关节向量夹角计算、姿态粗分类、多级状态机、滑动窗口稳定性判据等算法完成动作识别与计数，无需训练任何自定义模型。

**增强特性**：本项目在基础关节角度识别方案之上，新增以下三项关键改进：

1. 滑动窗口稳定性判据：通过统计窗口内肘角极差，区分俯卧撑与平板支撑，解决两类动作在 2D 投影下的混淆问题；
2. 姿态粗分类前置：基于躯干向量与竖直方向夹角，将姿态分为站立/俯卧/仰卧，显著降低跨动作误触发；
3. 帧级连续确认状态机：连续 N 帧满足阈值才切换状态，从根源抑制关键点抖动导致的误计数。



## 架构总览
```text
                    +----------------------------------+
                    |             输入视频              |
                    +------------------+---------------+
                                       |
                                       v
                    +----------------------------------+
                    |           视频预处理层            |
                    |         逐帧读取 RGB 转换         |
                    +------------------+---------------+
                                       |
                                       v
                    +----------------------------------+
                    |            姿态估计层             |
                    |          MediaPipe Pose          |
                    |          33 个 2D 关键点          |
                    +------------------+---------------+
                                       |
                                       v
                    +----------------------------------+
                    |            特征提取层             |
                    |          膝角 / 肘角             |
                    |        躯干角 / 平板直线度         |
                    +------------------+---------------+
                                       |
                                       v
                    +----------------------------------+
                    |           姿态粗分类层            |
                    |        站立 / 俯卧 / 仰卧         |
                    +------------------+---------------+
                                       |
              +------------------------+------------------------+
              |                        |                        |
              v                        v                        v
     +------------------+     +------------------+     +------------------+
     |       深蹲       |     |      俯卧撑      |     |     仰卧起坐     |
     |      状态机      |     |      状态机      |     |      状态机      |
     |                  |     |     平板计时器    |     |                  |
     +--------+---------+     +--------+---------+     +--------+---------+
              |                        |                        |
              +------------------------+------------------------+
                                       |
                                       v
                    +----------------------------------+
                    |              输出层               |
                    |          标注视频 + CSV           |
                    +----------------------------------+
```

## 核心模块详解

### 1. 关节角度计算（calculate_angle）

数学定义：给定三点 A、B、C，夹角 ∠ABC 定义为向量 BA 与 BC 之间的夹角。

```text
        A
         \
          \
           B ------ C

        BA = A - B
        BC = C - B

        angle ABC = arccos( (BA . BC) / (|BA| * |BC|) )
```

实现：

```python
def calculate_angle(a, b, c):
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
```

设计要点：

- 为什么不用 atan2 差值：atan2 差值在跨象限时会发生 2π 翻转，导致角度值剧烈抖动
- 为什么用点积加 acos：点积法天然抗象限翻转，输出稳定在 0 到 180 度区间
- 数值裁剪：np.clip 处理浮点误差导致的超出 -1 到 1 范围的边界情况

### 2. 姿态粗分类（classify_pose）

数学定义：给定肩中点 S、髋中点 H，定义躯干向量与图像竖直方向（y 轴向下）的夹角。

```text
        torso_vec = H - S
        vertical_vec = (0, 1)

        torso_vert_angle = arccos( (torso_vec . vertical_vec)
                                   / (|torso_vec| * |vertical_vec|) )
```

分类规则：

```python
def classify_pose(feats):
    tv = feats.get('torso_vert_angle')
    if tv is None:
        return "unknown"
    if tv < 50:
        return "stand"     # 站立：躯干接近竖直
    if tv > 55:
        return "lying"     # 躺/趴：躯干接近水平
    return "unknown"       # 中间地带：不进入动作判定
```

设计要点：

- 不依赖腿部关键点：脚踝容易被遮挡，若依赖 leg_vert_angle 会导致大量分类失败
- 单变量判断：只用躯干夹角，稳定且鲁棒
- 中间地带返回 unknown：避免硬判导致的误触发

### 3. 俯卧/仰卧方位细分（is_prone_orientation）

在 lying 状态下，进一步区分俯卧与仰卧：

```python
def is_prone_orientation(feats):
    dy = feats.get('sh_hp_dy')   # 髋 y 减 肩 y
    if dy is None:
        return "unknown"
    if dy > 0:
        return "prone"           # 肩在髋上方，俯卧
    else:
        return "supine"          # 肩在髋下方，仰卧
```

使用方式：结合 last_lying_type 做连续性过滤，避免单帧抖动导致分支切换。

### 4. 动作状态机（RepCounter）

状态转移图：

```text
                +---------------------------+
                |            UP             |   初始状态
                +-------------+-------------+
                              |
                              |  连续 N 帧 value < down_th
                              v
                +---------------------------+
                |           DOWN            |   count += 1
                +-------------+-------------+
                              |
                              |  连续 N 帧 value > up_th
                              v
                +---------------------------+
                |            UP             |
                +---------------------------+
```

实现：

```python
class RepCounter:
    def __init__(self, down_th, up_th, min_frames=4):
        self.down_th = down_th
        self.up_th = up_th
        self.min_frames = min_frames
        self.state = "up"
        self.down_cnt = 0
        self.up_cnt = 0
        self.count = 0

    def update(self, value):
        if value is None:
            return False
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
                    return True
            else:
                self.up_cnt = 0
        return False
```

参数说明：

| 参数 | 含义 | 深蹲 | 俯卧撑 |
|---|---|---|---|
| down_th | 进入下相位阈值 | 105 | 115 |
| up_th | 进入上相位阈值 | 150 | 150 |
| min_frames | 连续帧确认数 | 4 | 4 |

设计要点：

- 帧级防抖：连续 N 帧满足条件才切换，抑制关键点抖动
- 相位分离：必须有下到上的完整周期才计数，避免单帧误触
- 阈值可配置：不同动作使用不同阈值

### 5. 滑动窗口稳定性判据

问题分析：俯卧撑撑起的瞬间肘角约 170 度，与直臂平板支撑几乎一致，单帧角度无法区分。

解决方案：使用肘角滑动窗口（约 1 秒）的极差作为稳定性判据。

数学定义：给定长度为 W 的滑动窗口，其中保存最近 W 帧的肘角值。

```text
        elbow_range = max(elbow_history) - min(elbow_history)

        is_stable = (elbow_range < PLANK_STABILITY_TH)
```

实现：

```python
from collections import deque

elbow_history = deque(maxlen=25)   # 约 1 秒 @ 25fps

if elbow is not None:
    elbow_history.append(elbow)

if len(elbow_history) >= 12:
    elbow_range = max(elbow_history) - min(elbow_history)
    is_stable = elbow_range < PLANK_STABILITY_TH
else:
    is_stable = False
```

判别结果对比（基于实测数据）：

| 动作 | 肘角范围 | 窗口极差 | 判定 |
|---|---|---|---|
| 前臂平板 | 83 到 91 | 约 8 | Plank |
| 俯卧撑 | 101 到 171 | 约 70 | Pushup |
| 直臂平板 | 165 到 175 | 约 10 | Plank |

参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| PLANK_STABILITY_WIN | 25 | 滑动窗口帧数，约 1 秒 |
| PLANK_STABILITY_TH | 30 | 窗口内肘角极差阈值 |
| PLANK_MIN_FRAMES | 25 | 连续稳定帧数确认 |

### 6. 平板支撑计时器（PlankTimer）

设计目标：精确统计平板支撑的有效保持时长。

关键实现：

```python
class PlankTimer:
    def update(self, condition, frame_idx):
        if condition:
            self.cnt += 1
            if self.cnt >= self.min_frames:
                if not self.valid:
                    self.valid = True
                    self.start_frame = frame_idx - self.cnt + 1
                self.cur_frames = frame_idx - self.start_frame + 1
        else:
            self.cnt = 0
            if self.valid:
                self.total_frames += self.cur_frames
                if self.cur_frames > self.max_frames:
                    self.max_frames = self.cur_frames
                self.valid = False
```

设计要点：

- 用帧号而非时间戳：避免某些编码格式下 cv2.CAP_PROP_POS_MSEC 返回 0 的问题
- 连续帧确认：至少 N 帧满足才认为进入平板状态，过滤噪声
- 支持两种统计：单次最长或累计时长

### 7. Flask Web 端（app.py）

路由设计：

| 路由 | 方法 | 功能 |
|---|---|---|
| / | GET | 渲染上传页面 |
| /upload | POST | 接收视频，调用 process_one_video，返回 JSON |
| /output/filename | GET | 返回标注后的视频文件 |

返回 JSON 格式：

```json
{
  "squat": "8",
  "pushup": "0",
  "plank": "0.0",
  "video_url": "/output/out_squat_02.mp4"
}
```

设计要点：

- use_reloader=False：避免 OpenCV 写入临时文件触发 Flask 自动重启
- CSV 追加模式：每次上传追加一行，读取最后一行作为本次结果
- 静态目录分离：static/uploads 存上传视频，output 存标注结果

---

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| SQUAT_DOWN_TH | 105 | 深蹲：膝角低于此值视为下蹲 |
| SQUAT_UP_TH | 150 | 深蹲：膝角高于此值视为站起 |
| PUSHUP_DOWN_TH | 115 | 俯卧撑：肘角低于此值视为下压 |
| PUSHUP_UP_TH | 150 | 俯卧撑：肘角高于此值视为撑起 |
| PLANK_HIP_MIN | 130 | 平板：肩髋膝角度最小值 |
| PLANK_STABILITY_WIN | 25 | 平板：滑动窗口帧数 |
| PLANK_STABILITY_TH | 30 | 平板：窗口内肘角极差阈值 |
| PLANK_MIN_FRAMES | 25 | 平板：连续稳定帧数确认 |
| MIN_FRAMES | 4 | 状态机：连续帧确认数 |
| vis_th | 0.25 | 关键点可见性阈值 |

---

## 评估指标

| 指标 | 计算方式 | 说明 |
|---|---|---|
| 计数准确率 | 1 减去预测次数与真实次数差值绝对值除以真实次数 | 越接近 1 越好 |
| 平板时间误差 | 系统计时与真实保持时长差值绝对值除以真实时长 | 越接近 0 越好 |
| 误触发率 | 误判视频数除以总视频数 | 越低越好 |
| 漏检率 | 未识别次数除以实际次数 | 越低越好 |

---

## 实测结果

### 整体识别结果

在 12 段横屏侧拍测试视频（含 3 段静止视频、3 段平板、3 段俯卧撑、3 段深蹲）上的识别结果：

| 视频类型 | 数量 | 期望结果 | 系统识别 | 准确率 |
|---|---|---|---|---|
| idle 静止 | 3 | 全 0 | 0 / 0 / 0 | 100% |
| plank 平板 | 3 | 6 到 10 秒 | 6.2 / 9.1 / 9.1 秒 | 约 95% |
| pushup 俯卧撑 | 3 | 3 到 5 次 | 2 / 4 / 4 | 约 85% |
| squat 深蹲 | 3 | 2 到 10 次 | 2 / 8 / 2 | 约 90% |

### 混淆矩阵

| 真实 预测 | Squat | Pushup | Plank | 无动作 |
|---|---|---|---|---|
| Squat | 3 | 0 | 0 | 0 |
| Pushup | 0 | 3 | 0 | 0 |
| Plank | 0 | 0 | 3 | 0 |
| 静止 | 0 | 0 | 0 | 3 |

结论：三类动作均能稳定识别，动作间无相互误触发。

---

## 性能测试

| 项目 | 数值 |
|---|---|
| 模型体积 | 约 10 MB |
| CPU 单帧推理 | 约 30 毫秒 |
| 处理速度 | 约 30 FPS，1080p |
| 批量处理 12 个视频总耗时 | 约 2 分钟 |
| 内存占用 | 约 300 MB |

测试环境：Windows 11 / Intel i7-1165G7 / 16GB RAM / 无独立显卡

---

## 与基线方案对比

### 与基础关节角度识别方案的差异

| 模块 | 本项目 | 基础方案 |
|---|---|---|
| 姿态分类 | 躯干夹角加中间地带 unknown | 简单 y 差阈值 |
| 俯卧撑平板区分 | 滑动窗口稳定性判据 | 单帧肘角阈值 |
| 状态机防抖 | 连续 N 帧确认 | 单帧切换 |
| 平板计时 | 帧号计算加连续帧确认 | POS_MSEC 时间戳 |
| 关键点容错 | 左右平均加可见性过滤 | 只用左侧关键点 |
| 角度计算 | 向量点积法 | atan2 差值 |

### 与 OpenPose / HRNet 的对比

| 方案 | 模型体积 | CPU 推理速度 | 关键点数量 | 部署难度 |
|---|---|---|---|---|
| OpenPose | 约 200 MB | 约 3 FPS | 25 | 高，需 GPU |
| HRNet | 约 100 MB | 约 5 FPS | 17 | 中 |
| MediaPipe 本方案 | 约 10 MB | 约 30 FPS | 33 | 低，CPU 即可 |

选型理由：MediaPipe 模型体积小 10 到 20 倍，CPU 推理速度提升 6 到 10 倍，适合边缘部署和实时应用。

---

## 项目文件结构

```text
Fitness-Action-Recognition-and-Counting/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── app.py
├── video_new.py
├── image_pose.py
├── check_video.py
├── debug_angles.py
│
├── utils/
│   ├── __init__.py
│   └── pose_utils.py
│
├── templates/
│   └── index.html
│
├── archive/
│
├── docs/
│   ├── web_upload.png
│   ├── web_result.png
│   ├── terminal_output.png
│   └── result_csv.png
│
├── images/
├── videos/
└── output/
    └── result.csv
```

---

## 运行命令

### 环境安装

```bash
pip install -r requirements.txt
```

### 批量处理视频

```bash
python video_new.py
```

自动遍历 videos 目录下所有 mp4 文件，结果输出到 output 目录。

### Web 端在线推理

```bash
python app.py
```

浏览器打开 http://127.0.0.1:5000，上传视频即可在线识别。

### 调试关节角度

```bash
python debug_angles.py ./videos/squat_01.mp4
```

输出每帧的关节角度，用于阈值标定。

---

## 关键技术决策

### 1. 为什么用滑动窗口而非单帧角度区分俯卧撑与平板

踩坑经历：初期用单帧肘角大于 140 度判为平板，导致俯卧撑撑起的瞬间被误判，撑起时肘角也接近 170 度。

解决方案：改用肘角滑动窗口极差，让稳定性成为核心判据。平板支撑的肘角在窗口内极差小于 10 度，而俯卧撑的肘角在窗口内极差大于 60 度。

效果：两类动作完全区分，误触发率降为 0。

### 2. 为什么必须先做姿态粗分类

深蹲站立与俯卧撑俯卧的关节角度特征完全不同，直接判断容易混淆。通过躯干向量与竖直方向夹角先粗分类，可显著降低跨动作误触发。

关键设计：中间地带返回 unknown，宁可不判，也不误判。

### 3. 2D 姿态估计的视角依赖性问题

调试发现：正面拍摄的平板支撑视频完全无法识别。

原因分析：2D 投影下，肩髋踝在图像上不是一条直线，从正面看是上下分布的，躯干角度失真，torso_vert_angle 落在 30 到 50 度的灰色地带。

最终方案：所有测试视频采用横屏侧拍加全身入镜，关键点可见性从 0% 提升到 95% 以上。

方法论启示：2D 姿态估计对拍摄角度敏感，实际部署时需明确使用场景约束。

---

## 已知局限

| 局限 | 原因 | 改进方向 |
|---|---|---|
| 仅支持单人 | MediaPipe 只输出 1 组关键点 | 多人检测加关键点跟踪 |
| 依赖侧面拍摄 | 2D 姿态估计固有缺陷 | 3D 姿态估计 BlazePose GHUM |
| 平板时间偏短 | 稳定判定的头尾帧被过滤 | 引入时序平滑 卡尔曼滤波 |
| 严重遮挡下失效 | 关键点缺失 | 关键点补全 GCN 扩散模型 |
| 动作类型有限 | 硬编码规则 | 引入时序模型 LSTM Transformer |

---

## 未来工作

1. 多动作扩展：基于骨架序列训练轻量级时序分类器，支持更多动作类型，如弓步、引体向上等
2. 多人场景：结合目标跟踪 ByteTrack 或 DeepSORT 实现多人独立计数
3. 3D 姿态估计：使用 MediaPipe Holistic 或 BlazePose GHUM 获取 3D 关键点，解决遮挡与视角问题
4. 实时部署：结合 ONNX Runtime 或 TensorRT 进一步加速，支持移动端实时推理
5. 动作质量评估：不仅计数，还评估动作标准度，如深蹲深度、俯卧撑肘角范围

---

## 参考文献

1. Lugaresi C, Tang J, Nash H, et al. MediaPipe: A Framework for Building Perception Pipelines. arXiv preprint arXiv:1906.08172, 2019.
2. Bazarevsky V, Grishchenko I, Raveendran K, et al. BlazePose: On-device Real-time Body Pose Tracking. arXiv preprint arXiv:2006.10204, 2020.
3. Cao Z, Hidalgo G, Simon T, et al. OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. IEEE TPAMI, 2019.
4. Sun K, Xiao B, Liu D, et al. Deep High-Resolution Representation Learning for Visual Recognition. IEEE TPAMI, 2019.

---

## License

MIT
```