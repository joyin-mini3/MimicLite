# Mini3 训练流程说明

本文档说明当前 Mini3 训练代码中的 reference motion 格式、数据流、Actor/Critic 输入输出、动作执行链以及 reward 实现。

> 版本说明：本文以当前本地代码为准，包含新增的 <code>base_link</code> reference yaw-local 线速度输入。已经启动的旧 W&B run 使用的是修改前的网络输入维度，差异见文末。

核心任务配置：

- [Mini3 tracking 配置](../mimic-lite/cfg/task/tracking-base-mini3.yaml)
- [Sonic motion 配置](../mimic-lite/cfg/task/motion/mini3/sonic.yaml)
- [PPO 实现](../mimic-lite/mimic_lite_learning/ppo.py)

## 1. 总体训练链路

~~~text
PKL 原始动作
  ↓ 转换并重采样到 50 Hz
manifest.json + motions/**/*.npz
  ↓ 根据 Mini3 XML 计算 qvel 和 FK
motion cache：
body pose/velocity + joint position/velocity
  ↓ RobotTracking 采样 motion_id、时间 t 和 future steps
policy / command / priv observation
  ↓ 分组 VecNorm
Actor：采样 21 维 action
  ↓ 延迟 + 平滑 + action scale
21 个关节位置目标
  ↓ 500 Hz PD + real-motor 模型，积分 10 个 physics substep
得到 t+1 仿真状态
  ↓ 与 reference 的 t+1 帧计算 reward
收集 32 个控制步
  ↓
PPO：3 epochs × 8 minibatches
~~~

当前控制与仿真频率：

| 项目 | 配置 | 频率 |
|---|---:|---:|
| Actor/控制周期 | <code>step_dt=0.02 s</code> | 50 Hz |
| MuJoCo 物理周期 | <code>mujoco_physics_dt=0.002 s</code> | 500 Hz |
| 每个 action 的物理步数 | <code>decimation=10</code> | 10 substeps |

## 2. Reference motion 格式

### 2.1 单个 NPZ 文件

每个 motion NPZ 只保存一个数组：

~~~python
qpos: np.ndarray
shape = [T, 28]
dtype = np.float32
~~~

28 维的固定顺序为：

| 范围 | 内容 | 维度 | 单位/表示 |
|---|---|---:|---|
| <code>0:3</code> | <code>base_link</code> 世界坐标位置 xyz | 3 | m |
| <code>3:7</code> | <code>base_link</code> 世界坐标姿态 | 4 | 四元数 wxyz |
| <code>7:28</code> | Mini3 关节位置 | 21 | rad |

21 个关节的固定顺序是：

~~~text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_pitch_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_pitch_joint
right_ankle_pitch_joint
right_ankle_roll_joint
waist_yaw_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_pitch_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_pitch_joint
~~~

NPZ 内不单独保存 <code>joint_names</code>。关节顺序由数据集 [manifest.json](../any4hdmi/output/mini3/sonic/manifest.json)、Mini3 XML 和代码中的严格关节契约共同保证。

### 2.2 数据集 Manifest

当前 Sonic 数据集的主要信息：

| 字段 | 当前值 |
|---|---|
| 采样周期 | 0.02 s，即 50 Hz |
| motion 数量 | 135097 |
| 总时长 | 约 275.96 小时 |
| <code>qpos_dim</code> | 28 |
| Robot XML | <code>mini3.xml</code> |
| root 四元数顺序 | wxyz |
| 关节值 clip | 不执行 |
| root/joint 重采样 | 线性插值 |
| 四元数重采样 | shortest-path SLERP |

原始 PKL 要求包含：

~~~python
root_pos    # [T, 3]
root_rot    # [T, 4]，默认输入顺序为 xyzw
dof_pos     # [T, 21]
fps         # 标量
joint_names # 可选；存在时必须严格匹配 21 关节规范
~~~

转换器会将 root 四元数归一化并转换成 wxyz，随后把动作重采样到 50 Hz。位置和关节使用线性插值，旋转使用 SLERP。

相关实现：

- [Mini3 PKL 转换器](../any4hdmi/src/any4hdmi/scripts/preprocess/mini3_pkl.py)
- [Mini3 MJCF](../any4hdmi/assets/robots/mini3_mjlab/mini3.xml)

## 3. Reference motion 数据流

### 3.1 从 NPZ 到 FK cache

NPZ 只保存 <code>qpos</code>，加载阶段会：

1. 根据相邻 <code>qpos</code> 和 50 Hz 时间间隔计算 <code>qvel</code>。
2. 将 <code>qpos/qvel</code> 输入 Mini3 MuJoCo 模型执行 FK。
3. 生成以下 motion state：

~~~text
body_pos_w       各 body 世界坐标位置
body_quat_w      各 body 世界坐标姿态
body_lin_vel_w   各 body 世界坐标线速度
body_ang_vel_w   各 body 世界坐标角速度
joint_pos        21 个关节位置
joint_vel        21 个关节速度
~~~

当前 Sonic 配置为 <code>full_motion: false</code>，因此大数据集使用 windowed dataset：

- FK 结果保存在磁盘 cache；
- 共享 backing 使用 FP16，降低磁盘与内存占用；
- motion window 输出给 observation/reward 时转换为 FP32；
- 运行时只保留任务需要的 12 个 tracking body 和全部 21 个 joint。

### 3.2 Reference future steps

策略使用 8 个 reference 时间点：

~~~yaml
future_steps: [-8, -4, -2, 0, 1, 2, 3, 4]
~~~

在 50 Hz 下对应：

~~~text
[-160, -80, -40, 0, +20, +40, +60, +80] ms
~~~

每个控制步的 reference 对齐过程：

1. 当前 reference 时间为 <code>t</code>。
2. Actor 读取围绕 <code>t</code> 的 8 帧 reference。
3. Actor 输出 action。
4. 环境执行 10 个 500 Hz physics substep，机器人走到 <code>t+1</code>。
5. Reward 使用 future buffer 中的 <code>+1</code> 帧与新机器人状态比较。
6. 更新时间并生成下一次 observation。

因此 reward 使用的是 next-state 对齐，而不是拿动作执行前的状态和当前 reference 帧计算。

实现见 [RobotTracking command](../mimic-lite/mimic_lite/tasks/command.py)。

### 3.3 Root、anchor 和 reward body

当前配置区分三个概念：

| 用途 | body |
|---|---|
| Reference root、Actor command、anchor | <code>base_link</code> |
| 名为 <code>root_*</code> 的 tracking reward | <code>waist_yaw_link</code> |
| Root termination | <code>base_link</code> |

NPZ 中的 floating root 对应 <code>base_link</code>。<code>waist_yaw_link</code> 的 reference 位姿和速度由 <code>base_link</code> 浮动基座状态、21 个关节状态和 FK 共同得到，不需要在 NPZ 中额外保存。

## 4. Actor 和 Critic 输入输出

以下使用：

| 符号 | 含义 | 数量 |
|---|---|---:|
| J | 关节数 | 21 |
| K | reference 时间点数 | 8 |
| H | 本体历史帧数 | 7 |
| B | tracking body 数 | 12 |

12 个 tracking body 为：

~~~text
base_link
waist_yaw_link
left/right_hip_yaw_link
left/right_knee_pitch_link
left/right_ankle_roll_link
left/right_shoulder_yaw_link
left/right_elbow_pitch_link
~~~

### 4.1 Actor policy 输入：399 维

本体历史使用：

~~~text
history_steps = [0, 1, 2, 3, 4, 8, 16]
时间偏移       = [0, 20, 40, 60, 80, 160, 320] ms 历史
~~~

| 输入 | 形状 | 展平维度 | 意义 |
|---|---:|---:|---|
| <code>root_ang_vel_history</code> | 7×3 | 21 | 机器人 base 机体系角速度 |
| <code>projected_gravity_history</code> | 7×3 | 21 | 重力在机器人机体系的投影，用于感知 roll/pitch |
| <code>joint_pos_history</code> | 7×21 | 147 | 关节位置，减去随机 encoder/action offset |
| <code>joint_vel_history</code> | 7×21 | 147 | 关节速度 |
| <code>prev_actions</code> | 3×21 | 63 | 最近 3 次原始 policy action |
| 合计 |  | **399** | |

Actor 的 policy observation 不包含：

- 机器人世界坐标位置；
- 机器人线速度；
- 机器人绝对 yaw。

### 4.2 Actor command 输入：264 维

| 输入 | 形状 | 展平维度 | 意义 |
|---|---:|---:|---|
| <code>ref_root_pos_future_local</code> | 8×3 | 24 | <code>base_link</code> reference 位置，在当前 reference yaw-local 坐标系下 |
| <code>ref_root_ori_future_b</code> | 8×6 | 48 | reference root 相对当前 robot root 的姿态，使用旋转矩阵前两行的 6D 表示 |
| <code>ref_joint_pos_future</code> | 8×21 | 168 | 8 个时间点的 reference 关节角 |
| <code>ref_root_lin_vel_future_local</code> | 8×3 | 24 | reference <code>base_link</code> 世界线速度，旋转到当前 reference yaw-local 坐标系 |
| 合计 |  | **264** | |

当前本地代码的 Actor 输入总维度：

~~~text
policy 399 + command 264 = 663
~~~

<code>policy</code>、<code>command</code> 和 <code>priv</code> 分别经过独立 VecNorm。Actor 的拼接顺序为：

~~~text
[normalized policy, normalized command]
~~~

Observation 实现：

- [本体历史 observation](../mimic-lite/mimic_lite/tasks/observations/common.py)
- [Reference/tracking observation](../mimic-lite/mimic_lite/tasks/observations/track.py)

### 4.3 Actor 网络架构与输出

当前 Mini3 正式训练命令使用 <code>algo/ppo/module=residual</code>。归一化后的
<code>policy</code> 和 <code>command</code> 按固定顺序拼接成 663 维输入，随后同时进入
Base 分支和 Residual 分支；两个分支不共享参数。

~~~text
normalized policy (399) ─┐
                         ├─ concat → x (663)
normalized command (264) ┘

Base branch:
x
 └→ Linear(663, 256)  → LayerNorm(256)  → Mish
   → Linear(256, 256) → LayerNorm(256)  → Mish
   → Linear(256, 256) → LayerNorm(256)  → Mish
   → h_base (256)

Residual branch:
x
 └→ Linear(663, 1024)   → LayerNorm(1024) → Mish
   → Linear(1024, 1024) → LayerNorm(1024) → Mish
   → Linear(1024, 1024) → LayerNorm(1024) → Mish
   → Linear(1024, 256)
   → h_residual (256)

Feature gate:
h_actor = h_base + sigmoid(alpha_raw) × h_residual

Policy head:
h_actor (256) → Linear(256, 21) → μ (21)
~~~

<code>alpha_raw</code> 是一个可学习标量，初始值为 -4，因此训练开始时：

\[
\alpha=\operatorname{sigmoid}(-4)\approx 0.018
\]

这使网络初始行为主要由较小的 Base 分支决定，训练过程中再自行调节 1024 宽度
Residual 分支的贡献。Residual 分支最后的 <code>Linear(1024, 256)</code> 后没有额外的
LayerNorm 或 Mish。所有 Linear 层使用正交初始化，gain 为 0.01，bias 初始化为 0。

Actor 还有一个可学习、与状态无关的 21 维 <code>σ</code> 参数，初始值为 1。训练时
构造无界对角高斯：

\[
a_t \sim \mathcal{N}\left(\mu(s_t), \operatorname{diag}(\sigma^2)\right)
\]

| 输出 | 维度 | 用途 |
|---|---:|---|
| <code>loc/μ</code> | 21 | 高斯均值 |
| <code>scale/σ</code> | 21 | 各关节动作探索标准差 |
| sampled <code>action</code> | 21 | 训练环境实际使用的动作 |
| <code>log_prob</code> | 1 | PPO importance ratio |

Actor 不使用 tanh，也不对 action 做 clamp。ONNX/部署阶段使用确定性的 21 维均值 <code>μ</code>，不进行高斯采样。

在当前 663 维输入下，Residual Actor 约有 335.6 万个可训练参数，其中包括
Base 分支、Residual 分支、门控标量、均值输出头和 21 维标准差。

模型选择只修改 Actor，不修改 Critic：

| 启动配置 | Actor hidden dims | Residual 分支 |
|---|---|---|
| <code>small</code> | 128×3 | 无 |
| <code>base</code> | 256×3 | 无 |
| <code>large</code> | 512×3 | 无 |
| <code>huge</code> | 1024×3 | 无 |
| <code>base_deep</code> | 256×5 | 无 |
| <code>large_deep</code> | 512×5 | 无 |
| <code>residual</code> | Base 256×3 | 1024×3，再投影到 256，并通过可学习门控相加 |

<code>+exp=ppo/train</code> 在没有显式覆盖时默认选择 <code>large</code>；当前推荐命令和旧
W&B run 都显式使用 <code>algo/ppo/module=residual</code>。输入输出语义不因模型宽度改变。

### 4.4 Critic privileged 输入：474 维

Critic 除了获得 Actor 使用的 policy 和 command，还获得 privileged state：

| Privileged 输入 | 形状 | 展平维度 |
|---|---:|---:|
| Reference root position，相对 robot root | 8×3 | 24 |
| Reference root orientation，相对 robot root，6D | 8×6 | 48 |
| Reference/robot body local position difference | 2×12×3 | 72 |
| Reference/robot body local orientation difference | 2×12×6 | 144 |
| Reference/robot body 世界线速度差 | 2×12×3 | 72 |
| Reference/robot body 世界角速度差 | 2×12×3 | 72 |
| 实际应用的 action | 21 | 21 |
| 实际 actuator torque | 21 | 21 |
| 合计 |  | **474** |

Body difference 使用 <code>diff_future_steps=[0,1]</code>。

Critic 输入总维度：

~~~text
priv 474 + policy 399 + command 264 = 1137
~~~

Critic 与 Actor 不共享网络参数。它将单独归一化的 <code>priv</code>、<code>policy</code> 和
<code>command</code> 按这个顺序拼接为 1137 维输入，网络结构为：

~~~text
normalized priv (474) ────┐
normalized policy (399) ──┼─ concat → x_critic (1137)
normalized command (264) ─┘

x_critic
 └→ Linear(1137, 1024) → LayerNorm(1024) → Mish
   → Linear(1024, 512) → LayerNorm(512)  → Mish
   → Linear(512, 256)  → LayerNorm(256)  → Mish
   → Linear(256, 2)
   → [V_tracking, V_loco]
~~~

最后一层不使用激活函数。与 Actor 一样，Critic 的 Linear 层使用 gain 0.01 的正交
初始化，bias 初始化为 0。当前 1137 维输入下，Critic 约有 182.6 万个可训练参数。

输出不是单一 value，而是：

~~~text
V_tracking
V_loco
~~~

这是因为当前有两个启用的 reward group：<code>tracking</code> 和 <code>loco</code>。PPO 分别计算两路 GAE，再各乘 0.5 后相加并归一化，得到 Actor 使用的最终 advantage。

当前 Residual Actor 与 Critic 合计约有 518.2 万个网络参数；VecNorm 的运行统计是
buffer，不计入这个可训练参数数目。网络构建实现见
[PPOPolicy](../mimic-lite/mimic_lite_learning/ppo.py)、
[Actor head](../mimic-lite/mimic_lite_learning/common.py) 和
[MLP 构造函数](../active-adaptation/active_adaptation/learning/ppo/common.py)。

## 5. Actor 输出如何作用到环境

动作处理实现见 [JointPosition action](../mimic-lite/mimic_lite/tasks/actions.py)。

### 5.1 Action 延迟和平滑

原始 sampled action 首先写入历史 buffer。每个环境在 reset 时随机采样：

~~~text
delay = 1～3 个 500 Hz physics substep
      = 2～6 ms

alpha = 0.8～1.0
~~~

每个物理 substep 执行：

\[
a_{\mathrm{applied}}
\leftarrow
(1-\alpha)a_{\mathrm{applied}}
+
\alpha a_{\mathrm{delayed}}
\]

### 5.2 转换成关节位置目标

21 个关节的 action scale 均为 0.25 rad：

\[
q_{\mathrm{target}}
=
q_{\mathrm{default}}
+
q_{\mathrm{offset}}
+
0.25a_{\mathrm{applied}}
\]

其中：

- 当前 Mini3 default pose 全部为 0；
- <code>q_offset</code> 每个环境随机到 [-0.01, 0.01] rad；
- action 不进行 tanh 或 clamp；
- 最终运动仍会受到物理关节 range 和 motor torque limit 的约束。

### 5.3 PD 和 real-motor 模型

基础 PD 力矩：

\[
\tau_{\mathrm{PD}}
=
K_p(q_{\mathrm{target}}-q)
+
K_d(0-\dot q)
\]

当前 Mini3 的 Kp/Kd：

| 关节组 | Kp | Kd |
|---|---:|---:|
| hip pitch | 60 | 4.5 |
| hip roll | 55 | 2.8 |
| hip yaw | 25 | 1.1 |
| knee pitch | 60 | 4.5 |
| ankle pitch | 50 | 1.2 |
| ankle roll | 45 | 1.2 |
| waist yaw | 65 | 3.0 |
| shoulder pitch | 30 | 1.0 |
| shoulder roll | 25 | 2.0 |
| shoulder yaw | 30 | 1.0 |
| elbow pitch | 20 | 1.0 |

当前 Mini3 实际启用了 real-motor actuator。最终施加到 MuJoCo 的力矩链为：

~~~text
PD torque
 → torque-speed T-N 限制
 → 电流环响应
 → 1 个 physics step 力矩延迟
 → 再次 T-N 限制
 → Kt 输出映射
 → MuJoCo joint torque
~~~

左右脚踝会先通过并联脚踝 Jacobian 映射到两个物理电机空间，分别执行电机限幅，再通过 \(J^T\) 映射回 pitch/roll 关节力矩。

相关实现：

- [Mini3 asset 和 PD 参数](../mimic-lite/mimic_lite/assets/mini3.py)
- [Mini3 real-motor actuator](../mimic-lite/mimic_lite/assets/mini3_real_motor.py)

## 6. Reward、系数和实现

每个 reward term 先执行：

\[
r_i^{\mathrm{weighted}} = w_i f_i
\]

组内求和后，环境交给 PPO 前统一乘控制周期：

\[
\Delta t = 0.02
\]

因此 PPO 得到的 reward vector 是：

~~~text
reward = 0.02 × [R_tracking, R_loco]
~~~

W&B 中每个 <code>reward.group/term</code> 的 EMA 通常对应已经乘 weight、尚未乘 <code>step_dt</code> 的数值。

### 6.1 Tracking reward

Tracking reward 的统一形式：

\[
r =
w\exp\left(
-\frac{\operatorname{mean}(e)}{\sigma}
\right)
\]

| Reward | body/joint | weight | sigma | 误差定义 |
|---|---|---:|---:|---|
| <code>root_pos</code> | <code>waist_yaw_link</code> | 0.5 | 0.2 | 世界位置欧氏距离，m |
| <code>root_ori</code> | <code>waist_yaw_link</code> | 0.5 | 0.4 | 四元数角距离，rad |
| <code>root_linvel</code> | <code>waist_yaw_link</code> | 0.5 | 1.0 | 世界线速度差范数，m/s |
| <code>root_angvel</code> | <code>waist_yaw_link</code> | 0.5 | 2.5 | 世界角速度差范数，rad/s |
| <code>body_pos</code> | 12 bodies | 1.0 | 0.2 | yaw-local body 位置误差 |
| <code>body_ori</code> | 12 bodies | 1.0 | 0.4 | yaw-local body 姿态角误差 |
| <code>body_linvel</code> | 12 bodies | 0.5 | 1.0 | 世界线速度误差 |
| <code>body_angvel</code> | 12 bodies | 0.5 | 2.5 | 世界角速度误差 |
| <code>joint_pos</code> | 21 joints | 0.5 | 0.25 | 平均绝对关节角误差 |
| <code>joint_vel</code> | 21 joints | 0.5 | 2.5 | 平均绝对关节速度误差 |

<code>body_pos/body_ori</code> 在 reference 和 robot 各自的 anchor yaw-local 坐标系中比较，因此主要约束身体相对构型、身体高度和相对朝向；全局平移和全局 yaw 主要由 <code>root_*</code> reward 约束。

实现见 [tracking rewards](../mimic-lite/mimic_lite/tasks/rewards/track.py)。

### 6.2 Loco reward

| Reward | weight | 原始公式与意义 |
|---|---:|---|
| <code>action_rate_l2</code> | 0.1 | \(-\sum_j(a_t-a_{t-1})^2\)，惩罚原始 policy action 突变 |
| <code>joint_vel_l2</code> | \(10^{-4}\) | \(-\sum_j\dot q_j^2\)，惩罚过大关节速度 |
| <code>joint_pos_limits</code> | 10.0 | 对超出 90% soft joint range 的角度偏差求和并取负 |
| <code>self_collisions</code> | -1.0 | 任意有效自碰撞力超过 1 N 时原始值为 1，乘权重后为 -1 |
| <code>feet_slip</code> | 0.5 | 接触地面时，\(-\sum\operatorname{clip}(\lVert v_{\mathrm{foot},xy}\rVert,0,1)\) |
| <code>feet_air_time</code> | 4.0 | 落脚时惩罚有效腾空时间小于 0.8 s |
| <code>survival</code> | 4.0 | 每个控制步原始值为 1，即提供 +4 |

#### Action rate

使用当前和前一次原始 policy action，而不是延迟、平滑后的 applied action：

\[
r_{\mathrm{action\ rate}}
=
-0.1\sum_j(a_t-a_{t-1})^2
\]

#### Joint velocity

\[
r_{\mathrm{joint\ vel}}
=
-10^{-4}\sum_j \dot q_j^2
\]

#### Joint position limits

先把每个关节的硬限位缩放为以中点为中心的 90% soft range，然后计算：

\[
r_{\mathrm{joint\ limit}}
=
-10
\sum_j
\left[
\max(q_{\min,j}^{soft}-q_j,0)
+
\max(q_j-q_{\max,j}^{soft},0)
\right]
\]

#### Self collision

自碰撞 contact sensor 在历史窗口内检测到任意大于 1 N 的有效自碰撞时：

~~~text
raw hit = 1
weighted reward = -1 × hit
~~~

相邻 link 的碰撞由 Mini3 碰撞配置排除，不作为该 reward 的有效自碰撞。

#### Feet slip

使用左右 <code>ankle_roll_link</code>。脚接触地面时计算水平 COM 速度：

\[
r_{\mathrm{feet\ slip}}
=
-0.5
\sum_{\mathrm{feet}}
\operatorname{clip}
\left(
\lVert v_{\mathrm{foot},xy}\rVert,
0,
1
\right)
\]

#### Feet air time

使用左右 <code>ankle_roll_link</code>，高度范围为 [0.03, 0.10] m。

脚高映射：

\[
r_h
=
\operatorname{clip}
\left(
\frac{h-0.03}{0.10-0.03},
0,
1
\right)
\]

有效时间倍率：

\[
f_h = 0.2 + 0.8r_h
\]

腾空期间累积：

\[
T_{\mathrm{air}}
\leftarrow
T_{\mathrm{air}} + 0.02f_h
\]

第一次落地时：

\[
r_{\mathrm{air}}
=
4
\sum_{\mathrm{feet}}
\min(T_{\mathrm{air}}-0.8,0)
\]

所以该项本质上是短腾空惩罚：

- 有效腾空不足 0.8 s：负 reward；
- 达到或超过 0.8 s：该项为 0；
- 腾空超过 0.8 s 不会继续获得额外正奖励；
- standing motion 会屏蔽该项。

相关实现：

- [Feet rewards](../mimic-lite/mimic_lite/tasks/rewards/feet.py)
- [Joint limit/self-collision rewards](../mimic-lite/mimic_lite/tasks/rewards/common.py)
- [通用 joint/reward 实现](../active-adaptation/active_adaptation/envs/mdp/rewards)

### 6.3 只记录、不参与训练的 metrics

下面两个 reward group 配置为 <code>_enabled_: false</code>：

~~~text
tracking_metrics
loco_metrics
~~~

其中包括：

~~~text
root_pos/root_ori/body_pos/body_ori/joint_pos error
joint_pos_limits
feet_contact_count
feet_contact_duration
~~~

这些项目可以出现在 W&B 日志中，但不进入 PPO reward，也不会增加 Critic 的输出维度。

## 7. 当前代码与旧 W&B run 的维度差异

当前本地代码增加了：

~~~text
ref_root_lin_vel_future_local = 8 × 3 = 24 维
~~~

因此当前代码的网络输入为：

~~~text
Actor input  = 663
Critic input = 1137
~~~

此前已经启动的旧 W&B run 在创建网络时还没有这项输入，其维度为：

~~~text
旧 Actor command = 240
旧 Actor input   = 399 + 240 = 639

旧 Critic input  = 474 + 399 + 240 = 1113
~~~

运行中的训练进程不会自动加载之后的本地代码变更。要让 reference root 线速度真正进入 Actor，需要同步当前代码并重新启动训练。旧 checkpoint 的第一层输入宽度与当前网络不同，默认严格加载会发生 shape mismatch；若要继承旧权重，需要单独进行第一层权重迁移。
