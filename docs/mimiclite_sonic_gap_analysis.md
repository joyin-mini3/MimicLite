# MimicLite 与 SONIC 差距分析

## 对比范围

本文对比当前仓库的 G1 PPO 训练链路与本地
`/home/amax/Desktop/robot/GR00T-WholeBodyControl` 中的 SONIC 发布配置。MimicLite
以 `mimic-lite/cfg/task/tracking-base.yaml` 和 README 主训练命令为准；SONIC 以
`sonic_release/config.yaml` 及 `gear_sonic/` 实现为准。

## 总体结论

| 维度 | MimicLite | SONIC |
| --- | --- | --- |
| 策略范式 | 参考动作直接驱动 MLP | 多模态编码、FSQ token 化后解码动作 |
| 参考窗口 | `-160 ms` 至 `+80 ms` | G1/Teleop 约 `0.9 s` 前瞻 |
| Value | tracking、loco 两个 head | 单一 value head |
| Reward | 16 个有效项、偏全身精确跟踪 | 12 个有效项、偏关键点与稳定性 |
| 优化目标 | PPO | PPO + 重建/跨模态对齐损失 |

两者都以 50 Hz 输出 29 DoF 关节位置类动作，但 SONIC 不只是更大的 PPO
网络，而是增加了一套动作表征学习系统。

## 1. 网络结构

### MimicLite

Actor 将 `policy` 与 `command` 归一化后直接拼接，再输出高斯策略的均值和
逐关节可学习标准差：

```text
[proprioception, reference command] -> MLP -> 29-D action
```

- 基础配置为三层 `[512, 512, 512]` MLP。
- README 主训练命令使用 residual 配置：基础分支 `[256, 256, 256]`，残差分支
  `[1024, 1024, 1024]`，通过可学习 `sigmoid(alpha)` 融合。
- Critic 为 `[1024, 512, 256]`，分别输出 tracking 和 loco 两个 value。
- 没有 tokenizer、离散量化、动作重建或跨模态编码器。

### SONIC

SONIC 使用 Universal Token 网络：

```text
G1 / Teleop / SMPL reference
        -> modality encoder [2048, 1024, 512, 512]
        -> FSQ: 2 tokens x 32 dimensions
        -> [64-D token_state, proprioception]
        -> dynamic decoder [2048, 2048, 1024, 1024, 512, 512]
        -> 29-D action
```

它还包含用于重建 G1 参考运动的 kinematic decoder。Critic 使用六层大 MLP，
但只输出一个 value。相较 MimicLite，主要差距是网络规模、离散 token 瓶颈及
G1/Teleop/SMPL 共享动作空间，而不是关节动作输出形式。

## 2. 输入格式与数据流

### MimicLite 运行时输入

G1 29 DoF 配置下，Actor 输入约 839 维：

- `policy` 约 535 维：7帧根角速度和重力方向、7帧关节位置和速度、3帧历史
  action；历史索引为 `[0, 1, 2, 3, 4, 8, 16]`。
- `command` 为 304 维：8个参考时刻
  `[-8, -4, -2, 0, 1, 2, 3, 4]`，每帧包含 pelvis 局部位置3维、旋转6维和
  关节位置29维。
- Critic 额外接收 root/body 跟踪误差、动作和力矩等 privileged observation。

训练动作文件主要来自 Any4HDMI 生成的 NPZ/qpos 数据。NPZ 先由 motion loader
转换为参考运动张量，策略本身不直接读取 NPZ。

### SONIC 运行时输入

- Proprioception 使用10帧连续历史：重力方向、基座角速度、关节位置、关节
  速度和历史 action；按29 DoF及标准扁平化推导约为930维。
- G1/Teleop 使用10个、间隔0.1 s的参考帧；SMPL 使用10个、间隔0.02 s的
  参考帧。
- Tokenizer 根据 `encoder_index` 接收 G1 关节运动、VR三点目标或 SMPL
  骨架运动，输出统一的64维 `token_state`。
- 训练配置同时读取 robot motion 和 SMPL motion 目录，而不是 MimicLite 的
  单一 NPZ/qpos 接口。

因此，将 SONIC PKL 预处理成 MimicLite 所需 NPZ 是数据接入层工作，不等同于
移植 SONIC 的 tokenizer 输入和网络结构。

## 3. Reward 差异

### MimicLite 有效 Reward

Tracking 共10项：root位置、姿态、线速度、角速度，body局部位置、姿态、线
速度、角速度，以及 joint position/velocity tracking。

Loco 共6项：action rate、joint velocity、soft joint limit、自碰撞、feet air
time 和 survival。Feet slip 当前关闭。

### SONIC 有效 Reward

SONIC 共12项：anchor位置/姿态、相对body位置/姿态、body线速度/角速度、局部
关键点跟踪、action rate、joint limit、非期望接触、腕部/头部防抖和踝关节
加速度。配置中的 `tracking_vr_5point_local` 函数名是5点，但当前发布配置实际
使用 torso 与两个 wrist，共3个点。

主要差异如下：

- SONIC 没有显式 joint position/velocity tracking、survival 和 feet air time。
- MimicLite 跟踪核主要为 `exp(-mean(error) / sigma)`。
- SONIC 使用平方高斯核 `exp(-mean(error^2) / std^2)`。
- MimicLite 将 tracking/loco 分组并等权组合 advantage；SONIC 使用单标量
  reward/value。

两边 reward 的误差定义和衰减曲线不同，权重与 `sigma/std` 不能直接复制。

## 4. 优化算法

| 参数 | MimicLite PPO | SONIC PPO |
| --- | --- | --- |
| Rollout steps | 32 | 24 |
| PPO epochs | 3 | 5 |
| Minibatches | 8 | 4 |
| `gamma / lambda / clip` | `0.99 / 0.95 / 0.2` | `0.99 / 0.95 / 0.2` |
| Actor 学习率 | `3e-4` | `2e-5` |
| 梯度裁剪 | `1.0` | `0.1` |
| Value loss | 普通 MSE | Clipped value loss |
| 默认优化器链路 | MuonAdamW | TRL/AdamW |
| 辅助损失 | 无 | 重建、跨模态 latent 对齐、循环一致性 |

MimicLite 默认使用8192个环境、训练4000次迭代；SONIC 发布配置使用4096个环境，
最大100000次迭代，并启用基于失败率的动作自适应采样。SONIC 配置同时声明
`critic_learning_rate=1e-3`，但 TRL 顶层 learning rate 绑定到 Actor 的
`2e-5`；复现实验前应确认是否为 Critic 建立了独立参数组。

## Mini3 链路建议

第一阶段应保留 MimicLite 的直接 reference-conditioned PPO：完成 PKL 到 NPZ、
Mini3 joint/body 映射、观测维度和 action scale 适配，再调节 tracking 与稳定性
reward。只有在需要同时支持 SMPL、VR遥操作或跨来源动作泛化时，才有必要引入
SONIC 的多编码器、FSQ 和辅助损失，否则会显著增加训练、导出和 sim2sim 复杂度。

## 已知一致性风险

`sonic_release/g1_wrist_joints_10_clean.yaml` 标注部署总输入为436维，但其维度
分解更接近4帧历史，与训练配置中的10帧 history 不完全一致。正式参考 SONIC
部署链路前，需要使用 checkpoint 的首层权重形状或重新导出的模型确认真实输入
维度，不能只依据该 YAML 注释。
