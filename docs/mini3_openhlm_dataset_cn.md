# Mini3 随机搬运数据与 OpenHLM 训练接口

`record_mini3_openhlm.py` 录制完整的随机取物、携物行走和入篮任务，输出遵循 [OpenHLM 官方数据](https://huggingface.co/datasets/OpenHLM/OpenHLM-data)所用的 **LeRobot v2.1** 布局。Mini3 使用独立的 **36 维**状态和动作定义；这不是官方 G1 的 34 维关节接口。

OpenHLM 现在也已集成原生 `mini3_pick_carry` 配置。直接在该工程训练的命令和环境要求见 [Mini3 原生训练说明](../../OpenHLM/src/openpi4OpenHLM/docs/mini3_training_cn.md)。下文同时保留本工程的数据采集和适配脚本用法。

## 数据位置与采集

在工程根目录执行：

```bash
active-adaptation/venv/mjlab/.venv/bin/python record_mini3_openhlm.py collect \
  --episodes 200 --workers 4 --render-workers 6 \
  --output outputs/mini3_openhlm_recording
```

恢复已有批次时添加 `--resume`。恢复会核对采集配置和六个核心脚本的 SHA-256；已导出片段还会核对来源、全局索引和 Parquet 校验和。并发数可以调整。恢复前应确认上一调度进程及其子进程已经结束，不能向同一目录同时启动两个调度器。部分中断的原始尝试保留诊断并由后续种子补录。

单条原始试录：

```bash
active-adaptation/venv/mjlab/.venv/bin/python mini3_collect_episode.py \
  --seed 42 --output outputs/mini3_single_record --onnx-threads 1
```

输出组织：

```text
outputs/mini3_openhlm_recording/
  collection_config.json          # 种子、数量、质量门限
  collection_status.json          # accepted/exported/complete 与尝试记录
  source_hashes.json              # 核心脚本版本
  asset_provenance.json           # 模型、参考及场景来源
  attempts/attempt_000000/
    episode.json, episode_scene.xml
    report.json, trajectory.npz, task_reference.npz
    control_samples.npz, control_metadata.json
    collection.log, export.log
  lerobot/local/mini3_pick_carry/
    data/chunk-000/episode_000000.parquet
    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/episodes_stats.jsonl
    meta/mini3_schema.json
    meta/collection_manifest.json
    meta/recommended_split.json
```

正式训练数据仅包含通过筛选的 200 个 episode。失败、接触超限以及并行预取但未选入的尝试保留在 `attempts`，不计入这 200 条。状态与诊断产物均位于 Git 忽略的 `outputs` 中。

采集沿用当前场景随机化：各方块 XY 扰动 ±1.2 cm、yaw ±12°，目标颜色在红绿蓝中等概率抽取。固定主种子和尝试编号生成独立、可复现的场景种子。颜色仍有各自的摆放区域，桌面、篮子、背景和光照没有额外随机化；这是一组有限范围的示范，不能据此保证泛化到新布局。

## 质量与时序

每条入选记录必须成功完成入篮、原 21 轴 policy 输出覆盖量为 0、数值有效，并且每个 20 ms 动作确实执行了 10 个 2 ms 物理子步。采用现有 real-motor 和碰撞监控，不改变动作执行轨迹。额外要求整条任务的最大异常穿透不超过 **0.5 mm**；比控制器原来的 2 mm 中止门限更严格，但不等于完全没有接触。

相机、状态和动作按原控制频率 **50 Hz** 对齐；官方示例为 30 Hz，这里不对控制信号插值。先记录 `t` 时刻执行动作前的 qpos/qvel 和刚计算出的动作，再执行 `[t,t+0.02)`。`next_*` 保存实际后继状态。图像随后从同一个前状态离线渲染，不推进物理时间、不重新运行控制器。因此 RGB 与 state 对应同一时刻，actions 是接下来执行的目标。

## LeRobot 字段

| 字段 | 类型与含义 |
| --- | --- |
| `head_image_left` | 头部 `head_rgb`，45° 向下 |
| `left_wrist_image` | 左夹爪 `left_gripper_rgb` |
| `right_wrist_image` | 右夹爪 `right_gripper_rgb` |
| `state` | float32[36]，动作执行前的本体状态 |
| `actions` | float32[36]，下述 Mini3 分层控制参考 |
| `timestamp` | float32，每条从 0 开始，间隔 0.02 s |
| `frame_index` | int64，片段内连续帧编号 |
| `episode_index` | int64，0 到 199 |
| `index` | int64，整个数据集连续帧编号 |
| `task_index` | int64，0=red、1=green、2=blue |

三路图像均为 RGB 224×224，JPEG quality=95、subsampling=0，按 Hugging Face `Image` 特征直接嵌入 Parquet，不依赖外部 PNG/MP4。`total_videos=0` 是正确值。渲染先使用 320×240 保留现有相机宽高比，再缩放到 224×224；关闭阴影、反射和离屏抗锯齿以加快数据生成。相机安装位置和碰撞模型不变。左臂收纳期间，其相机多数看到地面和夹爪；各路仍按真实视角保存。

任务文本严格为以下三句之一：

```text
Please put the red square from the tabletop into the basket.
Please put the green square from the tabletop into the basket.
Please put the blue square from the tabletop into the basket.
```

## Mini3 的 36 维契约

精确逐列名称与单位见 `meta/mini3_schema.json` 和 `mini3_vla_features.py`。

| 维度（从 0 开始） | 内容 |
| --- | --- |
| 0–6 | 左肩 pitch/roll/yaw、原肘 pitch、新肘 yaw、腕 roll/pitch |
| 7 | 左夹爪闭合程度 |
| 8–14 | 右臂相同七轴顺序 |
| 15 | 右夹爪闭合程度 |
| 16–21、22–27 | 左、右腿：髋 pitch/roll/yaw、膝 pitch、踝 pitch/roll |
| 28 | 腰 yaw；没有虚构 G1 的另外两个腰轴 |
| 29–31 | 根 roll、pitch、yaw 角速度 |
| 32–34 | 当前实测 yaw 对齐的 heading 坐标系下根线速度 XYZ；Z 保持世界竖直 |
| 35 | 根高度，世界坐标 Z |

关节与姿态单位 rad，角速度 rad/s，线速度 m/s，高度 m。夹爪归一化为 `clip(1-half_opening/0.035,0,1)`，0 全开、1 全闭。

`state` 使用实测关节和根状态。`actions` 中，原 21 轴使用给 MimicLite 的当前参考角，新增六轴使用真实规划后、含积分补偿的电机目标，夹爪使用控制开度。根姿态、高度和速度使用门控后的参考，其中线速度包括闭环校正，yaw 速度来自参考下一帧的跨 ±π 正确角差。

该 36 维向量适合作为新 Mini3 配置的学习目标，但不是 MimicLite 完整输入的无损编码：原控制器还有历史与未来参考窗口。未来部署 VLA 时，需要实现并验证 Mini3 参考解码、时序管理和状态估计接口；不能直接接官方 G1 的部署程序。原始记录另外保留完整未来参考、264 维 policy command 输入、399 维本体历史输入、原关节 q/dq/effort/kp/kd、新增轴实际目标和力矩前馈，支持之后重做映射。

老师控制器仍使用 MuJoCo 的物体真值规划及判定抓取。训练用 `state` 不包含方块或篮子位置，但根高度、线速度等本体量也是仿真理想值，实机需要对应估计器。RGB 没有叠加目标框、状态机阶段或特权坐标。通过示范训练 VLA 不等于已经完成了实机视觉闭环。

## OpenHLM 验证、归一化与训练

`mini3_openhlm_training.py` 在运行时注册 `mini3_pick_carry` 配置，不修改旁边的 OpenHLM 仓库。它使用真实 OpenHLM 数据加载器，明确使用 36 维输入/输出和 50 帧动作窗口（1 s），不应用 G1 的 34 维截断或 delta-joint 规则。

验证所有 episode 的首、中、尾样本、任务文本、相机解码和跨片段边界；如果已有归一化统计，还会检查归一化后的数值和反归一化还原：

```bash
active-adaptation/venv/mjlab/.venv/bin/python mini3_openhlm_training.py validate \
  --report outputs/mini3_openhlm_recording/openhlm_validation.json
```

逐行对照原始记录、检查全部 Parquet SHA-256，并为全数据生成训练归一化统计：

```bash
active-adaptation/venv/mjlab/.venv/bin/python validate_mini3_openhlm_dataset.py \
  --norm-output outputs/mini3_openhlm_recording/openpi_assets/mini3_pick_carry/local/mini3_pick_carry
```

该命令调用 OpenHLM 的 `RunningStats` 和序列化器，统计所有状态与带末帧重复补齐的 50 帧动作窗口；计算数值统计不需要重复解码 RGB。OpenHLM 原始归一化脚本也可通过以下入口运行：

```bash
active-adaptation/venv/mjlab/.venv/bin/python mini3_openhlm_training.py norm \
  --workers 2 --batch-size 16 -- --max-frames 10000
```

后一命令使用抽样帧，会覆盖相同路径的统计；不要无意覆盖已经生成的全量统计。试验可用 `--artifacts-root outputs/another_norm_run` 隔离。

实际训练应使用按 OpenHLM README 配置的完整训练环境与 π0.5 基础权重，示例：

```bash
python mini3_openhlm_training.py train \
  --weights /path/to/pi05_base_pytorch --workers 4 --batch-size 16 \
  -- --exp-name mini3_pick_carry_v1
```

`/path/to/pi05_base_pytorch` 需要换成含 `model.safetensors` 的实际目录。本次工作不下载大模型或启动 VLA 训练。数据加载检查覆盖格式、动作窗口与图像预处理；不包含 tokenizer、模型权重加载或训练收敛验证。

数据根变量应为 **`HF_LEROBOT_HOME`**。OpenHLM 固定的 LeRobot 版本会拒绝旧 `LEROBOT_HOME`；启动器会设置正确变量。用于本次 CPU 验证的额外依赖隔离在 `.cache/mini3_vla_deps`，不替换仿真环境的 torch/numpy，也不等价于完整 GPU 训练环境。

默认元数据的 `train` 包含全部 200 条。`recommended_split.json` 另给出整 episode 的 180/20 划分建议，但当前训练入口不会自动启用该划分；需要评测时应按该列表建立训练/验证子集，并仅在训练子集计算统计，避免帧级切分造成泄漏。

最终数量、帧数、颜色分布、文件大小及逐行校验结果见 `collection_status.json`、`integrity_validation.json`；真实 OpenHLM 加载结果见 `openhlm_validation.json`。

## 本次完整录制结果（2026-09-18）

本地数据集位于 `outputs/mini3_openhlm_recording/lerobot/local/mini3_pick_carry`，已完成全部录制与验证。

| 项目 | 实测结果 |
| --- | --- |
| 完整成功片段 / 独立场景种子 | 200 / 200 |
| 目标颜色 | red 50、green 80、blue 70 |
| 总帧数 / 频率 | 239,861 / 50 Hz |
| 总仿真时长 | 4,797.22 s，约 79.95 min |
| 单条时长 | 23.76–24.66 s |
| 三路内嵌 RGB 总数 | 719,583 张，224×224 |
| Parquet 总大小 | 11,986,174,938 bytes，约 11.16 GiB |
| 入选记录最大异常穿透 | 0.4987 mm，低于 0.5 mm 门限 |

按顺序筛选了 213 次尝试：200 条入选，9 条因接触质量未入选，4 条任务失败。另有 3 条并行预取的原始尝试未进入筛选；`attempts` 总计 216 个目录。颜色没有强行均衡，保留随机抽取和质量筛选后的分布。

完整验证通过：

- 全部 200 个 Parquet 的 SHA-256 一致；239,861 行状态与动作逐行对应执行前的原始记录，索引和时间连续。
- 独立完整性检查解码了 1,800 张抽查图像；导出时已解码全部 JPEG 来计算图像统计。
- 本地 OpenHLM 的真实加载器读取每条的首、中、尾，共 600 个样本，检查三路图像、任务文字、50 帧动作窗口及末尾补齐；另对相同 600 个样本检查归一化与反归一化。
- 全量归一化统计已保存到 `outputs/mini3_openhlm_recording/openpi_assets/mini3_pick_carry/local/mini3_pick_carry/norm_stats.json`。
- 相关回归测试共 157 项通过；测试日志为 `mini3_tests.log` 和 `recording_validation_tests.log`，其中 7 项在后一个日志中重复执行。

格式和 Mini3 训练输入接口已经验证可用。尚未加载大模型权重、运行训练或验证训练收敛；部署仍需要前述 Mini3 参考解码接口。`openhlm_runtime_provenance.json` 记录本次使用的 OpenHLM 提交及关键源码校验和。
