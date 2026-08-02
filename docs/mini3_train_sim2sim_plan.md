# Mini3 Train 与 Sim2Sim 链路实施方案

> 状态：核心代码已实施并通过本地配置/CPU MuJoCo 测试；待服务器 RTX 4090 训练 smoke、正式训练、checkpoint 导出与策略 Sim2Sim 验收；更新日期：2026-08-02  
> 范围：Mini3 动作跟踪训练、策略导出和 MuJoCo Sim2Sim。训练环境以服务器 NVIDIA RTX 4090 为准；本阶段不处理本机 5090/SM120 适配。真机通信、PICO 遥操作及实机安全流程不在本阶段范围内。

## 1. 结论与目标

当前工程已经注册 Mini3 训练资产、任务与动作数据配置，并补齐 `sim2real` 的 Mini3 RobotCfg、严格接口校验和集成 Sim2Sim 映射。参考工程 `/home/amax/Desktop/robot/UFO` 提供的 Mini3 21 关节顺序、MJCF 和电机参数仍作为待实测确认的基线；其资产许可证与来源声明已随本地资产保留，重新分发或商用前仍需单独复核。

目标链路如下：

```text
合法且已验证的 Mini3 MJCF + URDF/USD
  -> any4hdmi 50 Hz qpos 动作
  -> MimicLite + MJLab 单阶段 PPO
  -> 单一完整 ONNX + 部署 YAML
  -> 最小 sim2real Mini3 RobotCfg/观测构造（仅服务集成 Sim2Sim）
  -> integrated_sim2sim
  -> trajectory.npz + tracking metrics
```

## 2. 不可变接口契约

### 2.1 关节与动作

训练、动作数据、ONNX 输出和 Sim2Sim 必须共同使用以下 21 关节顺序：

```text
 0 left_hip_pitch_joint         11 right_ankle_roll_joint
 1 left_hip_roll_joint          12 waist_yaw_joint
 2 left_hip_yaw_joint           13 left_shoulder_pitch_joint
 3 left_knee_pitch_joint        14 left_shoulder_roll_joint
 4 left_ankle_pitch_joint       15 left_shoulder_yaw_joint
 5 left_ankle_roll_joint        16 left_elbow_pitch_joint
 6 right_hip_pitch_joint        17 right_shoulder_pitch_joint
 7 right_hip_roll_joint         18 right_shoulder_roll_joint
 8 right_hip_yaw_joint          19 right_shoulder_yaw_joint
 9 right_knee_pitch_joint       20 right_elbow_pitch_joint
10 right_ankle_pitch_joint
```

MimicLite 的动作语义为：

```text
q_target = default_joint_pos + action_scale * policy_action
```

Mini3 沿用当前 MimicLite PPO 与 `JointPosition` 的既有动作语义，不引入 UFO 的动作归一化、缩放或限幅管线，也不为 Mini3 单独增加 tanh/clamp。每组 `action_scale` 必须显式配置，并保证训练、策略导出和 Sim2Sim 使用相同的 `default_joint_pos`、joint order 与 `action_scale`。

Mini3 启用严格模式：21 个 joint 名称必须唯一；`default_joint_pos`、Kp、Kd、`action_scale` 和 motion joint 必须对这 21 个 joint 完整覆盖、无重复、无额外项。任何缺失、重复或顺序不一致都必须在启动时失败，不允许静默补零或跳过。

### 2.2 动作数据

- 输入为可信来源的 Mini3 flat PKL，每条动作必须显式包含 `root_pos[T,3]`、`root_rot[T,4]`、`dof_pos[T,21]` 和 `fps`。源 PKL 的 `dof_pos` 位置契约固定为 2.1 节列出的 21 关节顺序；转换器不得从 XML 顺序、字母顺序或集合推断该顺序，也不接受未声明的其他排列。
- PKL 先通过 any4hdmi 离线转换为当前工程的 NPZ `qpos` 数据集。训练和 Sim2Sim 只读取转换后的 `manifest.json + motions/**/*.npz`，运行时不直接读取 PKL。
- 统一频率：reference/policy 为 50 Hz，`step_dt=0.02`；MuJoCo physics 为 500 Hz，`sim_dt=0.002`，decimation=10。源 PKL 不是 50 Hz 时，转换阶段复用 any4hdmi 的 MuJoCo-aware qpos 插值（`interpolate_qpos_qvel_batch_torch` 或等价实现）做确定性重采样，禁止对四元数分量直接线性插值。
- any4hdmi 输出每帧 `qpos.shape == (28,)`：`root_xyz[3] + root_quat_wxyz[4] + joints[21]`。
- PKL 的 `root_rot` 为 `xyzw` 时，转换为 MuJoCo 的 `wxyz`，归一化四元数并处理相邻帧符号连续性。
- Mini3 free joint 为 `floating_base`。转换器应按编译后的 MJCF 发现 free joint，并断言 `manifest.qpos_names` 与从该 MJCF 提取的 qpos names 完全一致，不能只检查宽度为 28。
- `manifest.json`、NPZ 动作文件和 MJCF 必须来自同一套 joint/body name contract；禁止使用 G1 动作或 checkpoint。

### 2.3 策略 I/O

首版只接单阶段 PPO，沿用当前 exporter 和 `sim2real` 的无 batch 推理约定，导出一个包含完整 Actor 的静态 ONNX，不要求动态 batch。沿用现有历史步和短期未来帧时，预期输入为 `policy[399]`、`command[240]`，输出为 `action[21]`：

- `policy = root_ang_vel(7*3) + gravity(7*3) + joint_pos(7*21) + joint_vel(7*21) + prev_action(3*21)`；
- `command = future_root_pos(8*3) + future_root_rot6d(8*6) + future_joint_pos(8*21)`；
- history 为当前帧优先，future steps 为 `[-8,-4,-2,0,1,2,3,4]`。

这些维度是实施前的预期值，最终必须用训练配置、导出 YAML 和 ONNX Runtime 三方校验，不能硬编码为事实。

## 3. 分阶段实施

### 阶段 0：4090 服务器环境

1. 以服务器 NVIDIA RTX 4090 作为唯一训练环境验收目标，不在本方案中处理本机 5090/SM120 的覆盖或隔离安装。
2. `any4hdmi/` 作为根仓库直接跟踪的普通源码目录，不设置独立 `.git`，也不加入 `.gitmodules`；服务器克隆根仓库即可获得转换代码和 Mini3 资产。既有 `active-adaptation`、`mimic-lite`、`sim2real` 仍保持固定 submodule commit，可用 `git clone --recurse-submodules` 一次取得。之后用 `uv --project venv/mjlab sync --locked` 创建或同步环境；禁止使用未纳入版本控制的本地目录满足 editable dependency。
3. 在服务器的 `active-adaptation/venv/mjlab` 环境中确认 PyTorch 可见全部训练 GPU、每张卡 capability 为 `(8, 9)`，并完成 CUDA tensor、Warp 和 MJLab 导入/运行检查。
4. 正式训练前先运行单卡 1-env 启动测试，再运行计划使用 GPU 数量的最小 DDP 测试；两者均通过后才允许开始 4000-iteration 训练。

### 阶段 A：资产与数据打底

1. 在 `mimic-lite/mimic_lite/assets/mini3.py` 定义 `AssetCfg`，在 `assets/__init__.py` 注册 `mini3-mesh`；固定 21 joint、body、root `base_link`、接触脚体和左右镜像映射。除合法 MJCF 外，还必须为 `AssetCfg.usd_path` 提供合法且可追溯的 Mini3 URDF/USD，不能用不存在的占位路径。
2. 以 UFO 参数为“待验证初值”录入关节位置/速度限位、effort、Kp/Kd、armature 和 friction。MJLab 当前要求单个 `ActuatorCfg` 的参数为标量，因此每个关节单独建 actuator，或只合并完整参数元组相同的关节。
3. 当前 `ActuatorCfg.mjlab()` 不会把 `velocity_limit` 传给 MJLab actuator，集成 Sim2Sim 也只裁剪 effort。首版将此记录为已知限制，并在站立、训练 smoke 和 Sim2Sim 中输出每关节最大/p99 速度及超限比例；不得把“参数已填写”表述为“速度限位已执行”。若持续超限，必须先增加约束、reward/termination 或 actuator 实现，再进入正式训练。
4. 将 Mini3 collision geom 默认值从 `contype=0` 改为 `contype=1, conaffinity=1`，打开机器人自碰撞能力；保留并测试 MJCF `<contact><exclude>` 中的相邻 link 排除关系。验收范围限定为具有 collision geom 的 body：至少一对未排除的非相邻 body 可以产生碰撞，明确排除且双方均有 collision geom 的相邻 body pair 不产生碰撞。
5. 修改 `active-adaptation/active_adaptation/envs/backends/mjlab/env.py`，从 `cfg.sim.mujoco_physics_dt` 读取物理步长，并保留 `0.005` 作为旧任务默认值；Mini3 任务设置为 `0.002`。同时校验 `step_dt / physics_dt` 是正整数，禁止静默截断 decimation。
6. 在 any4hdmi 增加 Mini3 flat PKL 到 NPZ qpos 的转换入口和 schema 测试。按 2.1 节固定顺序映射 `dof_pos`，输出前检查字段、`T x 28`、50 Hz、finite、四元数范数/连续性、关节限位、root 高度、manifest qpos names 和首帧站立姿态。
7. 用 MuJoCo 完成静态站立测试：首帧 PD 保持至少 10 秒，无穿模、发散、持续饱和力矩或明显脚底滑移后再进入 RL。

### 阶段 B：训练任务

1. 以 `tracking-base-atom.yaml` 为模板新增 `mimic-lite/cfg/task/tracking-base-mini3.yaml`，替换所有 G1 body/joint 表达式；root、anchor 使用 `base_link`，termination root 在 `base_link` 与 `waist_yaw_link` 中通过模型验证后固定。首版 tracking bodies 使用 `base_link`、`waist_yaw_link`、双侧 hip-yaw、knee-pitch、ankle-roll、shoulder-yaw 和 elbow-pitch link，并断言每条表达式至少解析到一个 body。
2. 新增 `mimic-lite/cfg/task/motion/mini3/sonic.yaml`，指向 Mini3 any4hdmi manifest/MJCF；reward、termination、contact body 只能引用 Mini3 实际名称。
3. Mini3 的 `feet_air_time` 只配置 `body_names=[left_ankle_roll_link, right_ankle_roll_link]`，省略 `body2_names`，使接触和脚高均直接依赖两个 ankle-roll link；对应 contact sensor 必须完整匹配这两个 body。
4. 启用 `self_collisions` reward/sensor，并用最小测试确认具有 collision geom 的相邻 body 排除生效、未排除的非相邻自碰撞仍可被传感器和 reward 观测到。
5. 首版保持现有 observation 顺序、history steps 和 future steps，不同时引入 ROA、SAC 或新观测，降低部署对齐变量。
6. 先运行配置展开和 64-env/10-iteration smoke test，确认 action dim=21、实际 physics dt=0.002、严格字段覆盖、无 G1 名称、无 NaN 后，再扩展到多 GPU 4000 iterations。

> 训练环境通过 `active-adaptation/venv/mjlab/pyproject.toml` 直接 editable 安装根仓库中的 `../mimic-lite`，不再依赖 `active-adaptation/projects/mimic-lite` 或额外第三方 checkout。

### 阶段 C：导出与数值对齐

1. 复用 `mimic-lite/scripts/play.py` 的 `export_policy=true export_only=true`，不要新增旁路 exporter。
2. 扩展导出 YAML，除已有 joint/body、default pose、Kp/Kd、action scale 和 observation 外，增加：
   - `evaluation.tracking_body_names`；
   - `evaluation.termination_root_body_name`；
   - `evaluation.anchor_body_name`。
3. 用 ONNX checker/Runtime 验证输入名称、dtype、无 batch 静态维度、21 维输出和 joint order；不修改 exporter 以支持动态 batch，也不在集成运行时额外增加 batch 维。
4. 导出时严格校验 21 个 joint 的 default pose、Kp、Kd 和 `action_scale` 全覆盖且无重复；不允许部署 YAML 中存在会在运行时补零的缺项。
5. 构造同一批固定状态，逐项比较训练端与集成 Sim2Sim 的 observation：坐标系、COM/root 角速度、四元数约定、rotation-6D、历史方向、future 索引、默认关节偏置、上一动作和 action scale；部署侧禁用训练噪声与随机化。默认门槛为 `max_abs_error <= 1e-5`；无法达到时记录数值来源和合理容差。
6. 单独验证 reset 语义：机器人同步到 motion frame 0 后再 reset observations；root angular velocity、projected gravity、joint position 和 joint velocity 的全部历史槽用当前值填满，previous actions 清零。比较“刚 reset 的第一帧”和“连续运行至少 16 帧”两种状态，二者都必须满足 parity 门槛。

### 阶段 D：Sim2Sim 接入

本阶段只修改完成 `integrated_sim2sim` 所必需的最小接口，不扩展真机通信、遥操作或其他 `sim2real` 功能，也不处理与本链路无关的 `sim2real` 环境问题。

1. 新增 `sim2real/sim2real/config/robots/mini3.py`，填充 21 joint、body、限位、armature/friction、MJCF、28 维 default qpos、`root_joint_names=("floating_base",)` 和 viewer body；在 `robots/__init__.py` 注册 `mini3`。
2. `integrated_sim2sim.py` 从策略 YAML 读取 tracking/root/anchor 字段，旧 G1 常量仅作向后兼容 fallback；Mini3 不允许命中 fallback。
3. 根据 MuJoCo actuator 的 transmission target joint 建立 `joint_name -> actuator_id` 映射，不依赖 actuator 自身名称；自动补 actuator 时也按 target joint 判断是否已存在。每个受控 joint 必须恰好映射到一个 actuator，缺失或一对多都 fail-fast，并在模型加载完成后再次断言 `nu=21`。
4. 对 RobotCfg、部署 YAML 和 NPZ motion 启用 Mini3 严格模式：21 个 joint、default pose、Kp、Kd、`action_scale`、motion joint 必须完整覆盖且无重复，不再静默补零、追加额外 joint 或跳过缺失项。
5. 为通用 history observations 实现与训练端一致的 `reset()`：frame 0 状态同步后填充当前状态历史，previous actions 在每次重新播放时清零，禁止沿用上一次 motion 的历史缓存。
6. 保持现有评测语义：frame 0 初始化、完成 observation reset 后在初始暂停期间激活策略、动作播完保持末帧、保存完整 trajectory 和 root/body tracking 数据。
7. 给 `run_tracking_metrics_eval.py` 增加 `--env-dt`、`--sim-dt` 并透传；首版固定只运行 `seed=0`。

## 4. 计划修改清单

| 组件 | 文件/目录 | 变更 |
| --- | --- | --- |
| 数据工具源码 | 根目录 `any4hdmi/` | 作为主仓库普通目录直接跟踪，不保留独立 `.git`，不设 submodule |
| 服务器环境 | `active-adaptation/venv/mjlab` | RTX 4090 CUDA、Warp、MJLab 与最小 DDP 验证 |
| 训练资产 | `mimic-lite/mimic_lite/assets/mini3.py` | MJCF、URDF/USD、joint/body order、actuator、sensor、symmetry |
| 训练配置 | `mimic-lite/cfg/task/tracking-base-mini3.yaml` | Mini3 reward/termination/observation/action 配置 |
| 动作配置 | `mimic-lite/cfg/task/motion/mini3/` | any4hdmi manifest 与 motion source |
| 仿真后端 | `active-adaptation/.../mjlab/env.py` | 可配置 MuJoCo physics dt |
| 数据工具 | `any4hdmi/src/any4hdmi/scripts/preprocess/` | Mini3 flat PKL 到 NPZ qpos 转换与严格校验 |
| 策略导出 | `mimic-lite/scripts/play.py` | 输出完整 ONNX、观测和评测语义 YAML |
| 部署配置 | `sim2real/sim2real/config/robots/mini3.py` | Mini3 `RobotCfg` 与注册 |
| 集成仿真 | `sim2real/.../integrated_sim2sim.py`、`rl_policy/observations/common.py` | 去除 G1 tracking 假设、按 actuator target joint 映射、严格检查、对齐 history reset |
| 批量评测 | `sim2real/scripts/tracking_experiment/run_tracking_metrics_eval.py` | 透传 env/sim dt |
| 测试 | 各组件 `tests/` | schema、order、observation parity、单步推理、smoke sim |

首版 actuator 使用训练与 Sim2Sim 一致的线性 position-PD。速度限位在当前 MJLab/集成 Sim2Sim 链路中不自动执行，只做显式监测和报告；UFO 的非线性电机响应、KT 查表和并联踝映射列为后续 fidelity 阶段，不作为首版训练链路前置条件，也不能据此宣称已具备真机迁移能力。

## 5. 执行命令模板

服务器用一条 clone 命令取得主仓库和既有 submodule 中的全部代码；`any4hdmi` 已直接包含在主仓库中：

```bash
git clone --recurse-submodules <mimiclite-repository-url>
cd MimicLite
test -f any4hdmi/pyproject.toml
test -f mimic-lite/pyproject.toml
test -f active-adaptation/venv/mjlab/pyproject.toml
```

以下命令在 `any4hdmi/` 下执行。单文件默认打开 MuJoCo/mjviser，批量目录默认不打开可视化：

```bash
# 单文件转换并可视化
uv run any4hdmi-convert-mini3-pkl \
  --input-path /home/amax/Desktop/robot/UFO/humanoidverse/data/pkl/230210/example.pkl

# 批量转换可信来源的 Mini3 flat PKL；日期目录保持不变
uv run any4hdmi-convert-mini3-pkl \
  --input-path /home/amax/Desktop/robot/UFO/humanoidverse/data/pkl \
  --output-path output/mini3/sonic \
  --mjcf assets/robots/mini3_mjlab/mini3.xml \
  --target-fps 50 --root-quat-order xyzw

# 确认转换结果不会进入 Git；随后将该目录单独上传到服务器同一路径
git -C .. check-ignore any4hdmi/output/mini3/sonic/manifest.json
rsync -avP output/mini3/sonic/ <user>@<server>:<MimicLite-path>/any4hdmi/output/mini3/sonic/
```

以下命令在 `active-adaptation/` 下执行；新增 Hydra 配置名在实施后以 `--cfg job` 先验证。

```bash
# 服务器 RTX 4090 环境验收
test -f ../any4hdmi/pyproject.toml
test -f ../mimic-lite/pyproject.toml
uv --project venv/mjlab sync --locked
uv --project venv/mjlab run aa-discover-projects
uv --project venv/mjlab run aa-project enable mimic_lite
nvidia-smi
uv --project venv/mjlab run python - <<'PY'
import torch
import warp as wp
import mjlab

assert torch.cuda.is_available()
assert torch.cuda.device_count() > 0
for index in range(torch.cuda.device_count()):
    assert "4090" in torch.cuda.get_device_name(index)
    assert torch.cuda.get_device_capability(index) == (8, 9)
    value = torch.ones((32, 32), device=f"cuda:{index}")
    _ = value @ value
torch.cuda.synchronize()
wp.init()
print("RTX 4090 CUDA / Warp / MJLab preflight passed")
PY

# 大数据集第一次加载会构建本地缓存；next-window 保留在 CPU，避免把全量动作窗口放入显存
export ANY4HDMI_CACHE_BUILD_BATCH_SIZE=4096
export ANY4HDMI_CACHE_BUILD_LOADER_BATCH_SIZE=8
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS=4
export ANY4HDMI_CACHE_BUILD_PREFETCH_FACTOR=2
export ANY4HDMI_CACHE_BUILD_WRITE_BUFFER_BYTES=1073741824
export ANY4HDMI_NEXT_WINDOW_DEVICE=cpu

# 当前约 276 小时数据首次会生成全量 FK cache 和裁剪后的 FP16 window backing。
# 建议缓存所在磁盘至少预留 150 GiB；构建临时目录可能让峰值占用更高。

# 单环境启动检查
uv --project venv/mjlab run ../mimic-lite/scripts/train.py \
  task=tracking-base-mini3 task/motion=mini3/sonic \
  +exp=ppo/train algo/ppo/module=small backend=mjlab \
  task.num_envs=1 total_iters=1 wandb.mode=disabled seed=0

# 64 环境、10 iteration 训练冒烟
uv --project venv/mjlab run ../mimic-lite/scripts/train.py \
  task=tracking-base-mini3 task/motion=mini3/sonic \
  +exp=ppo/train algo/ppo/module=small backend=mjlab \
  task.num_envs=64 total_iters=10 wandb.mode=disabled seed=0

# 正式多卡训练（按服务器实际 GPU 数量调整 ID 列表）
bash scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 \
  ../mimic-lite/scripts/train.py venv/mjlab \
  task=tracking-base-mini3 task/motion=mini3/sonic \
  +exp=ppo/train algo/ppo/module=residual backend=mjlab seed=0

# 从同一训练配置导出完整 ONNX + YAML
uv --project venv/mjlab run ../mimic-lite/scripts/play.py \
  task=tracking-base-mini3 task/motion=mini3/sonic \
  +exp=ppo/train algo/ppo/module=residual backend=mjlab \
  checkpoint_path=/abs/path/checkpoint.pt \
  export_policy=true export_only=true headless=true task.num_envs=1 seed=0
```

以下命令在 `sim2real/` 下执行：

```bash
# 安装仅用于 Sim2Sim 的本地环境（any4hdmi 从根仓库本地路径安装）
uv sync --locked --group cpu

# ONNX 单步接口检查
uv run python scripts/test_policy_inference.py \
  --policy-config checkpoints/mimic-lite-mini3/policy.yaml \
  --single --inference-backend onnx-cpu

# 单动作 Sim2Sim
uv run python -m sim2real.sim_env.integrated_sim2sim \
  --robot mini3 \
  --policy-config checkpoints/mimic-lite-mini3/policy.yaml \
  --motion-path ../any4hdmi/output/mini3/sonic/motions/230210/jog_ff_stop_315_003__A179_M.npz \
  --env-dt 0.02 --sim-dt 0.002 \
  --headless --run-once --initial-pause-s 5 \
  --trajectory-output outputs/mini3_smoke/trajectory.npz --seed 0

# 批量评测
uv run python scripts/tracking_experiment/run_tracking_metrics_eval.py \
  --robot mini3 --env-dt 0.02 --sim-dt 0.002 \
  --motions-root ../any4hdmi/output/mini3/sonic \
  --policy mini3=checkpoints/mimic-lite-mini3/policy.yaml \
  --num-motions 8 --seeds 0 \
  --output-dir outputs/mini3_eval
```

## 6. 验收门槛与完成定义

| Gate | 必须满足 |
| --- | --- |
| Environment | 服务器全部训练 GPU 均识别为 RTX 4090/SM89；CUDA tensor、Warp、MJLab、单卡 1-env 和最小 DDP 测试通过 |
| Asset | MJCF 与 `usd_path` 资产可解析；模型加载后 `nq/nv/nu=28/27/21`；21 joint 与 actuator target 一一对应；有 collision geom 的非相邻自碰撞开启且相邻 pair 排除生效 |
| Motion | flat PKL 按固定 21 列契约转换为 NPZ；qpos 为 `T x 28`、50 Hz、有限值；qpos/joint names 严格一致；MuJoCo-aware 重采样、四元数和限位检查通过；frame 0 可稳定站立 |
| Train | 64-env smoke 无 NaN；checkpoint 可 play；实际 500 Hz physics/50 Hz policy；`feet_air_time` 只依赖左右 ankle-roll link；输出每关节最大/p99 速度及超限比例 |
| Export | 一个静态无 batch 的完整 ONNX；输入与 YAML 顺序一致；输出严格为 `[21]`；21 joint 的 default pose/Kp/Kd/action scale 完整且唯一 |
| Parity | 固定状态、reset 第一帧及连续运行至少 16 帧时，训练端/集成 Sim2Sim 端各 observation 分量均通过容差检查 |
| Sim2Sim | seed 0 从 frame 0 运行至末帧并保持；无缺失映射/越界/NaN；trajectory 与 metrics 文件完整 |

完成定义是：根仓库直接包含 any4hdmi 源码和 Mini3 资产，转换后的 `any4hdmi/output/` 数据不进入 Git并单独上传；服务器 RTX 4090 环境通过 preflight，至少一个 Mini3 flat PKL 动作能按固定 21 列契约转换为严格 NPZ contract，并依次通过数据校验、训练 smoke、checkpoint 导出、静态无 batch ONNX 单步推理、observation reset parity 和 seed 0 集成 Sim2Sim；正式训练策略在固定 seed 0 的批量动作集上得到可复现结果，且每次实验记录代码 commit、完整命令、Hydra 配置、seed、模型/数据哈希、速度限位报告和跟踪指标。未经许可证审查的 UFO 资产、未经测量确认的电机参数，以及真机部署均保留为显式未完成项。

## 7. 当前实施结果（2026-08-02）

- 已完成 Mini3 训练资产、21 关节严格模式、任务/动作配置、500 Hz MuJoCo physics、ankle-roll `feet_air_time` 和自碰撞配置。
- 已完成部署 YAML 扩展、静态无 batch ONNX 输出校验、Mini3 RobotCfg、按 actuator transmission target joint 映射、history reset 对齐和关节速度统计报告。
- 本地已通过 Hydra 正式 PPO 配置展开、Mini3 contract 单元测试，以及 MuJoCo `nq/nv/nu=28/27/21`、21 actuator target、23 个相邻碰撞排除和全部 `sim2real` 单元测试。
- 用户已在本地完成 Mini3 全量数据转换；本次代码实施没有重新启动转换，也没有启动训练。仍需在服务器 RTX 4090 上完成 1-env、64-env/10-iteration、最小 DDP、正式训练、真实 checkpoint 导出和策略驱动的 Sim2Sim。这些结果依赖训练 GPU 和生成的 checkpoint，不能由纯 CPU 测试替代。
