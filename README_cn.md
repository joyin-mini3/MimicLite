[中文版](README_cn.md) | [English](README.md)

# MimicLite

MimicLite 是一个高效、通用的人形机器人动作跟踪系统。本仓库是包含训练框架、
Mini3/G1 任务、动作转换工具和部署运行时的单体仓库；普通 `git clone` 即可取得全部
源码和 Mini3 机器人资产，不需要初始化 submodule。动作数据、checkpoint、W&B 运行
目录和缓存不进入 Git，需要单独保存或上传。

技术报告位于 [`mimic-lite.pdf`](mimic-lite.pdf)。

## 导航

- [仓库结构](#仓库结构)
- [Mini3 环境安装](#mini3-环境安装)
- [Mini3 动作数据](#mini3-动作数据)
- [Mini3 训练](#mini3-训练)
- [Mini3 Inference](#mini3-inference)
- [Pico 数据转换和 Inference](#pico-数据转换和-inference)
- [常见问题](#常见问题)

## 仓库结构

| 组件 | 路径 | 内容 |
| --- | --- | --- |
| MimicLite | [`mimic-lite/`](mimic-lite/) | Mini3/G1 任务、PPO、reward、observation、训练和导出脚本。 |
| 训练框架 | [`active-adaptation/`](active-adaptation/) | MJLab/IsaacLab backend、分布式启动器和通用环境基础设施。 |
| 动作工具 | [`any4hdmi/`](any4hdmi/) | PKL/Pico 转换、NPZ 数据格式、FK cache 和动作可视化。 |
| 部署运行时 | [`sim2real/`](sim2real/) | ONNX 推理、MuJoCo Sim2Sim、遥操作和机器人接口。 |
| Mini3 资产 | [`any4hdmi/assets/robots/mini3_mjlab/`](any4hdmi/assets/robots/mini3_mjlab/) | 训练、转换和可视化共用的 MJCF、URDF 与 mesh。 |

Mini3 当前契约为 21 个受控关节、50 Hz policy/reference、500 Hz MuJoCo physics。
训练默认使用 `task=tracking-base-mini3`、`backend=mjlab`、`seed=0`。

## Mini3 环境安装

### 系统要求

- Linux x86-64；
- NVIDIA 驱动可用，`nvidia-smi` 能看到训练 GPU；
- 训练环境固定使用 Python 3.12；
- RTX 4090 的 compute capability 是 `(8, 9)`，RTX 5090 是 `(12, 0)`；不要在
  检查脚本中把本地 5090 错误地断言成 SM89；
- 首次在线安装需要访问 PyPI、GitHub 和 NVIDIA Python index。

### 在线安装

安装 [`uv`](https://docs.astral.sh/uv/getting-started/installation/) 并克隆仓库：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone <MimicLite-repository-url>
cd MimicLite
```

根据仓库 [`uv.lock`](https://docs.astral.sh/uv/concepts/projects/sync/) 创建环境并注册
MimicLite project。`--project` 的相对路径以当前工作目录为基准；如果当前位于仓库
根目录，必须包含 `active-adaptation/` 前缀：

```bash
test -f active-adaptation/venv/mjlab/pyproject.toml

uv --project active-adaptation/venv/mjlab sync --locked
uv --project active-adaptation/venv/mjlab run aa-discover-projects
uv --project active-adaptation/venv/mjlab run aa-project enable mimic_lite
```

也可以先进入 `active-adaptation/`，再使用较短的 project 路径：

```bash
cd active-adaptation

uv --project venv/mjlab sync --locked
uv --project venv/mjlab run aa-discover-projects
uv --project venv/mjlab run aa-project enable mimic_lite
```

实际虚拟环境位于：

```text
active-adaptation/venv/mjlab/.venv
```

源码依赖以 editable 形式指向当前仓库中的 `active-adaptation/`、`mimic-lite/` 和
`any4hdmi/`，修改源码后一般不需要重新安装。不要在这个环境中随意执行 `pip install`
覆盖 lockfile 中的 Torch、MJLab 或 Warp 版本。

### CUDA/MJLab 验收

```bash
cd /path/to/MimicLite/active-adaptation

uv --project venv/mjlab run python - <<'PY'
import importlib.metadata as metadata

import active_adaptation as aa
aa.set_backend("mjlab")

import any4hdmi
import mimic_lite
import mujoco
import torch
import warp

assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
device = torch.device("cuda:0")
value = torch.ones((32, 32), device=device)
_ = value @ value
torch.cuda.synchronize()
warp.init()

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("mujoco:", mujoco.__version__)
print("mjlab:", metadata.version("mjlab"))
print("warp:", warp.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
print("MimicLite MJLab environment passed")
PY
```

`mimic_lite` 的普通 Python 导入也依赖 backend。若单独写诊断脚本，必须在导入
`mimic_lite` 前调用 `aa.set_backend("mjlab")`；训练和 play 入口会自动完成该设置。

### 服务器离线安装

不建议直接复制 `.venv`：其中可能包含旧机器的绝对路径、解释器链接和 JIT cache。
推荐把 `uv` 可执行文件与已经填充的 uv cache 打包，在相同架构的服务器上根据
`uv.lock` 重建环境。假设离线包为 `/data/wzk/mjlab_uv_offline.tar.zst`：

```bash
mkdir -p /data/wzk/mjlab_uv_offline_unpack
tar --zstd -xf /data/wzk/mjlab_uv_offline.tar.zst \
  -C /data/wzk/mjlab_uv_offline_unpack

OFFLINE_ROOT=/data/wzk/mjlab_uv_offline_unpack/mjlab_uv_offline
export PATH="$OFFLINE_ROOT:$PATH"
export UV_CACHE_DIR="$OFFLINE_ROOT/cache"
export UV_OFFLINE=1
export UV_PYTHON_DOWNLOADS=never

cd /data/wzk/MimicLite/active-adaptation
uv --project venv/mjlab sync --locked --offline
uv --project venv/mjlab run aa-discover-projects
uv --project venv/mjlab run aa-project enable mimic_lite
```

之后用上一节的验收脚本检查 GPU。离线 cache 必须提前包含 lockfile 所需的 Python、
wheel、Git dependency 和 source archive；缺少任何条目时 `UV_OFFLINE=1` 会直接报错，
不会偷偷访问网络。

## Mini3 动作数据

### 数据格式和路径

Mini3 训练文件为压缩 NPZ，每个文件只保存：

```text
qpos: float32 [T, 28]
```

28 维由 `base_link` 的 3 维位置、wxyz 四元数和严格顺序的 21 个关节组成。关节顺序
保存在数据集级 `manifest.json`，无需在每个 NPZ 中重复保存。默认数据路径：

```text
any4hdmi/output/mini3/sonic/   # 训练 PKL 转换结果
any4hdmi/output/mini3/pico/    # Pico clip 重定向结果
```

这两个目录均被 Git 忽略。

### 转换 Mini3 PKL 数据

PKL/joblib 反序列化可以执行代码，只转换可信来源的数据：

```bash
cd /path/to/MimicLite/active-adaptation

uv --project venv/mjlab run any4hdmi-convert-mini3-pkl \
  --input-path /path/to/trusted/mini3/pkl \
  --output-path ../any4hdmi/output/mini3/sonic \
  --mjcf ../any4hdmi/assets/robots/mini3_mjlab/mini3.xml \
  --target-fps 50 \
  --root-quat-order xyzw
```

目录输入会批量转换且默认不打开 viewer；单文件输入默认转换后可视化。已有输出默认
复用，重新生成时增加 `--overwrite`。

### 单独上传数据到服务器

```bash
rsync -avP \
  /path/to/MimicLite/any4hdmi/output/mini3/sonic/ \
  <user>@<server>:/path/to/MimicLite/any4hdmi/output/mini3/sonic/
```

服务器只需要 `git clone` 代码，动作数据按上述路径单独上传。首次训练会在
`active-adaptation/.cache/motion/` 构建 FK cache 和 FP16 backing；大数据集需要预留
充足磁盘空间，生成物也不应提交 Git。

## Mini3 训练

### 数据缓存设置

正式训练前建议设置：

```bash
export ANY4HDMI_CACHE_BUILD_BATCH_SIZE=4096
export ANY4HDMI_CACHE_BUILD_LOADER_BATCH_SIZE=8
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS=4
export ANY4HDMI_CACHE_BUILD_PREFETCH_FACTOR=2
export ANY4HDMI_CACHE_BUILD_WRITE_BUFFER_BYTES=1073741824
export ANY4HDMI_NEXT_WINDOW_DEVICE=cpu
```

如果 FK cache 已经构建完成，这些变量不会重新转换原始 NPZ。

### 单卡 smoke test

```bash
cd /path/to/MimicLite/active-adaptation

CUDA_VISIBLE_DEVICES=0 uv --project venv/mjlab run \
  ../mimic-lite/scripts/train.py \
  task=tracking-base-mini3 \
  task/motion=mini3/sonic \
  +exp=ppo/train \
  algo/ppo/module=residual \
  backend=mjlab \
  task.num_envs=64 \
  total_iters=10 \
  wandb.mode=disabled \
  seed=0
```

### 8 卡正式训练

先配置 W&B；不要把 token 写入脚本或提交 Git：

```bash
export WANDB_API_KEY='<your_wandb_api_key>'
export WANDB_ENTITY='<your_wandb_entity>'
```

`tracking-base-mini3` 默认在 actor 的完整和短 command observation 中包含 reference
`base_link` yaw-local linear velocity。下面使用 8 张 GPU、每卡 8192 个环境、训练
4000 iterations，并每 200 iterations 保留和上传一个 checkpoint：

```bash
cd /path/to/MimicLite/active-adaptation
mkdir -p ../logs

nohup bash scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 \
  ../mimic-lite/scripts/train.py \
  venv/mjlab \
  task=tracking-base-mini3 \
  task/motion=mini3/sonic \
  +exp=ppo/train \
  algo/ppo/module=residual \
  backend=mjlab \
  task.num_envs=8192 \
  total_iters=4000 \
  checkpoint_interval=200 \
  upload_interval=200 \
  wandb.name=Mini3_RootLinVel_Ckpt200 \
  seed=0 \
  > ../logs/mini3_root_linvel_ckpt200.log 2>&1 &
```

查看进度：

```bash
tail -f /path/to/MimicLite/logs/mini3_root_linvel_ckpt200.log
```

这里的 interval 单位是 PPO iteration，不是 physics step。必须同时设置
`checkpoint_interval=200` 和 `upload_interval=200`；如果只修改前者，中间保存会反复
覆盖 `checkpoint_temp.pt`。

### 关闭 actor 的 reference root velocity

在训练命令末尾增加下面两个 Hydra 删除项：

```bash
'~task.observation.command.ref_root_lin_vel_future_local' \
'~task.observation.command_short.ref_root_lin_vel_future_local'
```

例如把它们放在 `seed=0` 后面：

```bash
  seed=0 \
  '~task.observation.command.ref_root_lin_vel_future_local' \
  '~task.observation.command_short.ref_root_lin_vel_future_local'
```

训练和 inference 必须使用相同的 observation 结构，否则 checkpoint 会因 actor 输入
维度不一致而无法加载。robot 的线速度仍可保留在 critic/reward 路径中；上述 override
只删除 reference velocity command feature。

## Mini3 Inference

### Checkpoint 兼容性

运行前需要确认三项与训练一致：

1. `algo/ppo/module`，本项目 Mini3 正式命令使用 `residual`；
2. actor 是否包含 `ref_root_lin_vel_future_local`；
3. `task/motion` 和 motion filename 是否匹配数据集。

checkpoint 已保存 policy、VecNorm 统计量、环境状态和训练配置。不要为新动作重新估计
VecNorm；加载 checkpoint 时会恢复训练阶段的均值和标准差。

### Inference 一个 Sonic motion

```bash
cd /path/to/MimicLite/active-adaptation

CKPT='/absolute/path/to/checkpoint_4000.pt'
MOTION='230531/jog_ff_loop_180_R_003__A415_M.npz'

CUDA_VISIBLE_DEVICES=0 uv --project venv/mjlab run \
  ../mimic-lite/scripts/play.py \
  task=tracking-base-mini3 \
  task/motion=mini3/sonic \
  +exp=ppo/train \
  algo/ppo/module=residual \
  backend=mjlab \
  task.num_envs=1 \
  task.command.start_from_zero=true \
  checkpoint_path="$CKPT" \
  headless=false \
  render_seconds=0 \
  seed=0 \
  "+task.command.motion_cfgs.sonic.filenames=[$MOTION]"
```

如果 checkpoint 是“不带 root velocity”训练的，在 play 命令末尾同样增加：

```bash
'~task.observation.command.ref_root_lin_vel_future_local' \
'~task.observation.command_short.ref_root_lin_vel_future_local'
```

不要给带 root velocity 的 checkpoint 增加这两个删除项。

## Pico 数据转换和 Inference

### 支持的 Pico clip

转换器接受一个 `.npz`，或一个每个字段分别保存为 `.npy` 的解包目录。当前 v3 数据
至少需要 pelvis、双 ankle-roll 和双 wrist-yaw 的世界位置/四元数、FPS、坐标版本和
body frame；优先使用 `sonic_smpl_anchor_orientation` 恢复 Mini3 `base_link` 朝向。

Pico 端点不是 Mini3 关节角。转换器会：

1. 根据 pelvis-to-foot 长度估计 Mini3 比例；
2. 把脚部朝向变换到 root-relative，并右乘 G1→Mini3 link-frame offset；
3. 用 MuJoCo damped least-squares IK 恢复严格顺序的 21 个关节；
4. 投影关节限位并抑制相邻帧 IK 分支跳变；
5. 重采样为 50 Hz `qpos [T, 28]`。

详细格式和实现见 [`docs/mini3_pico_data.md`](docs/mini3_pico_data.md)。

### 转换

```bash
cd /path/to/MimicLite/active-adaptation

uv --project venv/mjlab run any4hdmi-convert-mini3-pico \
  --input-path ../pico_source_data/sample_clip_20260726_171741 \
  --overwrite \
  --no-viewer
```

输出位于：

```text
any4hdmi/output/mini3/pico/
├── manifest.json
└── motions/
    ├── sample_clip_20260726_171741.npz
    └── sample_clip_20260726_171741.json
```

JSON sidecar 记录缩放比例、端点映射和 IK 误差，不参与训练或 inference。原始
`root_vel_w` 不直接写入 NPZ；reference velocity 会从缩放、重采样后的 root qpos
按 50 Hz 计算，从而与实际 reference trajectory 一致。

### 先检查重定向结果

```bash
uv --project venv/mjlab run any4hdmi-view \
  --motion ../any4hdmi/output/mini3/pico/motions/sample_clip_20260726_171741.npz \
  --loop
```

viewer 默认在 `http://localhost:8080` 提供 mjviser 页面。只有 reference motion 在这里
显示正确后，才应测试 policy；否则 policy 表现无法区分“数据重定向错误”和“策略 OOD”。

### Inference Pico motion

```bash
cd /path/to/MimicLite/active-adaptation

CKPT='/absolute/path/to/checkpoint_4000.pt'

CUDA_VISIBLE_DEVICES=0 uv --project venv/mjlab run \
  ../mimic-lite/scripts/play.py \
  task=tracking-base-mini3 \
  task/motion=mini3/pico \
  +exp=ppo/train \
  algo/ppo/module=residual \
  backend=mjlab \
  task.num_envs=1 \
  task.command.start_from_zero=true \
  checkpoint_path="$CKPT" \
  headless=false \
  render_seconds=0 \
  seed=0 \
  '+task.command.motion_cfgs.pico.filenames=[sample_clip_20260726_171741.npz]'
```

不带 root velocity 的 checkpoint 仍需在命令最后增加两个 `~task.observation...`
删除项。一个只在 Sonic/PKL 数据上训练的策略可能无法稳定跟踪新的 Pico motion；这属于
策略分布外泛化问题，不代表 Pico NPZ 或 FK 接口错误。

## 常见问题

### `set_backend() must be called before get_backend()`

仅在手写 Python 导入检查中常见。在导入 `mimic_lite` 前执行：

```python
import active_adaptation as aa
aa.set_backend("mjlab")
```

### `Distribution not found at file:///.../any4hdmi`

当前仓库中的 `mimic-lite/pyproject.toml` 已将 `any4hdmi` 指向根仓库本地目录。确认完整
clone 了单体仓库，然后重新执行：

```bash
cd active-adaptation
uv --project venv/mjlab sync --locked
```

不要用旧 lockfile 或缺失的外部 `any4hdmi` checkout。

### `Project directory venv/mjlab does not exist`

`--project` 使用相对于当前工作目录的路径。如果当前位于仓库根目录：

```bash
cd /path/to/MimicLite
test -f active-adaptation/venv/mjlab/pyproject.toml
uv --project active-adaptation/venv/mjlab sync --locked
```

如果 `test` 失败，说明当前 checkout 不包含环境定义，需要检查拉取的分支和 commit：

```bash
git branch --show-current
git status
git log -1 --oneline
```

如果已经进入 `active-adaptation/`，才使用
`uv --project venv/mjlab ...`。不要在仓库根目录直接把 `venv/mjlab` 传给 uv。

### Checkpoint actor 输入维度不匹配

通常是训练和 play 对 root velocity observation 的开启状态不一致。带 root velocity
checkpoint 不加 `~...`；不带 root velocity checkpoint 同时增加完整和短 command 的
两个删除项。

### Pico 修改后仍看到旧动作

转换器默认复用已有 NPZ。代码或重定向参数变化后必须用 `--overwrite`。FK cache key
会根据 NPZ/sidecar 的文件指纹自动变化并构建新 cache；只有手工复制文件时同时保留了
相同大小和时间戳等指纹，才需要手动清理对应旧 cache。

### 数据、日志和 checkpoint 在哪里

- 转换数据：`any4hdmi/output/`；
- FK/runtime cache：`active-adaptation/.cache/motion/`；
- 本文 nohup 日志：根目录 `logs/`；
- checkpoint：以训练日志打印的 `Final checkpoint:` 或 `Latest checkpoint:` 为准；
- W&B 本地运行目录：由 `WANDB_DIR` 或 W&B 默认路径决定。

上述目录不应提交 Git。

## 已发布 Checkpoint

上游发布了 3 个训练 4,000 iterations 的 PPO 策略：

| 策略 | Actor hidden dimensions | 并行环境 | Checkpoint | 训练时间 |
| --- | --- | ---: | --- | ---: |
| MimicLite-Huge | `[1024, 1024, 1024]` | `32 × 8192` | [`xua2csee`](https://wandb.ai/elijahgalahad/mimic_lite/runs/xua2csee) | 3 小时 30 分钟 |
| MimicLite-Base | `[256, 256, 256]` | `8 × 8192` | [`iij0q0b5`](https://wandb.ai/elijahgalahad/mimic_lite/runs/iij0q0b5) | 2 小时 57 分钟 |
| MimicLite-Small | `[128, 128, 128]` | `4 × 8192` | [`zb9e19ih`](https://wandb.ai/elijahgalahad/mimic_lite/runs/zb9e19ih) | 3 小时 00 分钟 |

这些是上游 G1 发布策略，不是本仓库训练得到的 Mini3 checkpoint。Mini3 checkpoint 的
网络类型和 observation 契约以各自保存的训练配置为准。

## 更多文档

- [`docs/mini3_training_pipeline.md`](docs/mini3_training_pipeline.md)：Mini3 reference、actor/critic、action 和 reward 数据流；
- [`docs/mini3_train_sim2sim_plan.md`](docs/mini3_train_sim2sim_plan.md)：Mini3 实施与 Sim2Sim 方案；
- [`docs/mini3_pico_data.md`](docs/mini3_pico_data.md)：Pico clip 格式和 IK 重定向细节；
- [`any4hdmi/README.md`](any4hdmi/README.md)：数据工具完整命令；
- [`sim2real/README.md`](sim2real/README.md)：部署运行时。

## 许可证

本集成仓库采用 GPL-3.0-or-later。纳入仓库的组件目录保留各自上游许可证；重新分发
前需要分别确认机器人资产、数据集和组件许可证。
