# Mini3 Checkpoint 全量 Reference Motion 过滤方案

> 状态：核心实现已完成，待 GPU smoke / 全量运行  
> 更新时间：2026-08-04  
> 目标 checkpoint：`wandb/wandb/run-20260803_140412-pjvzcfe7/files/checkpoint_3000.pt`  
> 目标数据集：`any4hdmi/output/mini3/sonic`

## 1. 目标与结论

本方案使用指定的 Mini3 `checkpoint_3000.pt` (使用residual模式训练的) 对 Sonic 数据集中的每一条原始
reference motion 做一次从开头到结尾的闭环 MJLab rollout，并完全按照该次训练
保存的 termination 配置判断能否完成，做筛选的时候不需要可视化界面：

- rollout 在任一 tracking-error termination 触发前到达 `motion_timeout`，记为
  `passed`；
- 任一 tracking-error termination 先触发，或与 `motion_timeout` 同帧触发，记为
  `failed`；
- 数据加载、checkpoint 加载、NaN、仿真异常等记为 `runtime_error`，不得混入
  `failed`。

最终必须至少生成 `failed_motions.txt`，每行是一条相对
`any4hdmi/output/mini3/sonic/motions` 的 NPZ 路径，供后续逐条可视化复查。同时保存
失败帧、termination 原因和当时的跟踪误差，方便按原因抽样，而不是只得到一个无上下文
的黑名单。

这里的“可以完成”严格表示：**指定 checkpoint 在固定的 nominal-v1 仿真条件下可以
走完整条 motion**。它不代表 Mini3 机械上绝对可实现，也不代表其他 checkpoint 或真机
一定可以完成。

## 2. 已确认的输入契约

### 2.1 Checkpoint 与训练快照

目标文件已经存在，大小约 20 MiB。过滤结果的 manifest 必须记录以下指纹：

```text
checkpoint_3000.pt sha256:
78d6c387b764c6b426d171b833c300484a925c3a9efbb00698a37a7a8d08c1e7

cfg.yaml sha256:
a02e73700453a1968955a0c0756eb65e65a485a315947aeb386c993ab2ae2923

训练代码 commit:
4a0046d7acda5842829dcff6205b9b76b50875d9
```

W&B 保存的训练参数包括：

- `task=tracking-base-mini3`；
- `task/motion=mini3/sonic`；
- `algo/ppo/module=residual`；
- `backend=mjlab`；
- `seed=0`；
- 50 Hz policy/reference、500 Hz MuJoCo physics；
- Mini3 21 关节 strict contract 与 real-motor actuator。

过滤器必须以同一 W&B run 中的 `files/cfg.yaml` 为配置真值来源，不能只用当前仓库的
Hydra 默认配置。该 checkpoint 的 Actor command 中没有后来加入的
`ref_root_lin_vel_future_local`；按保存配置推导，Actor 输入应为 `policy[399] +
command[240] = 639`，Critic 输入为 1113。当前配置直接加载该 checkpoint 可能产生
663/1137 与 639/1113 的输入宽度不匹配。

实现时应先加载保存的完整 `cfg.yaml`，再叠加一份只包含评测行为的 overlay，并在 rollout
前验证 observation key、展开顺序、宽度、21 关节顺序和 checkpoint state dict shape。

### 2.2 数据规模与全长要求

`conversion_index.json` 当前包含：

| 项目 | 数值 |
| --- | ---: |
| motion 数量 | 135,097 |
| 总 reference 帧数 | 49,673,682 |
| 50 Hz reference 时长 | 约 275.96 小时 |
| motion 长度中位数 | 287 帧 |
| P90 / P95 / P99 | 635 / 793 / 1,538 帧 |
| 最大长度 | 9,007 帧 |
| 长于 512 帧 | 22,098 条 |
| 长于 1,000 帧 | 3,797 条 |

训练配置使用 `full_motion: false`，当前 `WindowedMotionDataset` 每个运行窗口固定为
512 帧。因此，直接复用训练 sampler 或现有 `scripts/eval.py` 只能评估随机窗口，不能
满足“走完整条原始 reference motion”的定义。不能把 512 帧窗口完成误报为完整 motion
完成，也不能在窗口边界 reset policy/history 后再拼接结果。

## 3. 唯一判定规则

### 3.1 复用保存的 termination 配置

过滤器不得重新实现一套近似阈值，而应直接读取并运行 W&B `cfg.yaml` 中实例化的
termination components：

| termination | body | 阈值 | 连续帧数 | 判定 |
| --- | --- | ---: | ---: | --- |
| `motion_timeout` | - | motion 结束 | 1 | 成功候选 |
| `root_pos_error` | `base_link` | 0.4 m | 25 | 失败 |
| `root_ori_error` | `base_link` | 1.2 rad | 25 | 失败 |
| `body_pos_error` | tracking bodies local error | 0.4 m | 5 | 失败 |
| `body_ori_error` | tracking bodies local error | 1.2 rad | 5 | 失败 |

tracking bodies 使用保存配置解析后的 Mini3 body 列表：`base_link`、
`waist_yaw_link`、左右 hip-yaw、knee-pitch、ankle-roll、shoulder-yaw 和
elbow-pitch links。

必须读取 `stats/termination/<name>`，不能用 `terminated` 与 `truncated` 区分成败。
原因是 `root_pos_error` 在训练配置中也标记为 `is_timeout: true`，但对本过滤器而言它仍是
跟踪失败。现有 `scripts/eval.py` 把 `root_pos_error` 加入 `success_rate` 的逻辑不得复用。

首个 `done` 帧的分类伪代码如下：

```python
tracking_failures = {
    "root_pos_error",
    "root_ori_error",
    "body_pos_error",
    "body_ori_error",
}

if any(term[name] for name in tracking_failures):
    status = "failed"
elif term["motion_timeout"]:
    status = "passed"
else:
    status = "runtime_error"
```

如果 tracking failure 和 `motion_timeout` 同帧为真，采用保守规则记为 `failed`。

### 3.2 完整 motion 的时间定义

为匹配现有 `RobotTracking.start_from_zero` 行为，nominal-v1 在 reference `t=1` 初始化
机器人，并保持 policy observation/action history 的正常 reset 语义。之后连续运行，不在
512 帧边界重置机器人、action history、observation history 或 actuator state，直到：

1. 某个 tracking failure 首次触发；或
2. `motion_timeout` 触发。

固定的 `max_episode_length=1000` 不能截断长 motion。评测循环应以每条 motion 的真实
`motion_len` 为上限，并设置 `motion_len + 1` 的 watchdog；watchdog 到期却没有任何已知
termination 时记为 `runtime_error`。

进度只作为诊断指标，不参与最终二分类：

```text
progress = clamp(termination_t / (motion_len - 1), 0, 1)
```

## 4. nominal-v1 评测条件

首版过滤的目标是隔离“reference + checkpoint + nominal Mini3 仿真”是否可跟踪，而不是
估计随机扰动下的成功概率。每条 motion 必须在与调度顺序、env ID 和并行度无关的固定
条件下得到相同结果：

- policy 使用 deterministic action，即 PPO 分布的确定性输出；
- `terrain=plane`；
- `step_dt=0.02`、`mujoco_physics_dt=0.002`；
- 保留 Mini3 nominal real-motor actuator、并联踝映射、电流环、T-N 和 KT 模型；
- reference 初始 root/joint pose 和 velocity 不加噪声；
- observation noise 全部设为 0；
- 禁用 material/mass/COM/motor domain randomization、joint offset 和 interval root push；
- `rewind_prob=0`；
- action delay 固定为训练区间中值 2 physics substeps：`min_delay=max_delay=2`；
- action smoothing 固定为训练区间中值：`alpha=0.9`；
- 一个 motion 只评测一次，首版不通过多 seed 投票。

将 delay 与 alpha 固定为中值是为了避免同一 motion 因分配到不同 env、随机数消费顺序或
断点续跑位置不同而改变分类。所有 overlay 值都必须写入输出 manifest。后续如果需要
“训练随机化下的鲁棒可完成率”，应新增 `robust-v1` profile，多 seed 统计成功率，不能
修改 nominal-v1 的含义或覆盖其结果。

## 5. 设计架构

### 5.1 新的入口脚本

已新增：

```text
mimic-lite/scripts/filter_reference_motions.py
```

职责：

1. 加载 W&B `cfg.yaml` 与 checkpoint；
2. 应用并记录 nominal-v1 overlay；
3. 构造完整且稳定的 motion manifest；
4. 驱动向量化 full-motion rollout；
5. 捕获每条 motion 的首个 termination；
6. 原子写出分片结果并支持断点续跑；
7. 合并、校验并生成 passed/failed/runtime-error 名单。

建议接口：

```bash
uv --project active-adaptation/venv/mjlab run \
  mimic-lite/scripts/filter_reference_motions.py \
  --checkpoint-path wandb/wandb/run-20260803_140412-pjvzcfe7/files/checkpoint_3000.pt \
  --run-cfg-path wandb/wandb/run-20260803_140412-pjvzcfe7/files/cfg.yaml \
  --motions-root any4hdmi/output/mini3/sonic \
  --output-dir outputs/mini3_motion_filter/checkpoint_3000_nominal_v1 \
  --num-envs 512 \
  --window-frames 512 \
  --resume
```

`--num-envs` 是性能参数，不得影响结果。首轮先用 `--limit 32` 做 smoke，再逐步尝试
512、2048、4096 env；不能未经显存与吞吐实测就默认 8192。脚本默认值为 512，实际可用值
应以显存 smoke 为准。

### 5.2 全长顺序流式 dataset

建议在 any4hdmi 中新增 evaluator 专用的顺序流式 runtime，例如：

```text
SequentialWindowedMotionDataset
```

其核心契约：

- FK cache 和 FP16 motion fields 常驻 CPU；
- GPU 只保留每个活跃 env 的 current/next 固定长度窗口；
- scheduler 显式指定 `env_id -> source_motion_id`，禁止随机抽 motion 或随机 start；
- 对外暴露原始 `source_motion_id`、`motion_path` 和真实 `motion_len`；
- `t` 始终是原始 motion 的绝对帧号；
- v1 在请求越过窗口边界时同步装载同一 motion 的下一窗口；异步预取作为后续吞吐优化，
  不改变判定语义；
- 窗口切换时不 reset 仿真、policy history 或 actuator state；
- 切换窗口必须保留 future/history 所需 overlap。当前 checkpoint 的 reference steps 为
  `[-8, -4, -2, 0, 1, 2, 3, 4]`，因此至少保留 8 帧左上下文和 4 帧右上下文；
- 最后一帧可以为 observation future step 做边界 clamp，但 `motion_timeout` 必须基于原始
  `motion_len`，不能基于 512 帧 runtime window。

不要简单把 `RUNTIME_MOTION_MAX_LEN` 从 512 改成 9007。4096/8192 env 乘以最大 motion
长度会造成不可接受的 GPU 内存占用，而且大多数 motion 远短于最大值。

### 5.3 Motion scheduler

使用持久 env pool 异步消费确定性 motion 队列：

1. motion manifest 按相对路径排序，并赋予稳定的 `dataset_motion_id`；
2. 首批 motion 分配给 env；
3. 每个 env 独立运行到自己的首个 `done`；
4. 先记录结果，再显式给该 env 分配队列中的下一条 motion，并只 reset 该 env；
5. 所有 135,097 个 ID 都完成后停止。

实现会在每次 step 前快照 `env_id -> source_motion_id`，因此可以安全使用
`step_and_maybe_reset`：done 帧证据从返回的 TensorDict 读取，新 motion 由 reset 中的顺序
dataset 分配，旧身份由快照恢复。最终输出按 `dataset_motion_id` 排序，不能按异步完成顺序
排序。

长 motion 会造成 env 间负载不均。首版可按长度桶调度，但桶只影响执行顺序，不得改变
`dataset_motion_id`、初始条件或结果。任意并行度和 resume 后都应得到逐条一致的结果。

### 5.4 首次失败证据

每条 motion 只记录首个 done。对于 `failed`，至少保存：

- `termination_reason`；
- 所有同帧为真的 termination flags；
- `termination_t`、`motion_len` 和 `progress`；
- root/body position/orientation error 的当前最大值；
- body error 最大值对应的 body name；
- 当前 robot/reference root pose；
- 当前 joint position error 的最大值和 joint name；
- policy action、applied action 和 applied torque 的最大绝对值；
- checkpoint/config/profile/dataset 指纹。

首轮不保存所有 rollout 的逐帧 trajectory，避免失败数量较大时产生海量磁盘数据。人工选中
某条失败 motion 后再用单条可视化工具重跑，并按需保存 trajectory 或视频。

## 6. 输出格式

建议目录结构：

```text
outputs/mini3_motion_filter/checkpoint_3000_nominal_v1/
  manifest.json
  summary.json
  results.jsonl
  passed_motions.txt
  failed_motions.txt
  runtime_errors.txt
  failure_by_reason/
    root_pos_error.txt
    root_ori_error.txt
    body_pos_error.txt
    body_ori_error.txt
  shards/
    shard_000000.jsonl
    shard_000001.jsonl
```

`failed_motions.txt` 示例：

```text
230323/stairs_climbing_down_loop_R_001__A300.npz
230323/stairs_climbing_up_start_R_001__A300.npz
```

名单只包含确实触发 tracking termination 的 motion。`runtime_errors.txt` 必须独立，避免
把软件错误误当成 Mini3 能力边界。

`results.jsonl` 每行建议采用以下 schema：

```json
{
  "schema_version": 1,
  "dataset_motion_id": 12345,
  "relative_path": "230323/stairs_climbing_down_loop_R_001__A300.npz",
  "motion_len": 308,
  "status": "failed",
  "termination_reason": "root_pos_error",
  "termination_flags": {
    "motion_timeout": false,
    "root_pos_error": true,
    "root_ori_error": false,
    "body_pos_error": false,
    "body_ori_error": false
  },
  "termination_t": 87,
  "progress": 0.2834,
  "max_errors": {
    "root_pos_m": 0.51,
    "root_ori_rad": 0.24,
    "body_pos_m": 0.33,
    "body_ori_rad": 0.41
  },
  "max_body_pos_error_name": "left_ankle_roll_link"
}
```

`manifest.json` 还必须记录：完整命令、checkpoint/config SHA256、代码 commit、dataset
manifest/conversion-index SHA256、profile overlay、GPU/backend、num-envs、window size、开始和
结束时间、已完成数量及结果 schema version。

## 7. 断点续跑与多 GPU

全量 reference 约 4967 万 policy 帧，必须从一开始支持断点续跑：

- 每个 shard 先写 `*.tmp`，flush/fsync 后原子 rename；
- resume 时按 `dataset_motion_id` 读取已完成集合，只调度缺失 ID；
- 单条结果的唯一键是
  `(dataset_fingerprint, checkpoint_fingerprint, profile, dataset_motion_id)`；
- 同一唯一键出现内容不一致时立即报错，禁止静默覆盖；
- 合并阶段验证结果数恰好为 135,097、ID 连续、路径与初始 manifest 一一对应；
- 只有所有 shard 合并和校验通过后，才生成最终 `failed_motions.txt`。

多 GPU 不需要让一个仿真 env 做 DDP。更简单可靠的方式是按稳定 motion ID 区间启动相互
独立的 evaluator 进程，每个 GPU 写不同 shard 目录，最后由单独的 merge 命令校验合并。
分片规则和 GPU 数量只能影响吞吐，不能参与随机种子或 nominal 参数计算。

## 8. 人工可视化复查

已新增单条复现入口：

```text
mimic-lite/scripts/inspect_reference_motion.py
```

它读取过滤输出的 manifest 和一条 result，自动复用完全相同的 checkpoint、保存配置和
nominal-v1 overlay，然后：

- 使用 `num_envs=1` 和该单条 NPZ；
- 使用与批量过滤相同的顺序流式 runtime，只分配该一条 motion；
- 从 `t=1` 正常运行，保留 policy/history/actuator 连续状态；
- viewer 同时显示 robot 与 reference ghost；
- 到首次 done 时停止在自动 reset 之前，并打印 replay 与批量记录是否一致；
- 可用 `--hold-at-end` 保持最终帧，便于在 Viser viewer 中人工检查。

建议调用方式：

```bash
uv --project active-adaptation/venv/mjlab run \
  mimic-lite/scripts/inspect_reference_motion.py \
  --filter-output-dir outputs/mini3_motion_filter/checkpoint_3000_nominal_v1 \
  --motion 230323/stairs_climbing_down_loop_R_001__A300.npz \
  --hold-at-end
```

可视化复现的首次 termination reason 和 frame 必须与批量过滤结果一致；不一致时应视为
过滤器缺陷，而不是人工忽略。

## 9. 验证计划

### 9.1 单元测试

1. **全长窗口连续性**：构造超过 512 帧的合成 motion，在窗口边界前后将流式 dataset
   输出与 `FullMotionDataset` 逐字段比较；覆盖 `[-8, ..., +4]` reference steps。
2. **无隐式 reset**：跨窗口后 robot、policy history、previous actions 和 real-motor state
   不被清零。
3. **确定性分配**：不同 num-envs、不同完成顺序和 resume 后，每个 motion 的初始状态及
   profile 参数完全一致。
4. **termination 分类**：分别构造 timeout-only、四种 tracking failure、timeout 与 failure
   同帧、无已知 termination watchdog，验证三态分类。
5. **root position 特例**：验证 `root_pos_error` 即使属于 timeout/truncated 仍记为 failed。
6. **结果合并**：重复 ID、路径漂移、缺失 shard、哈希变化和不完整临时文件必须失败。
7. **checkpoint 契约**：当前 663 维配置加载 639 维 checkpoint 时必须在 rollout 前给出
   清晰错误；使用保存配置时正常加载。

### 9.2 集成 smoke

先构造不少于 32 条的固定 smoke list，包含：

- 已知普通 walk/jog；
- 上下楼梯；
- 短于 512 帧；
- 跨一次和多次 512 帧边界；
- 长于 1000 帧；
- 数据集最大长度附近的 motion。

同一 smoke list 连续运行两次，除时间和设备统计外，逐 motion 结果应完全一致。再选择若干
motion 使用单条 resident full-motion runtime 交叉验证，确保流式窗口不改变 termination
帧。

### 9.3 全量验收

全量运行完成需同时满足：

- `passed + failed + runtime_error == 135097`；
- 每条原始 NPZ 恰好对应一条结果；
- runtime error 为 0，或被显式修复并重跑，不能当 failed 接受；
- resume 前后的结果一致；
- 随机抽取每种 failure reason 至少 10 条人工可视化，批量与单条复现的 reason/frame
  一致；
- 至少抽取 10 条 passed motion 可视化确认确实运行到 motion end；
- 最终名单、summary、manifest 和结果文件完整且可由脚本重新生成。

## 10. 实施步骤与建议改动范围

按以下顺序实施：

1. 增加保存配置加载、nominal-v1 overlay 和 checkpoint/observation contract 校验；
2. 实现 evaluator 专用的全长顺序流式 dataset，并完成与 full resident dataset 的边界测试；
3. 为 `RobotTracking` 增加显式 eval motion assignment 接口，不改变默认训练 sampler；
4. 实现异步 env scheduler、首次 termination 捕获和三态分类；
5. 实现原子 shard、resume、merge 和名单输出；
6. 实现单条 viewer 复现工具；
7. 运行 32-motion smoke、边界 motion 集和重复性测试；
8. 单 GPU 小规模性能测试后确定 num-envs，再扩展到全量或多 GPU 分片；
9. 全量完成后人工检查 failure 分层样本，确认楼梯、地形不匹配、快速动作、低姿态等实际
   失败模式。

预计主要涉及：

```text
any4hdmi/src/any4hdmi/dataset/        # evaluator 顺序流式 runtime
mimic-lite/mimic_lite/tasks/command.py # 显式 eval motion assignment
mimic-lite/scripts/                    # filter / merge / inspect 入口
mimic-lite/tests/                      # termination、窗口、resume 回归测试
docs/                                  # 使用说明与结果解释
```

训练默认路径、随机 sampler 和现有数据文件不得因本过滤器改变行为。

## 11. 当前实现与运行命令

当前代码包括：

- `SequentialWindowedMotionDataset`：CPU backing + 固定 GPU window，窗口切换不 reset；
- `filter_reference_motions.py`：确定性 rollout、首个 done 分类、原子 shard、resume、merge；
- `inspect_reference_motion.py`：按 motion ID 或相对路径精确复现，校验输入文件 SHA256；
- `tracking_filter_diagnostics`：记录失败帧误差、最大误差 body/joint、reference/robot root pose；
- 单元测试：窗口边界、runtime factory、分类优先级、nominal overlay、shard/名单合并。

推荐先跑 32 条 smoke：

```bash
uv --project active-adaptation/venv/mjlab run \
  mimic-lite/scripts/filter_reference_motions.py \
  --checkpoint-path wandb/wandb/run-20260803_140412-pjvzcfe7/files/checkpoint_3000.pt \
  --run-cfg-path wandb/wandb/run-20260803_140412-pjvzcfe7/files/cfg.yaml \
  --motions-root any4hdmi/output/mini3/sonic \
  --output-dir outputs/mini3_motion_filter/checkpoint_3000_nominal_v1_smoke \
  --num-envs 32 \
  --window-frames 512 \
  --limit 32
```

确认显存与结果后，将 `--output-dir` 换为正式目录、移除 `--limit` 并加 `--resume`。多 GPU
时每个进程使用相同的 `--num-shards N` 和不同的 `--shard-index 0..N-1`；所有 worker 完成
后执行：

```bash
uv --project active-adaptation/venv/mjlab run \
  mimic-lite/scripts/filter_reference_motions.py \
  --checkpoint-path wandb/wandb/run-20260803_140412-pjvzcfe7/files/checkpoint_3000.pt \
  --run-cfg-path wandb/wandb/run-20260803_140412-pjvzcfe7/files/cfg.yaml \
  --motions-root any4hdmi/output/mini3/sonic \
  --output-dir outputs/mini3_motion_filter/checkpoint_3000_nominal_v1 \
  --merge-only
```

## 12. 结果解释边界

- `failed` 是 `checkpoint_3000.pt` 的能力边界，不是 reference 永久不可学。
- nominal-v1 禁用训练随机化，因此结果不是鲁棒成功率；如需真机筛选，应另做 robust profile。
- 当前任务 terrain 是 plane，上下楼梯等依赖非平地接触的动作失败是预期结果。
- checkpoint 训练时使用 512 帧 window；全长评测保持状态连续，但超过训练窗口长度的行为
  仍可能暴露长时漂移，这正是本过滤器需要捕获的失败。
- 失败名单只服务人工复查和后续数据决策；在完成人工抽检前，不自动删除、移动或覆盖原始
  motion 文件。
