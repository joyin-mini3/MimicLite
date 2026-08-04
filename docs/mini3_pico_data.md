# Mini3 Pico reference motion 接入

## 数据流

当前 Pico clip 不是 Mini3 关节角。它提供的是 pelvis、双踝、双腕的世界位姿，
以及 Sonic SMPL 姿态。接入流程为：

```text
Pico clip (.npz 或解包后的 .npy 目录)
  -> Sonic 6D anchor 恢复 root 四元数
  -> pelvis-to-foot 比例估计
  -> MuJoCo DLS IK 恢复 Mini3 21 个关节角
  -> 50 Hz qpos [T, 28]
  -> any4hdmi/output/mini3/pico
  -> task/motion=mini3/pico
```

双腕分别映射到 Mini3 左右小臂末端虚拟点。由于 Mini3 没有 wrist link，手部只约束
位置，不约束腕部朝向。双脚同时约束位置和朝向。IK 会投影到 Mini3 XML 关节范围，
并默认把相邻 source frame 的单关节变化限制为 0.12 rad，避免稀疏 IK 分支跳变；
这些约束只用于生成 reference motion，不会对策略 action 增加限幅。

脚部朝向先转换为 source root-relative，再消除 G1 与 Mini3 link frame 的首帧固定
偏置，最后乘回当前 Mini3 `base_link` 朝向。不能直接在世界坐标中用首帧脚姿态对齐，
否则当 reference root 带有非零 yaw 时，hip-yaw 会产生大小相反的补偿角。

## 转换

在 `any4hdmi` 目录执行：

```bash
uv run any4hdmi-convert-mini3-pico \
  --input-path ../pico_source_data/sample_clip_20260726_171741 \
  --no-viewer
```

去掉 `--no-viewer` 会在转换后打开 MuJoCo viewer。重新生成已有输出时增加
`--overwrite`。转换结果为：

```text
any4hdmi/output/mini3/pico/
  manifest.json
  motions/sample_clip_20260726_171741.npz
  motions/sample_clip_20260726_171741.json
```

NPZ 仍然只包含 `qpos`；JSON sidecar 保存缩放比例、目标映射和 IK 误差，不参与训练
或推理加载。Pico 原始 `root_vel_w` 不直接写入 NPZ：any4hdmi 会用缩放、重采样后的
root qpos 在 50 Hz 下计算 reference root velocity，因此 reward 和 actor 的
`ref_root_lin_vel_future_local` 与实际加载的 Mini3 reference trajectory 保持一致。

## 作为 reference motion 测试

旧的 `pjvzcfe7/checkpoint_3000.pt` 使用 residual actor，且不含 root velocity
observation，因此命令为：

```bash
cd active-adaptation

CUDA_VISIBLE_DEVICES=0 uv --project venv/mjlab run \
  ../mimic-lite/scripts/play.py \
  task=tracking-base-mini3 \
  task/motion=mini3/pico \
  +exp=ppo/train \
  algo/ppo/module=residual \
  backend=mjlab \
  task.num_envs=1 \
  task.command.start_from_zero=true \
  seed=0 \
  checkpoint_path=/absolute/path/to/checkpoint_3000.pt \
  headless=false \
  render_seconds=0 \
  '~task.observation.command.ref_root_lin_vel_future_local' \
  '~task.observation.command_short.ref_root_lin_vel_future_local' \
  '+task.command.motion_cfgs.pico.filenames=[sample_clip_20260726_171741.npz]'
```

使用带 root velocity 输入的新 checkpoint 时，删除上面两个 `~task.observation...`
override，并确保模型类型与训练时一致。
