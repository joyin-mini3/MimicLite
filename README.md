[中文版](README_cn.md) | [English](README.md)

# MimicLite

MimicLite is an efficient, general humanoid motion-tracking system. This is a
self-contained monorepo with the training framework, Mini3/G1 tasks, motion
conversion tools, and deployment runtime. A normal `git clone` obtains all
source code and Mini3 robot assets; no submodule initialization is required.
Motion data, checkpoints, W&B runs, and generated caches are intentionally kept
out of Git and must be stored or uploaded separately.

The technical report is available at [`mimic-lite.pdf`](mimic-lite.pdf).

## Contents

- [Repository layout](#repository-layout)
- [Mini3 environment setup](#mini3-environment-setup)
- [Mini3 motion data](#mini3-motion-data)
- [Mini3 training](#mini3-training)
- [Mini3 inference](#mini3-inference)
- [Pico conversion and inference](#pico-conversion-and-inference)
- [Troubleshooting](#troubleshooting)

## Repository layout

| Component | Path | Contents |
| --- | --- | --- |
| MimicLite | [`mimic-lite/`](mimic-lite/) | Mini3/G1 tasks, PPO, rewards, observations, training, and export scripts. |
| Training framework | [`active-adaptation/`](active-adaptation/) | MJLab/IsaacLab backends, distributed launchers, and shared environment infrastructure. |
| Motion toolkit | [`any4hdmi/`](any4hdmi/) | PKL/Pico conversion, the NPZ format, FK caching, and motion visualization. |
| Deployment runtime | [`sim2real/`](sim2real/) | ONNX inference, MuJoCo sim2sim, teleoperation, and robot interfaces. |
| Mini3 assets | [`any4hdmi/assets/robots/mini3_mjlab/`](any4hdmi/assets/robots/mini3_mjlab/) | MJCF, URDF, and meshes shared by training, conversion, and visualization. |

The current Mini3 contract has 21 controlled joints, a 50 Hz policy/reference
rate, and 500 Hz MuJoCo physics. Training uses `task=tracking-base-mini3`,
`backend=mjlab`, and `seed=0` unless explicitly overridden.

## Mini3 environment setup

### Requirements

- Linux x86-64;
- a working NVIDIA driver, with all intended GPUs visible in `nvidia-smi`;
- Python 3.12, as pinned by the MJLab environment;
- RTX 4090 reports compute capability `(8, 9)` and RTX 5090 reports `(12, 0)`;
  do not incorrectly assert SM89 on a local 5090;
- the first online installation needs access to PyPI, GitHub, and the NVIDIA
  Python index.

### Online installation

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and clone
the repository:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone <MimicLite-repository-url>
cd MimicLite
```

Create the environment from [`uv.lock`](https://docs.astral.sh/uv/concepts/projects/sync/)
and enable the MimicLite project. A relative `--project` path is resolved from
the current working directory. From the repository root, include the
`active-adaptation/` prefix:

```bash
test -f active-adaptation/venv/mjlab/pyproject.toml

uv --project active-adaptation/venv/mjlab sync --locked
uv --project active-adaptation/venv/mjlab run aa-discover-projects
uv --project active-adaptation/venv/mjlab run aa-project enable mimic_lite
```

Alternatively, enter `active-adaptation/` before using the shorter project
path:

```bash
cd active-adaptation

uv --project venv/mjlab sync --locked
uv --project venv/mjlab run aa-discover-projects
uv --project venv/mjlab run aa-project enable mimic_lite
```

The actual environment is created at:

```text
active-adaptation/venv/mjlab/.venv
```

The environment installs `active-adaptation/`, `mimic-lite/`, and `any4hdmi/`
as editable local sources. Source edits are therefore visible without a
reinstall. Avoid manually replacing the locked Torch, MJLab, or Warp versions
with `pip install` inside this environment.

### CUDA/MJLab preflight

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

A standalone Python import of `mimic_lite` also requires a selected backend.
Call `aa.set_backend("mjlab")` before importing `mimic_lite` in diagnostic
scripts. The training and play entry points select it automatically.

### Offline server installation

Do not copy `.venv` directly: it can contain absolute paths, interpreter links,
and machine-specific JIT caches. Package the `uv` executable and a populated uv
cache, then reconstruct the environment from `uv.lock` on a server with the
same OS and architecture. For an archive at
`/data/wzk/mjlab_uv_offline.tar.zst`:

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

Run the preflight above afterward. The offline cache must already contain the
locked Python interpreter, wheels, Git dependencies, and source archives. A
missing object causes an explicit offline error rather than a network request.

## Mini3 motion data

### Format and paths

Each Mini3 motion is a compressed NPZ containing only:

```text
qpos: float32 [T, 28]
```

The 28 values are the `base_link` position, a wxyz quaternion, and 21 joints in
the strict Mini3 order. Joint names are stored once in the dataset-level
`manifest.json`, not duplicated in every NPZ. Default dataset roots are:

```text
any4hdmi/output/mini3/sonic/   # converted training PKLs
any4hdmi/output/mini3/pico/    # retargeted Pico clips
```

Both roots are Git-ignored.

### Convert Mini3 PKLs

PKL/joblib deserialization can execute arbitrary code. Convert trusted inputs
only:

```bash
cd /path/to/MimicLite/active-adaptation

uv --project venv/mjlab run any4hdmi-convert-mini3-pkl \
  --input-path /path/to/trusted/mini3/pkl \
  --output-path ../any4hdmi/output/mini3/sonic \
  --mjcf ../any4hdmi/assets/robots/mini3_mjlab/mini3.xml \
  --target-fps 50 \
  --root-quat-order xyzw
```

A directory input is converted in batch without a viewer. A single file opens
the viewer by default. Existing outputs are reused; pass `--overwrite` to
rebuild them.

### Upload data separately

```bash
rsync -avP \
  /path/to/MimicLite/any4hdmi/output/mini3/sonic/ \
  <user>@<server>:/path/to/MimicLite/any4hdmi/output/mini3/sonic/
```

The server can clone source code from Git while motion data is uploaded to the
same relative path separately. The first training load builds FK caches and an
FP16 backing under `active-adaptation/.cache/motion/`; large datasets require
substantial free disk space, and these generated files must not enter Git.

## Mini3 training

### Data cache settings

Recommended settings for a full run:

```bash
export ANY4HDMI_CACHE_BUILD_BATCH_SIZE=4096
export ANY4HDMI_CACHE_BUILD_LOADER_BATCH_SIZE=8
export ANY4HDMI_CACHE_BUILD_NUM_WORKERS=4
export ANY4HDMI_CACHE_BUILD_PREFETCH_FACTOR=2
export ANY4HDMI_CACHE_BUILD_WRITE_BUFFER_BYTES=1073741824
export ANY4HDMI_NEXT_WINDOW_DEVICE=cpu
```

These settings do not rebuild an already valid FK cache.

### Single-GPU smoke test

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

### Eight-GPU training

Configure W&B without writing tokens into scripts or Git:

```bash
export WANDB_API_KEY='<your_wandb_api_key>'
export WANDB_ENTITY='<your_wandb_entity>'
```

`tracking-base-mini3` includes reference `base_link` linear velocity in the
full and short actor command observations by default. The command below uses
eight GPUs, 8192 environments per GPU, 4000 PPO iterations, and retains and
uploads a checkpoint every 200 iterations:

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

Monitor the log with:

```bash
tail -f /path/to/MimicLite/logs/mini3_root_linvel_ckpt200.log
```

Intervals are PPO iterations, not physics steps. Set both
`checkpoint_interval=200` and `upload_interval=200`. Changing only the former
repeatedly overwrites `checkpoint_temp.pt` instead of preserving every version.

### Disable reference root velocity in the actor

Append both Hydra deletions to the training command:

```bash
'~task.observation.command.ref_root_lin_vel_future_local' \
'~task.observation.command_short.ref_root_lin_vel_future_local'
```

For example, place them after `seed=0`:

```bash
  seed=0 \
  '~task.observation.command.ref_root_lin_vel_future_local' \
  '~task.observation.command_short.ref_root_lin_vel_future_local'
```

Training and inference must use the same observation structure, or the actor
input shape will not match the checkpoint. The robot velocity can remain in
critic and reward paths; these overrides delete only the reference velocity
command feature.

## Mini3 inference

### Checkpoint compatibility

Keep the following identical to training:

1. `algo/ppo/module` (`residual` in the commands in this README);
2. whether `ref_root_lin_vel_future_local` is present in the actor;
3. `task/motion` and the dataset-relative motion filename.

Checkpoints contain the policy, VecNorm statistics, environment state, and
training config. Do not estimate new VecNorm statistics for each motion; the
training mean and standard deviation are restored from the checkpoint.

### Run one Sonic motion

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

For a checkpoint trained without root velocity, append both observation
deletions:

```bash
'~task.observation.command.ref_root_lin_vel_future_local' \
'~task.observation.command_short.ref_root_lin_vel_future_local'
```

Do not add them for a checkpoint that was trained with root velocity.

## Pico conversion and inference

### Supported Pico clips

The converter accepts a `.npz` or an unpacked directory with one `.npy` file
per field. The current v3 contract requires world poses for the pelvis, two
ankle-roll links, and two wrist-yaw links, plus FPS, coordinate version, and
body-frame metadata. `sonic_smpl_anchor_orientation` is preferred for the
Mini3 `base_link` orientation.

Pico endpoints are not Mini3 joint angles. The converter:

1. estimates scale from pelvis-to-foot length;
2. expresses feet in the source-root frame and right-multiplies the fixed
   G1-to-Mini3 link-frame offset;
3. solves the strict 21-joint pose with MuJoCo damped least-squares IK;
4. projects joint limits and suppresses temporal IK branch jumps;
5. resamples the output to 50 Hz `qpos [T, 28]`.

See [`docs/mini3_pico_data.md`](docs/mini3_pico_data.md) for the complete format
and implementation notes.

### Convert a clip

```bash
cd /path/to/MimicLite/active-adaptation

uv --project venv/mjlab run any4hdmi-convert-mini3-pico \
  --input-path ../pico_source_data/sample_clip_20260726_171741 \
  --overwrite \
  --no-viewer
```

Output layout:

```text
any4hdmi/output/mini3/pico/
├── manifest.json
└── motions/
    ├── sample_clip_20260726_171741.npz
    └── sample_clip_20260726_171741.json
```

The JSON sidecar records scale, endpoint mapping, and IK errors, but is not
loaded by training or inference. The raw `root_vel_w` is not stored in the NPZ;
reference velocity is derived at 50 Hz from the scaled and resampled root qpos,
so it matches the trajectory actually consumed by the policy.

### Inspect retargeting before policy inference

```bash
uv --project venv/mjlab run any4hdmi-view \
  --motion ../any4hdmi/output/mini3/pico/motions/sample_clip_20260726_171741.npz \
  --loop
```

The viewer serves an mjviser page at `http://localhost:8080`. Verify the raw
reference first; otherwise policy playback cannot distinguish a retargeting
error from policy out-of-distribution behavior.

### Run Pico inference

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

A checkpoint trained without root velocity still needs the two
`~task.observation...` deletions. A policy trained only on Sonic/PKL motions may
not stably track a new Pico clip; that is an out-of-distribution generalization
issue and does not by itself indicate an NPZ or FK integration error.

## Troubleshooting

### `set_backend() must be called before get_backend()`

This usually occurs only in handwritten import checks. Select the backend
before importing `mimic_lite`:

```python
import active_adaptation as aa
aa.set_backend("mjlab")
```

### `Distribution not found at file:///.../any4hdmi`

The current `mimic-lite/pyproject.toml` points `any4hdmi` to the local monorepo
directory. Confirm that the complete monorepo was cloned, then run:

```bash
cd active-adaptation
uv --project venv/mjlab sync --locked
```

Do not use a stale lockfile or a missing external `any4hdmi` checkout.

### `Project directory venv/mjlab does not exist`

`--project` is relative to the current working directory. From the repository
root, run:

```bash
cd /path/to/MimicLite
test -f active-adaptation/venv/mjlab/pyproject.toml
uv --project active-adaptation/venv/mjlab sync --locked
```

If the `test` fails, the checkout does not contain the environment definition.
Inspect the branch and commit that were pulled:

```bash
git branch --show-current
git status
git log -1 --oneline
```

Use `uv --project venv/mjlab ...` only after entering `active-adaptation/`.
Do not pass that shorter path while still at the repository root.

### Checkpoint actor input mismatch

The usual cause is a different root-velocity observation setting between
training and play. Use no `~...` overrides for a root-velocity checkpoint; add
both full and short command deletions for a no-root-velocity checkpoint.

### Pico output did not change after a retargeting update

Existing NPZ files are reused by default. Pass `--overwrite` after changing
code or retargeting parameters. The FK cache key fingerprints the NPZ and its
sidecar, so a normal rewrite builds a new cache automatically. Manual cleanup
is only needed if a hand-copied file preserves an otherwise identical size and
timestamp fingerprint.

### Data, logs, and checkpoints

- converted data: `any4hdmi/output/`;
- FK/runtime cache: `active-adaptation/.cache/motion/`;
- nohup logs from this README: root `logs/`;
- checkpoints: use the `Final checkpoint:` or `Latest checkpoint:` path printed
  by the training log;
- local W&B run directory: controlled by `WANDB_DIR` or W&B defaults.

None of these generated directories should be committed to Git.

## Released checkpoints

The upstream project released three G1 PPO policies trained for 4,000
iterations:

| Policy | Actor hidden dimensions | Parallel environments | Checkpoint | Wall-clock time |
| --- | --- | ---: | --- | ---: |
| MimicLite-Huge | `[1024, 1024, 1024]` | `32 × 8192` | [`xua2csee`](https://wandb.ai/elijahgalahad/mimic_lite/runs/xua2csee) | 3 h 30 min |
| MimicLite-Base | `[256, 256, 256]` | `8 × 8192` | [`iij0q0b5`](https://wandb.ai/elijahgalahad/mimic_lite/runs/iij0q0b5) | 2 h 57 min |
| MimicLite-Small | `[128, 128, 128]` | `4 × 8192` | [`zb9e19ih`](https://wandb.ai/elijahgalahad/mimic_lite/runs/zb9e19ih) | 3 h 00 min |

These are upstream G1 policies, not Mini3 checkpoints trained by this
repository. Each Mini3 checkpoint must be used with its saved model and
observation contract.

## Additional documentation

- [`docs/mini3_training_pipeline.md`](docs/mini3_training_pipeline.md): Mini3 reference, actor/critic, action, and reward data flow;
- [`docs/mini3_train_sim2sim_plan.md`](docs/mini3_train_sim2sim_plan.md): Mini3 implementation and sim2sim plan;
- [`docs/mini3_pico_data.md`](docs/mini3_pico_data.md): Pico format and IK retargeting details;
- [`any4hdmi/README.md`](any4hdmi/README.md): complete data-tool commands;
- [`sim2real/README.md`](sim2real/README.md): deployment runtime.

## License

This integration repository is released under GPL-3.0-or-later. Vendored
components retain their upstream licenses; verify robot-asset, dataset, and
component licenses before redistribution.
