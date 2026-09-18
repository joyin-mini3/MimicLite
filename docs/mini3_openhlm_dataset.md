# Mini3 randomized demonstrations for OpenHLM

`record_mini3_openhlm.py` records complete randomized pickup, walking, and basket-placement episodes using the **LeRobot v2.1** layout of the [official OpenHLM dataset](https://huggingface.co/datasets/OpenHLM/OpenHLM-data). Mini3 has its own **36-dimensional** state/action contract; it does not use the official G1 34-dimensional joint interface.

OpenHLM now also includes the native `mini3_pick_carry` configuration. See the [native Mini3 training guide](../../OpenHLM/src/openpi4OpenHLM/docs/mini3_training.md) for commands and environment requirements. This document covers collection and the adapters in MimicLite.

## Collection and files

Run from the repository root:

```bash
active-adaptation/venv/mjlab/.venv/bin/python record_mini3_openhlm.py collect \
  --episodes 200 --workers 4 --render-workers 6 \
  --output outputs/mini3_openhlm_recording
```

Add `--resume` to continue an existing batch. Resumption checks the collection configuration and SHA-256 of six core scripts. Existing exports must also match their source attempt, global index, and Parquet checksum. Concurrency may change. Ensure the previous coordinator and its children have exited before resuming; do not start two coordinators against the same output. Partially interrupted raw attempts remain for diagnosis and are replaced by later seeds.

Record one raw attempt:

```bash
active-adaptation/venv/mjlab/.venv/bin/python mini3_collect_episode.py \
  --seed 42 --output outputs/mini3_single_record --onnx-threads 1
```

```text
outputs/mini3_openhlm_recording/
  collection_config.json          # Seed, count, quality limits
  collection_status.json          # accepted/exported/complete and attempts
  source_hashes.json              # Core script versions
  asset_provenance.json           # Model, reference, scene provenance
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

The training dataset contains exactly 200 admitted episodes. Failed attempts, contact-quality rejections, and unused parallel prefetched attempts remain under `attempts` and are excluded from those 200. All runtime data stays in the Git-ignored `outputs` directory.

Collection retains the existing scene randomization: independent cube XY offsets within ±1.2 cm, yaw within ±12°, and uniform red/green/blue target selection. A base seed and attempt index produce reproducible episode seeds. Colors retain separate placement regions. Tables, basket, background, and lighting receive no additional randomization; these demonstrations do not establish generalization to new layouts.

## Admission and synchronization

Accepted episodes must finish with the cube settled in the basket, preserve all original policy outputs, contain valid numerical data, and fully execute every 20 ms action through ten 2 ms physics steps. Existing real motors and collision monitoring remain active. Maximum unexpected penetration must be no greater than **0.5 mm**, stricter than the controller's 2 mm stop threshold; acceptance does not mean zero contact.

Images, state, and action use the native **50 Hz** control rate. The official example uses 30 Hz; this exporter avoids resampling the control signal. It captures qpos/qvel and newly computed commands before executing `[t,t+0.02)`, then saves the actual successor in `next_*`. Images are rendered afterward from those exact pre-action states without stepping physics or rerunning the controller. RGB and state therefore describe the same observation, while actions are the following control targets.

## LeRobot columns

| Column | Type and meaning |
| --- | --- |
| `head_image_left` | Head `head_rgb`, pitched downward by 45° |
| `left_wrist_image` | Left gripper `left_gripper_rgb` |
| `right_wrist_image` | Right gripper `right_gripper_rgb` |
| `state` | float32[36], pre-action proprioceptive state |
| `actions` | float32[36], Mini3 hierarchical reference described below |
| `timestamp` | float32, starts at zero per episode with 0.02 s spacing |
| `frame_index` | int64, consecutive index within the episode |
| `episode_index` | int64, 0 through 199 |
| `index` | int64, consecutive index across the dataset |
| `task_index` | int64, 0=red, 1=green, 2=blue |

All images are RGB 224×224, encoded as JPEG quality=95 with subsampling=0, embedded directly in Parquet as Hugging Face `Image` features. No external PNG/MP4 files are needed; `total_videos=0` is intentional. Rendering uses 320×240 to retain the camera aspect ratio before resizing to 224×224. Shadows, reflections, and offscreen multisampling are disabled for throughput. Camera mounts and collision models are unchanged. The stowed left arm's camera mostly sees the floor and gripper; all views retain their actual contents.

Task text is exactly one of:

```text
Please put the red square from the tabletop into the basket.
Please put the green square from the tabletop into the basket.
Please put the blue square from the tabletop into the basket.
```

## Mini3's 36-dimensional contract

Exact names and units are stored in `meta/mini3_schema.json` and `mini3_vla_features.py`.

| Zero-based channels | Contents |
| --- | --- |
| 0–6 | Left shoulder pitch/roll/yaw, original elbow pitch, added elbow yaw, wrist roll/pitch |
| 7 | Left gripper closedness |
| 8–14 | Same seven right-arm joints |
| 15 | Right gripper closedness |
| 16–21, 22–27 | Left/right leg: hip pitch/roll/yaw, knee pitch, ankle pitch/roll |
| 28 | Waist yaw; no fabricated extra G1 waist axes |
| 29–31 | Root roll, pitch, and yaw angular velocity |
| 32–34 | Root linear XYZ velocity in the current measured-yaw heading frame, retaining world-vertical Z |
| 35 | Root height, world Z |

Joint/orientation units are rad, angular velocity rad/s, linear velocity m/s, and height m. Gripper closedness is `clip(1-half_opening/0.035,0,1)`: zero fully open, one fully closed.

`state` uses measured joint/root quantities. Original 21-axis `actions` come from the current MimicLite joint reference; added-axis actions use actual planned motor targets including integral correction; grippers use commanded openings. Root pose, height, and velocity come from gated references, including closed-loop velocity correction. Yaw velocity uses the wrapped angle difference to the next reference frame.

This compact vector is a learning target for a new Mini3 configuration, not a lossless encoding of MimicLite's complete historical/future inputs. Deployment still requires an implemented and tested Mini3 reference decoder, timing logic, and state-estimation interface; the G1 deployment program cannot be reused unchanged. Raw sidecars additionally preserve complete future references, 264-dimensional policy command inputs, 399-dimensional proprioceptive history inputs, original q/dq/effort/kp/kd, exact added-joint targets, and feedforward efforts so the mapping can be revisited.

The teacher still uses true MuJoCo object poses for planning and grasp decisions. Dataset `state` excludes cube and basket positions, but root height and velocity are ideal simulated proprioceptive quantities requiring corresponding real-world estimators. RGB has no overlaid target boxes, phase labels, or privileged coordinates. Demonstration collection does not establish a deployed real-world visual control loop.

## OpenHLM validation, statistics, and training

When normalization statistics exist, validation also checks normalized values and their round trip back to the original state and action values.

`mini3_openhlm_training.py` registers `mini3_pick_carry` at runtime without modifying the neighboring OpenHLM checkout. It uses the actual OpenHLM loader with 36-dimensional transforms and 50-frame action chunks (1 s), avoiding G1's 34-channel output truncation and delta-joint rules.

Check every episode's first, middle, and final samples, language, decoded cameras, and episode-boundary padding:

```bash
active-adaptation/venv/mjlab/.venv/bin/python mini3_openhlm_training.py validate \
  --report outputs/mini3_openhlm_recording/openhlm_validation.json
```

Compare every numeric row against raw synchronized records, verify all Parquet checksums, and generate full-data training statistics:

```bash
active-adaptation/venv/mjlab/.venv/bin/python validate_mini3_openhlm_dataset.py \
  --norm-output outputs/mini3_openhlm_recording/openpi_assets/mini3_pick_carry/local/mini3_pick_carry
```

This uses OpenHLM's `RunningStats` and serializer over every state and every 50-frame action window, with repeated-final-action padding. Computing numerical moments does not require decoding RGB again. The original OpenHLM normalization script is also available through:

```bash
active-adaptation/venv/mjlab/.venv/bin/python mini3_openhlm_training.py norm \
  --workers 2 --batch-size 16 -- --max-frames 10000
```

That command samples frames and overwrites statistics at the same location. Avoid unintentionally replacing full-data statistics; use `--artifacts-root outputs/another_norm_run` for isolated experiments.

Actual training needs a full OpenHLM training environment configured following its README and a π0.5 base checkpoint:

```bash
python mini3_openhlm_training.py train \
  --weights /path/to/pi05_base_pytorch --workers 4 --batch-size 16 \
  -- --exp-name mini3_pick_carry_v1
```

Replace the checkpoint placeholder with a real directory containing `model.safetensors`. This collection task does not download large models or start VLA training. Loader checks cover format, action chunks, and image preprocessing, not tokenizer execution, model loading, or training convergence.

Use **`HF_LEROBOT_HOME`**. OpenHLM's pinned LeRobot dependency rejects the old `LEROBOT_HOME` variable; the launcher sets the correct one. Extra CPU-validation packages are isolated under `.cache/mini3_vla_deps`, without replacing the simulation environment's torch/numpy. This overlay is not a complete GPU training environment.

The default metadata `train` split includes all 200 episodes. `recommended_split.json` separately suggests a whole-episode 180/20 split, which the current launcher does not automatically enable. For evaluation, create train/validation subsets from those lists and compute statistics only on the training subset, avoiding leakage from splitting neighboring frames.

Final counts, colors, file sizes, and row audits are recorded in `collection_status.json` and `integrity_validation.json`; actual OpenHLM loading results are in `openhlm_validation.json`.

## Completed collection (2026-09-18)

The local dataset is `outputs/mini3_openhlm_recording/lerobot/local/mini3_pick_carry`. Collection and validation are complete.

| Item | Measured result |
| --- | --- |
| Complete successful episodes / unique scene seeds | 200 / 200 |
| Target colors | red 50, green 80, blue 70 |
| Total frames / frequency | 239,861 / 50 Hz |
| Total simulated duration | 4,797.22 s, approximately 79.95 min |
| Episode duration | 23.76–24.66 s |
| Embedded RGB images across three cameras | 719,583 at 224×224 |
| Total Parquet size | 11,986,174,938 bytes, approximately 11.16 GiB |
| Maximum unexpected penetration in accepted episodes | 0.4987 mm, below the 0.5 mm limit |

The collector screened 213 attempts in order: 200 accepted, nine rejected for contact quality, and four failed tasks. Three additional prefetched raw attempts were not screened, giving 216 directories in `attempts`. Color counts retain the distribution after random sampling and quality filtering; they were not forced to be balanced.

Full validation passed:

- SHA-256 checks passed for all 200 Parquet files. Every state/action row matches its pre-action raw record, with continuous indices and timestamps.
- The independent audit decoded 1,800 sampled images. Export had already decoded every JPEG to compute image statistics.
- The actual local OpenHLM loader checked the first, middle, and final sample of every episode: 600 samples covering all cameras, prompts, 50-frame action chunks, and terminal padding. It also checked normalization and inverse normalization on the same 600 samples.
- Full-data normalization statistics are saved at `outputs/mini3_openhlm_recording/openpi_assets/mini3_pick_carry/local/mini3_pick_carry/norm_stats.json`.
- All 157 relevant regression tests passed. Logs are `mini3_tests.log` and `recording_validation_tests.log`; seven tests were repeated in the latter log.

The format and Mini3 training-input interface have been verified. Model weights, training execution, and convergence have not been tested; deployment still requires the Mini3 reference decoder described above. `openhlm_runtime_provenance.json` records the OpenHLM commit and key source checksums used for these checks.
