[中文版](README_cn.md) | [English](README.md)

# MimicLite

MimicLite is an efficient, general humanoid motion-tracking system that can train a deployable policy in 3 hours on 8 RTX 4090 GPUs while retaining competitive tracking quality. Under a matched MuJoCo evaluation, MimicLite improves global root tracking over SONIC while achieving comparable local tracking accuracy. The same policy supports low-latency Pico-driven teleoperation and highly dynamic motion tracking on a physical Unitree G1.

The technical report is available at [`mimic-lite.pdf`](mimic-lite.pdf).

## Project Components

This repository is a self-contained monorepo. Training, task, conversion, and deployment sources are tracked directly in the directories below; the upstream links are retained for provenance only. A normal `git clone` obtains all source code without submodule initialization:

| Component | Path / upstream | Contents |
| --- | --- | --- |
| MimicLite | [`mimic-lite/`](mimic-lite/) ([upstream](https://github.com/EGalahad/mimic-lite)) | Training, evaluation, policy export, task configs, and learning code. |
| Training framework | [`active-adaptation/`](active-adaptation/) ([upstream](https://github.com/Agent-3154/active-adaptation)) | Simulation backends, distributed launchers, environments, and shared infrastructure. |
| Motion data toolkit | [`any4hdmi/`](any4hdmi/) | Motion conversion, validation, visualization, and dataset tooling bundled in this repository. |
| Deployment runtime | [`sim2real/`](sim2real/) ([upstream](https://github.com/EGalahad/sim2real)) | ONNX inference, MuJoCo sim2sim, Pico teleoperation, and Unitree G1 deployment. |

## Released Checkpoints

The released checkpoint set contains three PPO policies trained for 4,000 iterations. The wall-clock column reports the 4,000-update training time on RTX 4090 GPUs. The tracking panels below evaluate the released checkpoints listed in the table.

| Policy | Actor hidden dimensions | Parallel environments | Checkpoint | Wall-clock time |
| --- | --- | ---: | --- | ---: |
| MimicLite-Huge | `[1024, 1024, 1024]` | `32 × 8192` | [`xua2csee`](https://wandb.ai/elijahgalahad/mimic_lite/runs/xua2csee) | 3 h 30 min |
| MimicLite-Base | `[256, 256, 256]` | `8 × 8192` | [`iij0q0b5`](https://wandb.ai/elijahgalahad/mimic_lite/runs/iij0q0b5) | 2 h 57 min |
| MimicLite-Small | `[128, 128, 128]` | `4 × 8192` | [`zb9e19ih`](https://wandb.ai/elijahgalahad/mimic_lite/runs/zb9e19ih) | 3 h 00 min |

Training-time sources: Huge [`55ie49o5`](https://wandb.ai/elijahgalahad/mimic_lite/runs/55ie49o5), Base [`07k900hl`](https://wandb.ai/elijahgalahad/mimic_lite/runs/07k900hl), and Small [`akq50h1n`](https://wandb.ai/elijahgalahad/mimic_lite/runs/akq50h1n).

![Unified cross-codebase tracking evaluation](assets/mimiclite_vs_sonic_readme.png)

Compared with SONIC, MimicLite retains more progress on dynamic LAFAN motions and improves global root tracking while maintaining comparable local tracking accuracy.

For a fair comparison, we report the motion-lookahead latency required by each
policy, defined by its furthest required future-reference frame. All values use
the shared 50 Hz reference-motion contract.

| Policy | MimicLite | BFM-Zero | SONIC release | SONIC low-latency | HoloMotion | TeleopIT | Humanoid-GPT | HEFT | TWIST2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Motion-lookahead latency | 0.08 s | 0.12 s | 0.90 s | 0.18 s | 0.20 s | 0.00 s | 0.02 s | 0.12 s | 0.00 s |

## Training Data

Released training datasets are collected in the [`any4hdmi` Hugging Face collection](https://huggingface.co/collections/elijahgalahad/any4hdmi). The [`BONES-SEED` dataset](https://huggingface.co/datasets/bones-studio/seed) is the exception: to respect its license and redistribution terms, users obtain it from the original source. This repository bundles the conversion scripts and processing tools under [`any4hdmi/`](any4hdmi/).

### Local Mini3 dataset

Convert Mini3 PKLs locally from `any4hdmi/`:

```bash
uv run any4hdmi-convert-mini3-pkl \
  --input-path /home/amax/Desktop/robot/UFO/humanoidverse/data/pkl
```

Converted files are written to `any4hdmi/output/mini3/sonic/`. That directory
is intentionally Git-ignored: upload it separately to the same path in the
server checkout. Source code and the Mini3 MJCF/meshes are part of this main
repository; converted motions are not.

## Mini3 Training and Sim2Sim

The 21-joint Mini3 training task, strict export contract, and MuJoCo Sim2Sim
path are integrated. See the [implementation and command guide](docs/mini3_train_sim2sim_plan.md#5-执行命令模板)
for RTX 4090 setup, training smoke tests, PPO training, static ONNX export, and
seed-0 evaluation. Training reads `any4hdmi/output/mini3/sonic/` directly and
does not download motion data from Git or a third-party repository.

## Deployment Support

The [`sim2real/`](sim2real/) runtime provides a modular observation interface that separates policy-specific input construction from the shared deployment runtime. Integrating a policy requires only an observation class and a YAML specification; the inference, simulator, and robot interfaces remain unchanged. This common path supports integrated MuJoCo evaluation and real-robot execution for MimicLite, HEFT, TeleopIT, Humanoid-GPT, BFM-Zero, SONIC, and TWIST2. Policy inference is decoupled from robot I/O through interchangeable MuJoCo and physical Unitree G1 backends.

## License

This integration repository is released under GPL-3.0-or-later. Vendored component directories retain their upstream license files; verify dataset and component licenses before redistribution.
