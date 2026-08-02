# any4hdmi

`any4hdmi` defines one simple `qpos`-based motion format for HDMI-related datasets.

- one dataset-level `manifest.json`
- one `motions/**/*.npz` file per motion
- each motion file stores only `qpos`
- each manifest stores an MJCF reference, either hosted or local to the repository

The motion plus the dataset timestep from `manifest.json` is enough to replay the clip in MuJoCo.

## Datasets

Clone the source datasets from Hugging Face:

```bash
git clone https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset data/LAFAN1_Retargeting_Dataset
git clone https://huggingface.co/datasets/bones-studio/seed data/seed

tar xzf data/seed/g1.tar -C data/seed/g1
```

## Commands

Convert LAFAN:

```bash
uv run any4hdmi-convert-lafan \
  --csv-dir data/LAFAN1_Retargeting_Dataset/g1 \
  --out-dir output/lafan
```

Convert SONIC:

```bash
uv run any4hdmi-convert-sonic \
  --csv-dir data/seed/g1/csv \
  --out-dir output/sonic
```

Convert one trusted UFO Mini3 PKL and open the MuJoCo/mjviser visualization:

```bash
uv run any4hdmi-convert-mini3-pkl \
  --input-path /home/amax/Desktop/robot/UFO/humanoidverse/data/pkl/230210/jog_ff_stop_225_003__A179_M.pkl
```

Convert the complete Mini3 PKL tree without visualization:

```bash
uv run any4hdmi-convert-mini3-pkl \
  --input-path /home/amax/Desktop/robot/UFO/humanoidverse/data/pkl
```

The default output is `output/mini3/sonic`: source subdirectories are retained,
so `pkl/230210/example.pkl` becomes
`output/mini3/sonic/motions/230210/example.npz`. Each NPZ contains only
`qpos`. Existing files are reused; pass `--overwrite` to rebuild them. Directory
conversion is headless by default, while a single file opens the viewer by
default (`--no-viewer` disables it).

The converter loads joblib/PKL data, which can execute arbitrary code. Only run
it on trusted inputs. The copied Mini3 asset and its license notice live under
`assets/robots/mini3_mjlab/`.

Convert 100STYLE from the Axellwppr `MotionDataset` tarball:

```bash
uv run any4hdmi-convert-axellwppr \
  --input /home/elijah/Downloads/100style.tar \
  --out-dir output/100style
```

Override the MJCF reference if needed:

```bash
uv run any4hdmi-convert-sonic \
  --csv-dir data/seed/g1/csv \
  --out-dir output/sonic \
  --mjcf-repo elijahgalahad/g1_xmls \
  --mjcf-path g1-mode_13_15.xml \
  --mjcf-revision main
```

Replay a converted motion:

```bash
uv run any4hdmi-view --motion output/lafan/motions/dance1_subject2.npz
```

Runtime loading also accepts a hosted dataset root:

```python
load_any4hdmi_dataset(
  root_path="hf://elijahgalahad/any4hdmi-lafan",
  target_fps=50,
  base_dir=Path.cwd(),
  num_envs=1,
  full_motion=True,
  shard=False,
)
```

Upload a converted dataset folder to Hugging Face:

```bash
uv run any4hdmi-upload output/lafan elijahgalahad/any4hdmi-lafan
```

Headless check:

```bash
uv run any4hdmi-view \
  --motion output/sonic/motions/230322/reach_jump_R_001__A299_M.npz \
  --headless
```

Pipeline details live in [docs/pipeline.md](docs/pipeline.md).
Dataset format details live in [docs/dataset_format.md](docs/dataset_format.md).
