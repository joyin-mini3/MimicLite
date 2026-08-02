# Repository Guidelines

## Project Structure & Module Organization

This monorepo tracks all project components directly. `active-adaptation/` provides simulation backends and distributed training infrastructure; `mimic-lite/` contains task code, learning algorithms, Hydra YAML under `cfg/`, and training/evaluation scripts; `sim2real/` contains deployment code, robot configuration, utilities, scripts, and `tests/`; and `any4hdmi/` contains motion conversion tooling. Root-level `README.md`, `README_cn.md`, `assets/`, and `mimic-lite.pdf` form the public project landing page. Documentation sources live in `sim2real/docs/`; large checkpoints and runtime artifacts must remain outside Git.

## Build, Test, and Development Commands

- `git clone <repository-url>`: clone the complete source tree; no submodule initialization is required.
- `cd sim2real && uv sync`: create the Python 3.10 deployment environment and install dependencies.
- `cd sim2real && uv run python -m unittest discover -s tests -p 'test_*.py'`: run the deployment regression suite.
- `cd sim2real && uv run sim2real/sim_env/base_sim.py --robot g1`: start the local MuJoCo simulation; run the tracking policy separately as described in `sim2real/README.md`.
- `cd sim2real/docs && npm ci && npm run build`: reproduce the Docusaurus documentation build used by CI.

Training requires the `active-adaptation` environment and GPU backend setup. Follow `mimic-lite/README.md`; do not assume the deployment `uv` environment can run training.

## Coding Style & Naming Conventions

Use four-space indentation and modern Python type hints. Follow Python conventions: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Keep Hydra configuration names lowercase and descriptive (for example, `cfg/exp/ppo/train.yaml`). No repository-wide formatter is configured, so match neighboring imports and line wrapping and keep diffs focused.

## Testing Guidelines

Tests use the standard-library `unittest` framework. Place deployment tests in `sim2real/tests/`, name files `test_<feature>.py`, and name methods `test_<behavior>`. Add regression coverage for behavior changes, especially motion data, observation construction, policy inference, or robot I/O. There is no stated numeric coverage threshold; all relevant tests must pass.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, sentence-style subjects such as `Describe MimicLite project scope`; Conventional Commit prefixes are not used. Keep each commit scoped. Pull requests must summarize the change, identify changed components, keep English and Chinese landing pages synchronized, and avoid secrets/private endpoints/datasets/checkpoints. Follow `.github/PULL_REQUEST_TEMPLATE.md`, including its AI-content policy, and obtain approval from a listed maintainer.
