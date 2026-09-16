"""Export supported Mini3 PPO checkpoints without allocating a GPU simulator.

Network shapes come from saved VecNorm tensors. Observation definitions and
their ordering stay in the training configuration; robot metadata comes from
the same canonical asset configuration used by training. This is intentionally
limited to MimicLite PPO and the Mini3 asset, rather than guessing other layouts.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent


def _observation_spec(cfg, state):
    import torch
    from torchrl.data import Composite, Unbounded

    if cfg.algo.vecnorm is not True:
        raise ValueError("CPU export requires saved VecNorm statistics to identify observation shapes")
    statistics = state.get("vecnorms", {})
    specs = {}
    for key in cfg.algo.in_keys:
        if not isinstance(key, str) or key not in cfg.task.observation:
            raise ValueError(f"Missing observation configuration for {key!r}")
        moments = [statistics.get(f"{key}.{name}") for name in ("sum", "ssq")]
        if any(not isinstance(value, torch.Tensor) or value.ndim != 1 for value in moments):
            raise ValueError(f"Missing one-dimensional VecNorm moments for {key!r}")
        if moments[0].shape != moments[1].shape or moments[0].numel() == 0:
            raise ValueError(f"Inconsistent VecNorm shape for {key!r}")
        specs[key] = Unbounded(shape=(1, moments[0].numel()), device="cpu")
    return Composite(specs, shape=(1,), device="cpu")


def _validate_checkpoint_config(cfg, saved_cfg) -> bool:
    """Reject a same-shape YAML that changes observation meaning or term order."""
    from omegaconf import OmegaConf

    if saved_cfg is None:
        return False
    saved_cfg = OmegaConf.create(saved_cfg)
    fields = [
        "algo._target_", "algo.in_keys", "algo.vecnorm", "algo.layer_norm",
        "task.robot", "task.input.action._target_", "task.input.action.action_scaling",
        "task.command._target_", "task.command.future_steps", "task.command.diff_future_steps",
        "task.command.tracking_body_names", "task.command.tracking_joint_names",
        "task.command.obs_body_names", "task.command.root_body_name", "task.command.anchor_body_name",
        "task.shared.termination_root_body_name", "task.sim.step_dt", "task.sim.mujoco_physics_dt",
        *(f"task.observation.{key}" for key in cfg.algo.in_keys),
    ]

    def representation(config, field):
        value = OmegaConf.select(config, field)
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        # Dict equality ignores order. Serialization deliberately preserves it,
        # because observation term insertion order determines feature positions.
        return json.dumps(value, sort_keys=False)

    mismatched = [field for field in fields if representation(cfg, field) != representation(saved_cfg, field)]
    if mismatched:
        raise ValueError(
            "Training YAML does not match checkpoint observation/control metadata: "
            + ", ".join(mismatched)
        )
    return True


def _metadata_env(cfg, observation_spec):
    import torch
    from omegaconf import OmegaConf
    from active_adaptation.envs.utils import find_bodies, find_joints
    from mimic_lite.assets.mini3 import MINI3_CFG
    from mimic_lite.tasks.multi_dataset import normalize_motion_cfgs

    if dict(cfg.task.robot) != {"name": "mini3-mesh"}:
        raise ValueError("CPU export supports mini3-mesh without asset overrides")
    if cfg.task.input.action._target_ != "mimic_lite.JointPosition":
        raise ValueError("CPU export supports mimic_lite.JointPosition actions only")
    if cfg.task.command._target_ != "mimic_lite.RobotTracking":
        raise ValueError("CPU export supports mimic_lite.RobotTracking commands only")

    joint_names = list(MINI3_CFG.joint_names_simulation)
    body_names = list(MINI3_CFG.body_names_simulation)
    scaling = OmegaConf.to_container(cfg.task.input.action.action_scaling, resolve=True)
    if not isinstance(scaling, dict) or list(scaling) != joint_names:
        raise ValueError("Action scaling must explicitly cover all 21 canonical Mini3 joints in order")
    asset = SimpleNamespace(
        cfg=MINI3_CFG, joint_names=joint_names, body_names=body_names,
        actuators=[SimpleNamespace(cfg=actuator) for actuator in MINI3_CFG.actuators.values()],
    )
    command = cfg.task.command
    _, tracking_bodies = find_bodies(asset, list(command.tracking_body_names))
    _, tracking_joints = find_joints(asset, list(command.tracking_joint_names))
    if tracking_joints != joint_names:
        raise ValueError("Tracking joints must follow the canonical Mini3 contract")
    for name in (command.root_body_name, command.anchor_body_name, cfg.task.shared.termination_root_body_name):
        if name not in body_names:
            raise ValueError(f"Unknown Mini3 tracking body: {name}")
    future_steps = torch.tensor(list(command.future_steps), dtype=torch.long)
    if future_steps.ndim != 1 or (future_steps == 0).sum().item() != 1:
        raise ValueError("Tracking future_steps must include frame zero exactly once")
    action_scaling = torch.tensor(list(scaling.values()), dtype=torch.float32)
    if not torch.isfinite(action_scaling).all():
        raise ValueError("Action scaling contains non-finite values")
    return SimpleNamespace(
        cfg=cfg.task,
        observation_spec=observation_spec,
        scene=SimpleNamespace(articulations={"robot": asset}),
        action_manager=SimpleNamespace(
            action_dim=len(joint_names), joint_names=joint_names,
            action_scaling=action_scaling,
        ),
        command_manager=SimpleNamespace(
            motion_cfgs=normalize_motion_cfgs(OmegaConf.to_container(command.motion_cfgs, resolve=True)),
            future_steps=future_steps, tracking_body_names=tracking_bodies,
            tracking_joint_names=tracking_joints, root_body_name=str(command.root_body_name),
            anchor_body_name=str(command.anchor_body_name),
        ),
    )


def _verify_loaded_state(policy, saved_state) -> int:
    """Catch partial loading and inconsistent duplicated/shared state tensors."""
    import torch

    current = policy.state_dict()
    if set(current) != set(saved_state):
        raise ValueError(f"Checkpoint module keys differ: {set(current) ^ set(saved_state)}")
    count = 0
    for module_name, saved in saved_state.items():
        if module_name == "last_iter":
            continue
        actual = current[module_name]
        if set(saved) != set(actual):
            raise ValueError(f"Incomplete checkpoint state for {module_name}")
        for name, value in saved.items():
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint tensor {module_name}.{name}")
            if not torch.equal(actual[name], value):
                raise ValueError(f"Checkpoint tensor was not restored exactly: {module_name}.{name}")
            count += 1
    return count


def _verify_onnx(policy, observation_spec, onnx_path: Path) -> float:
    import numpy as np
    import onnxruntime as ort
    import torch
    from active_adaptation.learning.modules.vecnorm import VecNorm

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    deploy = policy.get_rollout_policy("deploy").eval()
    input_names = [item.name for item in session.get_inputs()]
    if set(input_names) != set(deploy.in_keys):
        raise ValueError(f"Unexpected ONNX observation names: {input_names}")
    generator = torch.Generator(device="cpu").manual_seed(12345)
    largest_error = 0.0
    with torch.inference_mode(), VecNorm.freeze():
        for amplitude in (0.0, 0.5, 1.0, 2.0, 5.0):
            td = observation_spec[0].zero()
            for key in deploy.in_keys:
                mean, std = policy.vecnorms[key]._compute()
                td[key] = mean + amplitude * std * torch.randn(mean.shape, generator=generator)
            inputs = {key: td[key].numpy().copy() for key in input_names}
            expected = deploy(td.clone())["action"].numpy()
            actual = session.run(["action"], inputs)[0]
            if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                raise ValueError("Non-finite action during ONNX validation")
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
            largest_error = max(largest_error, float(np.max(np.abs(actual - expected))))
    return largest_error


def export_checkpoint_cpu(cfg, directory: Path) -> Path:
    """Write standard deploy ONNX/YAML and validation evidence under directory.

    The caller supplies a resolved full training config with checkpoint_path.
    Checkpoints are read with pickle enabled, as in the project's normal loader;
    the caller must select only trusted checkpoints.
    """
    import torch
    import active_adaptation as aa
    from torchrl.data import Unbounded

    if cfg.algo._target_ != "mimic_lite_learning.ppo.PPOPolicy":
        raise ValueError("CPU export currently supports mimic_lite_learning.ppo.PPOPolicy only")
    if aa.is_distributed():
        raise ValueError("CPU export must run outside torchrun/distributed training")
    try:
        aa.get_backend()
    except RuntimeError:
        aa.set_backend("mjlab")
    from mimic_lite_learning.ppo import PPOPolicy

    checkpoint_path = Path(cfg.checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("policy"), dict):
        raise ValueError("Expected a full MimicLite training checkpoint containing policy state")
    if checkpoint.get("env"):
        raise ValueError("CPU export does not support non-empty environment checkpoint state")
    config_matched = _validate_checkpoint_config(cfg, checkpoint.get("cfg"))
    saved_state = checkpoint["policy"]
    observation_spec = _observation_spec(cfg, saved_state)
    env = _metadata_env(cfg, observation_spec)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(min(old_threads, 4))
    try:
        policy = PPOPolicy(
            cfg.algo, observation_spec,
            Unbounded(shape=(1, 21), device="cpu"),
            Unbounded(shape=(1, 1), device="cpu"),
            "cpu", env,
        )
        failed = policy.load_state_dict(saved_state, strict=True)
        if failed:
            raise ValueError(f"Checkpoint failed strict loading for modules: {failed}")
        restored_count = _verify_loaded_state(policy, saved_state)
        policy.eval()

        spec = importlib.util.spec_from_file_location("mimiclite_standard_export", ROOT / "mimic-lite/scripts/play.py")
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        exporter.FILE_PATH = directory
        # The standard tag helper reloads CUDA tensors unnecessarily. Use the
        # exact same tags from the checkpoint already loaded on CPU.
        run_id = str(checkpoint.get("wandb", {}).get("id", "unknown"))
        match = re.search(r"checkpoint_(\d+)", checkpoint_path.name)
        step = match.group(1) if match else ("final" if checkpoint_path.name.endswith("_final.pt") else "unknown")
        exporter._checkpoint_tags = lambda _: (run_id, step)
        exporter.export_policy(cfg, env, policy)
        onnx_path = directory / "exports" / str(cfg.task.name) / f"policy-{run_id}-{step}.onnx"
        largest_error = _verify_onnx(policy, observation_spec, onnx_path)
        _verify_loaded_state(policy, saved_state)
        evidence = {
            "device": "cpu", "strict_load": True, "restored_tensors": restored_count,
            "vecnorm_restored_exactly": True,
            "checkpoint_config_matched": config_matched,
            "observation_shapes": {key: list(observation_spec[key].shape[1:]) for key in cfg.algo.in_keys},
            "onnx_comparison_samples": 5, "onnx_max_abs_error": largest_error,
        }
        (directory / "cpu_export_validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(f"CPU export validated: {json.dumps(evidence)}", flush=True)
        return onnx_path
    finally:
        torch.set_num_threads(old_threads)
