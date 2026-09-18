"""Build overlapping task references without commanding original robot joints.

The successful contact-manipulation recording supplies the right upper-arm
reference and the added-joint plan. Locomotion retains the original *reference*
poses rather than feeding a policy its own measured tracking error again.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from mini3_motion_reference import MINI3_XML, blend_poses, place_qpos, shorten_walk


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY = ROOT / "outputs/mini3_pick_carry_7dof_50mm_viewer/trajectory.npz"
DEFAULT_MOTION = ROOT / "any4hdmi/output/mini3/sonic/motions/211117/Neutral_walk_forward_005__A057.npz"
DEFAULT_SCENE = ROOT / "any4hdmi/assets/robots/mini3_mjlab/scene_pick_carry_7dof.xml"
RIGHT_ARM_NAMES = tuple(f"right_{part}_joint" for part in
                        ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch"))
EXTRA_NAMES = tuple(f"{side}_{part}_joint" for side in ("left", "right")
                    for part in ("elbow_yaw", "wrist_roll", "wrist_pitch"))
PHASES = ("APPROACH", "CLEAR_ARM", "REACH", "LOWER", "CLOSE", "LIFT",
          "CARRY", "PLACE", "RELEASE", "SUCCESS")


def _interpolate(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Column interpolation with endpoint holds, independent of array ownership."""
    return np.column_stack([np.interp(query, times, column) for column in values.T])


def _poses(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    values = values.copy()
    # q and -q represent one rotation. Unwrap signs before normalized LERP.
    for index in range(1, len(values)):
        if np.dot(values[index - 1, 3:7], values[index, 3:7]) < 0:
            values[index, 3:7] *= -1
    result = _interpolate(times, values, query)
    result[:, 3:7] /= np.linalg.norm(result[:, 3:7], axis=1, keepdims=True)
    return result


def _pose_at(times: np.ndarray, values: np.ndarray, time: float) -> np.ndarray:
    right = int(np.clip(np.searchsorted(times, time, side="right"), 1, len(times) - 1))
    return _poses(times[right - 1:right + 1], values[right - 1:right + 1], np.array([time]))[0]


@dataclass
class TaskReferencePlan:
    """50 Hz canonical policy reference, independent added-joint plan and clock."""

    times: np.ndarray
    qpos: np.ndarray
    extra_command: np.ndarray
    body_source_time: np.ndarray
    arm_source_time: np.ndarray
    phase: np.ndarray
    events: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    gates: list[dict[str, Any]]
    metadata: dict[str, Any]
    source_times: np.ndarray = field(repr=False)
    source_scene_qpos: np.ndarray = field(repr=False)

    @property
    def dt(self) -> float:
        return float(self.metadata["dt_s"])

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def sample(self, time_s: float) -> dict[str, Any]:
        """Sample references; this method never reads or writes simulation state."""
        if not math.isfinite(time_s):
            raise ValueError("Reference time must be finite")
        query = np.array([time_s])
        index = int(np.clip(np.searchsorted(self.times, time_s, side="right") - 1,
                            0, len(self.times) - 1))
        return {
            "qpos": _pose_at(self.times, self.qpos, time_s),
            "extra_command": _interpolate(self.times, self.extra_command, query)[0],
            "body_source_time": float(np.interp(time_s, self.times, self.body_source_time)),
            "arm_source_time": float(np.interp(time_s, self.times, self.arm_source_time)),
            "phase": str(self.phase[index]),
        }

    def sample_source_qpos(self, source_time: float) -> np.ndarray:
        """Return a copied scene pose for scratch FK, never an actuator command."""
        return _pose_at(self.source_times, self.source_scene_qpos, source_time)

    def save(self, output: Path) -> None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "task_reference.npz", time=self.times, qpos=self.qpos,
                            extra_command=self.extra_command,
                            body_source_time=self.body_source_time,
                            arm_source_time=self.arm_source_time, phase=self.phase)
        (output / "task_reference.json").write_text(json.dumps({
            **self.metadata, "events": self.events, "segments": self.segments,
            "gates": self.gates,
        }, indent=2) + "\n")


def _source_body_reference(
    query: np.ndarray, report: dict[str, Any], source_events: dict[str, dict[str, Any]],
    original_motion: Path, dt: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct the old walking reference, excluding its arm IK overrides."""
    with np.load(original_motion) as archive:
        original = np.asarray(archive["qpos"], dtype=float)
    approach, approach_meta = shorten_walk(original, 2.65)
    approach = place_qpos(approach, origin_xy=report["initial_root_xyz"][:2], align_travel=True)
    carry, carry_meta = shorten_walk(original, 1.0)
    carry_origin = source_events["CARRY"]["base_xyz"][:2]
    carry = place_qpos(carry, origin_xy=carry_origin, align_travel=True)
    pick_hold = approach[-1].copy()
    pick_hold[:2] = source_events["CLEAR_ARM"]["base_xyz"][:2]
    carry_start = pick_hold.copy()
    carry_start[:2] = carry_origin
    carry = np.concatenate((blend_poses(carry_start, carry[0], 1.0, dt)[:-1], carry))
    basket_hold = carry[-1].copy()
    basket_hold[:2] = source_events["PLACE"]["base_xyz"][:2]
    clear_time = float(source_events["CLEAR_ARM"]["time"])
    carry_time = float(source_events["CARRY"]["time"])
    place_time = float(source_events["PLACE"]["time"])
    result = np.tile(pick_hold, (len(query), 1))
    mask = query < clear_time - 1e-8
    result[mask] = _poses(np.arange(len(approach)) * dt, approach, query[mask])
    mask = (query >= carry_time - 1e-8) & (query < place_time - 1e-8)
    result[mask] = _poses(carry_time + np.arange(len(carry)) * dt, carry, query[mask])
    result[query >= place_time - 1e-8] = basket_hold
    return result, {"approach": approach_meta, "carry": carry_meta,
                    "source": "reconstructed original walking reference; measured root XY at transitions"}


def build_task_reference(
    trajectory_path: Path = DEFAULT_TRAJECTORY, *, report_path: Path | None = None,
    original_motion: Path = DEFAULT_MOTION, scene_path: Path = DEFAULT_SCENE,
    dt: float = 0.02, approach_overlap: float = 3.0, approach_wait: float = 0.0,
    lift_wait: float = 0.0, basket_wait: float = 0.0, basket_overlap: float = 0.6,
    arm_blend: float = 0.5, arm_source: str = "actual", extra_source: str = "command",
    arm_reference_bias: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> TaskReferencePlan:
    """Overlap arm clearance/placement with walking without speeding either up.

    ``approach_overlap=0, basket_overlap=0`` with the source report's waits
    reproduces its phase timing. The overlap-only arm blend is bypassed in that
    baseline. Durations include the original short settling tails of each active
    manipulation segment; only explicit approach/lift/basket waits are removed.
    """
    for name, value in {"dt": dt, "approach_overlap": approach_overlap,
                        "approach_wait": approach_wait, "lift_wait": lift_wait,
                        "basket_wait": basket_wait, "basket_overlap": basket_overlap,
                        "arm_blend": arm_blend}.items():
        if not math.isfinite(value) or value < 0 or (name == "dt" and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'dt' else 'non-negative'}")
    if not math.isclose(dt, 0.02, abs_tol=1e-9):
        raise ValueError("The source locomotion clip and task require dt=0.02 seconds")
    if arm_source not in ("actual", "command") or extra_source not in ("actual", "command"):
        raise ValueError("arm_source and extra_source must be 'actual' or 'command'")
    arm_reference_bias = np.asarray(arm_reference_bias, dtype=float)
    if arm_reference_bias.shape != (4,) or not np.isfinite(arm_reference_bias).all():
        raise ValueError("arm_reference_bias must contain four finite reference offsets")
    trajectory_path = Path(trajectory_path)
    report_path = Path(report_path) if report_path is not None else trajectory_path.with_name("report.json")
    report = json.loads(report_path.read_text())
    if not report.get("success"):
        raise ValueError("A successful source contact-manipulation episode is required")
    source_events = {event["phase"]: event for event in report["events"]}
    if any(name not in source_events for name in PHASES):
        raise ValueError("The source report is missing required task phase events")
    old = {name: round(float(source_events[name]["time"]) / dt) * dt for name in PHASES}
    if any(old[a] >= old[b] for a, b in zip(PHASES, PHASES[1:])):
        raise ValueError("Source task phases must have strictly increasing times")
    old_wait = report["wait_times_s"]
    walk_end = old["CLEAR_ARM"] - float(old_wait["approach"])
    lift_motion = old["CARRY"] - old["LIFT"] - float(old_wait["lift"])
    carry_motion = old["PLACE"] - old["CARRY"] - float(old_wait["basket"])
    if min(walk_end, lift_motion, carry_motion) <= 0:
        raise ValueError("Source waits exceed the recorded movement durations")
    approach_end = walk_end + approach_wait
    clear_start = max(0.0, approach_end - approach_overlap)
    blend_duration = arm_blend if approach_overlap > 0 else 0.0
    clear_motion_start = clear_start + blend_duration
    reach = max(approach_end, clear_motion_start + old["REACH"] - old["CLEAR_ARM"])
    lower = reach + old["LOWER"] - old["REACH"]
    close = lower + old["CLOSE"] - old["LOWER"]
    lift = close + old["LIFT"] - old["CLOSE"]
    carry = lift + lift_motion + lift_wait
    carry_end = carry + carry_motion + basket_wait
    place = max(carry, carry_end - basket_overlap)
    release = max(carry_end, place + old["RELEASE"] - old["PLACE"])
    success = release + old["SUCCESS"] - old["RELEASE"]
    new = dict(zip(PHASES, (0.0, clear_start, reach, lower, close, lift,
                            carry, place, release, success)))
    # Inputs are quantized to the control clock so every event is reviewable and
    # a held gate can stop at an exact sample. The time mapping still advances at
    # 1x within all active recorded movements.
    new = {phase: round(value / dt) * dt for phase, value in new.items()}
    clear_start, reach, lower, close, lift, carry, place, release, success = (
        new[name] for name in PHASES[1:])
    clear_motion_start = clear_start + blend_duration
    carry_end = carry + carry_motion + basket_wait
    times = np.arange(int(round(success / dt)) + 1, dtype=float) * dt
    arm_times = times.copy()
    body_times = times.copy()
    for index, time in enumerate(times):
        if time < clear_start:
            arm_times[index] = min(time, walk_end)
        elif time < reach:
            arm_times[index] = np.clip(old["CLEAR_ARM"] + time - clear_motion_start,
                                       old["CLEAR_ARM"], old["REACH"])
        elif time < lift:
            arm_times[index] = old["REACH"] + time - reach
        elif time < carry:
            arm_times[index] = min(old["LIFT"] + time - lift, old["LIFT"] + lift_motion)
        elif time < place:
            arm_times[index] = min(old["CARRY"] + time - carry, old["PLACE"])
        elif time < release:
            arm_times[index] = min(old["PLACE"] + time - place, old["RELEASE"])
        else:
            arm_times[index] = min(old["RELEASE"] + time - release, old["SUCCESS"])
        if time < approach_end:
            body_times[index] = min(time, old["CLEAR_ARM"] - dt)
        elif time < reach:
            # All poses in this interval use the same standing body reference.
            body_times[index] = old["CLEAR_ARM"]
        elif time < carry:
            body_times[index] = arm_times[index]
        elif time < carry_end:
            body_times[index] = min(old["CARRY"] + time - carry,
                                    old["CARRY"] + carry_motion)
        else:
            body_times[index] = max(old["PLACE"], arm_times[index])

    with np.load(trajectory_path) as archive:
        source_times = np.asarray(archive["time"], dtype=float)
        source_qpos = np.asarray(archive["qpos"], dtype=float)
        recorded_arm = np.asarray(archive["arm_command"], dtype=float) if arm_source == "command" else None
        recorded_extra = np.asarray(archive["extra_command"], dtype=float) if extra_source == "command" else None
    if (source_times.ndim != 1 or len(source_times) < 2
            or not np.isfinite(source_times).all() or np.any(np.diff(source_times) <= 0)
            or not np.isfinite(source_qpos).all()):
        raise ValueError("The source recording must contain finite poses and increasing sample times")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    canonical = mujoco.MjModel.from_xml_path(str(MINI3_XML))
    if source_qpos.shape != (len(source_times), model.nq):
        raise ValueError("The recording qpos width does not match its scene XML")
    arm_ids = np.array([model.joint(name).qposadr[0] for name in RIGHT_ARM_NAMES])
    canonical_arm_ids = np.array([canonical.joint(name).qposadr[0] for name in RIGHT_ARM_NAMES])
    extra_ids = np.array([model.joint(name).qposadr[0] for name in EXTRA_NAMES])
    recorded_arm = source_qpos[:, arm_ids] if recorded_arm is None else recorded_arm[:, :4]
    recorded_extra = source_qpos[:, extra_ids] if recorded_extra is None else recorded_extra
    if recorded_arm.shape != (len(source_times), 4) or recorded_extra.shape != (len(source_times), 6):
        raise ValueError("Recorded arm and added-joint channels must have widths four and six")
    qpos, body_meta = _source_body_reference(body_times, report, source_events, Path(original_motion), dt)
    arm = _interpolate(source_times, recorded_arm, arm_times)
    extra = _interpolate(source_times, recorded_extra, arm_times)
    active = times >= clear_start
    weight = active.astype(float)
    if blend_duration > 0:
        weight = np.clip((times - clear_start) / blend_duration, 0.0, 1.0)
        weight = weight * weight * (3.0 - 2.0 * weight)
    qpos[:, canonical_arm_ids] += weight[:, None] * (
        arm + arm_reference_bias - qpos[:, canonical_arm_ids])
    # The new joints stay in their recorded stow posture during locomotion; the
    # same blend applies when their manipulation reference begins.
    walking_extra = _interpolate(source_times, recorded_extra, np.minimum(times, walk_end))
    extra = walking_extra + weight[:, None] * (extra - walking_extra)
    phase = np.full(len(times), "APPROACH", dtype="U10")
    events = []
    for name in PHASES:
        phase[times >= new[name] - 1e-8] = name
        events.append({"phase": name, "time": new[name], "source_time": old[name]})
    segments = [{"phase": first, "start_s": new[first], "end_s": new[second],
                 "source_start_s": old[first], "source_end_s": old[second]}
                for first, second in zip(PHASES, PHASES[1:])]
    gates = [
        {"phase": "REACH", "time": reach, "condition": "approach reference complete; cube remains reachable"},
        {"phase": "CLOSE", "time": close, "condition": "actual finger-pad alignment around cube"},
        {"phase": "LIFT", "time": lift, "condition": "opposing actual finger contacts"},
        {"phase": "CARRY", "time": carry, "condition": "cube lifted clear of table with opposing contacts"},
        {"phase": "RELEASE", "time": release, "condition": "cube footprint inside basket and above rim"},
    ]
    metadata = {
        "dt_s": dt, "duration_s": float(times[-1]), "source_duration_s": old["SUCCESS"],
        "source_trajectory": str(trajectory_path.resolve()), "source_report": str(report_path.resolve()),
        "source_scene": str(Path(scene_path).resolve()), "original_motion": str(Path(original_motion).resolve()),
        "arm_source": arm_source, "extra_source": extra_source,
        "body_reference": body_meta, "original_policy_joint_count": 21,
        "original_right_arm_reference_joints": list(RIGHT_ARM_NAMES), "extra_joints": list(EXTRA_NAMES),
        "wait_times_s": {"approach": approach_wait, "lift": lift_wait, "basket": basket_wait},
        "overlap_s": {"approach_requested": approach_overlap,
                      "approach_actual": approach_end - clear_start,
                      "basket_requested": basket_overlap, "basket_actual": carry_end - place},
        "arm_blend_s": blend_duration, "approach_walk_end_s": walk_end,
        "carry_walk_end_s": carry + carry_motion,
        "arm_reference_bias_rad": arm_reference_bias.tolist(),
        "arm_raise_start": clear_start, "approach_end": approach_end,
        "close_start": close, "lift_start": lift, "carry_start": carry,
        "place_start": place, "release_start": release,
        "control_contract": "All original 21 joints are policy outputs; only the six added joints use this plan directly",
        "gates_are_advisory": True,
    }
    return TaskReferencePlan(times, qpos, extra, body_times, arm_times, phase,
                             events, segments, gates, metadata, source_times, source_qpos)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/mini3_policy_task_reference")
    parser.add_argument("--approach-overlap", type=float, default=3.0)
    parser.add_argument("--basket-overlap", type=float, default=0.6)
    parser.add_argument("--arm-source", choices=("actual", "command"), default="actual")
    args = parser.parse_args()
    plan = build_task_reference(args.trajectory, approach_overlap=args.approach_overlap,
                                basket_overlap=args.basket_overlap, arm_source=args.arm_source)
    plan.save(args.output)
    print(json.dumps({"duration_s": plan.duration, "events": plan.events}, indent=2))
