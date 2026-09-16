#!/usr/bin/env python3
"""Audit recorded arm collisions, including pairs hidden by MuJoCo filters.

This tool loads an independent model/data pair and calls mj_forward only. It
never advances physics, changes collision masks, or edits the source trajectory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ARM_WORDS = ("elbow", "forearm", "wrist", "gripper")
GRASP_PHASES = {"LOWER", "CLOSE", "LIFT", "CARRY", "PLACE", "RELEASE", "SUCCESS"}
TARGET_GEOM = "pick_cube_3_geom"


def make_distance_model(model: mujoco.MjModel) -> mujoco.MjModel:
    """Legacy query copy for reproducing engine distance-query differences.

    Do not use this copy as the collision-avoidance interface: large-margin
    queries can also fail in this backend. Use collision_distance instead.
    Retained only for diagnostic comparisons; the audit never uses this copy.
    """
    distance_model = copy.copy(model)
    distance_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    return distance_model


def collision_distance(model: mujoco.MjModel, data: mujoco.MjData, first: int,
                       second: int, distmax: float = .005) -> float:
    """Robust signed clearance for IK/auditing without changing model/data.

    Box pairs use all 15 separating axes. Negative values are their minimum
    translation penetration; positive values are conservative gap lower bounds.
    Other pairs first query zero-margin penetration, then at most 5 mm positive
    clearance to avoid large-margin convex-query artefacts in MuJoCo 3.11.
    """
    if model.geom_type[first] == model.geom_type[second] == mujoco.mjtGeom.mjGEOM_BOX:
        rotation1 = data.geom_xmat[first].reshape(3, 3)
        rotation2 = data.geom_xmat[second].reshape(3, 3)
        axes = np.concatenate((rotation1.T, rotation2.T,
                               np.cross(rotation1.T[:, None, :], rotation2.T[None, :, :]).reshape(-1, 3)))
        norms = np.linalg.norm(axes, axis=1)
        axes = axes[norms > 1e-10] / norms[norms > 1e-10, None]
        center = data.geom_xpos[second] - data.geom_xpos[first]
        gaps = (np.abs(axes @ center) - np.abs(axes @ rotation1) @ model.geom_size[first]
                - np.abs(axes @ rotation2) @ model.geom_size[second])
        return min(float(np.max(gaps)), distmax)
    penetration = float(mujoco.mj_geomDistance(model, data, first, second, 0.0, None))
    if penetration < 0:
        return penetration
    near = float(mujoco.mj_geomDistance(model, data, first, second, min(distmax, .005), None))
    return max(0.0, near)


def geom_name(model: mujoco.MjModel, geom: int) -> str:
    return model.geom(int(geom)).name or f"geom_{geom}"


def arm_side(name: str) -> str | None:
    return next((side for side in ("left", "right") if name.startswith(side + "_")), None)


def is_arm_geom(name: str, side: str | None = None) -> bool:
    return any(word in name for word in ARM_WORDS) and (side is None or arm_side(name) == side)


def physical_geoms(model: mujoco.MjModel) -> list[int]:
    """Ignore massless camera decoration and visual-only duplicate STL meshes."""
    return [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]


def pair_classification(model: mujoco.MjModel, first: int, second: int) -> str:
    """Classify intended mechanical interfaces narrowly, independent of filters."""
    names = (geom_name(model, first), geom_name(model, second))
    bodies = tuple(int(model.geom_bodyid[i]) for i in (first, second))
    sides = tuple(arm_side(name) for name in names)
    if TARGET_GEOM in names:
        other = names[1] if names[0] == TARGET_GEOM else names[0]
        if "gripper_finger" in other:
            return "allowed_finger_target"
        return "unexpected_arm_target"
    if bodies[0] == bodies[1]:
        return "allowed_same_component"
    same_arm = sides[0] is not None and sides[0] == sides[1]
    if same_arm and all("gripper_finger" in name for name in names):
        return "allowed_finger_closure"
    if same_arm:
        parent_child = (int(model.body_parentid[bodies[0]]) == bodies[1]
                        or int(model.body_parentid[bodies[1]]) == bodies[0])
        # Direct structural neighbours may intersect at their shaft/slider.
        # A grandparent relationship produced by rigid fusion is NOT sufficient.
        if parent_child and all(any(word in name for word in (*ARM_WORDS, "shoulder")) for name in names):
            return "allowed_mechanical_interface"
    if any(name.startswith(("pick_table", "basket_table")) for name in names):
        return "unexpected_arm_table"
    if any(name.startswith("pick_basket") for name in names):
        return "unexpected_arm_basket"
    if any(name.startswith("pick_cube") for name in names):
        return "unexpected_arm_other_cube"
    if any(name == "pick_floor" or model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
           for name, i in zip(names, (first, second), strict=True)):
        return "unexpected_arm_floor"
    return "unexpected_arm_self"


def collision_filter_reasons(model: mujoco.MjModel, first: int, second: int) -> list[str]:
    """Explain automatic model-level filters; explicit geom pairs override these.

    Mirrors MuJoCo 3.11 collision driver's bitmask/weld-parent/body-exclude rules.
    Sleeping and custom contact callbacks are outside this offline model audit.
    """
    pair = tuple(sorted((int(first), int(second))))
    explicit = {tuple(sorted((int(a), int(b)))) for a, b in zip(model.pair_geom1, model.pair_geom2)}
    if pair in explicit:
        return []
    body1, body2 = int(model.geom_bodyid[first]), int(model.geom_bodyid[second])
    weld1, weld2 = int(model.body_weldid[body1]), int(model.body_weldid[body2])
    result = []
    if not (int(model.geom_contype[first]) & int(model.geom_conaffinity[second])
            or int(model.geom_contype[second]) & int(model.geom_conaffinity[first])):
        result.append("contype_conaffinity")
    if weld1 == weld2:
        result.append("same_welded_body")
    elif not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)):
        parent1 = int(model.body_weldid[int(model.body_parentid[weld1])])
        parent2 = int(model.body_weldid[int(model.body_parentid[weld2])])
        if weld1 and weld2 and (weld1 == parent2 or weld2 == parent1):
            result.append("welded_parent_child")
    low, high = sorted((body1, body2))
    if (low << 16) + high in set(map(int, model.exclude_signature)):
        result.append("explicit_body_exclude")
    if int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT):
        result.append("contacts_globally_disabled")
    return result


def all_arm_pairs(model: mujoco.MjModel, side: str | None = None) -> list[tuple[int, int]]:
    geoms = physical_geoms(model)
    arms = {i for i in geoms if is_arm_geom(geom_name(model, i), side)}
    return [(a, b) for k, a in enumerate(geoms) for b in geoms[k + 1:] if a in arms or b in arms]


def arm_collision_pairs(model: mujoco.MjModel, side: str | None = None) -> list[tuple[int, int]]:
    """Pairs for collision-avoiding IK, including unwanted filtered self pairs.

    Finger/blue-cube contact and direct mechanical interfaces are omitted.
    Palm, wrist and forearm contact with the blue cube remain prohibited.
    Collision filters do not establish mechanical permission: a distal forearm
    crossing the upper arm still appears even if welded-parent filtering hides it.
    """
    if side not in (None, "left", "right"):
        raise ValueError("side must be left, right, or None")
    return [pair for pair in all_arm_pairs(model, side)
            if pair_classification(model, *pair).startswith("unexpected_")]


def audit(trajectory: Path, scene: Path, *, distance_limit: float = .10,
          contact_tolerance: float = .002) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    with np.load(trajectory, allow_pickle=False) as saved:
        times, phases, qpos = saved["time"].copy(), saved["phase"].copy(), saved["qpos"].copy()
        qvel = saved["qvel"].copy() if "qvel" in saved else np.zeros((len(times), model.nv))
    if len(times) == 0 or qpos.shape != (len(times), model.nq) or qvel.shape != (len(times), model.nv):
        raise ValueError("Trajectory dimensions do not match this scene")
    if phases.shape != times.shape or np.any(np.diff(times) <= 0):
        raise ValueError("Expected one phase per sample and strictly increasing timestamps")
    if not all(np.isfinite(values).all() for values in (times, qpos, qvel)):
        raise ValueError("Trajectory contains nonfinite values")
    pairs = all_arm_pairs(model)
    records = []
    for first, second in pairs:
        box_pair = model.geom_type[first] == model.geom_type[second] == mujoco.mjtGeom.mjGEOM_BOX
        pair_limit = distance_limit if box_pair else min(distance_limit, .005)
        records.append({
            "geoms": [geom_name(model, first), geom_name(model, second)],
            "bodies": [model.body(int(model.geom_bodyid[i])).name for i in (first, second)],
            "classification": pair_classification(model, first, second),
            "filter_reasons": collision_filter_reasons(model, first, second),
            "minimum_gap_m": pair_limit, "minimum_gap_is_lower_bound": True,
            "distance_cap_m": pair_limit,
            "gap_metric": "15-axis OBB separation lower bound" if box_pair else "MuJoCo zero-margin penetration and <=5mm near clearance",
            "maximum_penetration_m": 0.0, "first_penetration_time_s": None,
            "worst_time_s": None, "worst_phase": None, "penetrating_frames": 0,
            "maximum_consecutive_penetrating_frames": 0, "actual_contact_frames": 0,
            "maximum_actual_contact_penetration_m": 0.0,
            "unexpected_frames": 0, "excessive_allowed_contact_frames": 0,
            "first_unexpected_time_s": None,
        })
    consecutive = np.zeros(len(pairs), dtype=int)
    per_phase: dict[str, dict[str, int | float]] = {}
    for frame, (timestamp, phase, pose, velocity) in enumerate(zip(times, phases, qpos, qvel, strict=True)):
        data.qpos[:] = pose
        data.qvel[:] = velocity
        data.time = float(timestamp)
        mujoco.mj_forward(model, data)
        active_pairs: dict[tuple[int, int], float] = {}
        for contact in data.contact:
            key = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            active_pairs[key] = max(active_pairs.get(key, 0.0), max(0.0, -float(contact.dist)))
        stats = per_phase.setdefault(str(phase), {"frames": 0, "frames_with_unexpected_collision": 0,
                                                 "maximum_unexpected_penetration_m": 0.0})
        stats["frames"] += 1
        unexpected_this_frame = False
        for pair_id, ((first, second), record) in enumerate(zip(pairs, records, strict=True)):
            distance = collision_distance(model, data, first, second, distance_limit)
            if distance < record["minimum_gap_m"]:
                record["minimum_gap_m"] = distance
                record["minimum_gap_is_lower_bound"] = distance > 0 and record["gap_metric"].startswith("15-axis")
                record["worst_time_s"] = float(timestamp)
                record["worst_phase"] = str(phase)
            record["actual_contact_frames"] += (first, second) in active_pairs
            record["maximum_actual_contact_penetration_m"] = max(
                record["maximum_actual_contact_penetration_m"], active_pairs.get((first, second), 0.0))
            if distance >= -1e-7:
                consecutive[pair_id] = 0
                continue
            depth = -distance
            record["maximum_penetration_m"] = max(record["maximum_penetration_m"], depth)
            record["penetrating_frames"] += 1
            consecutive[pair_id] += 1
            record["maximum_consecutive_penetrating_frames"] = max(
                record["maximum_consecutive_penetrating_frames"], int(consecutive[pair_id]))
            if record["first_penetration_time_s"] is None:
                record["first_penetration_time_s"] = float(timestamp)
            classification = record["classification"]
            unexpected = classification.startswith("unexpected_") or (
                classification == "allowed_finger_target" and str(phase) not in GRASP_PHASES)
            if unexpected:
                record["unexpected_frames"] += 1
                unexpected_this_frame = True
                stats["maximum_unexpected_penetration_m"] = max(stats["maximum_unexpected_penetration_m"], depth)
                if record["first_unexpected_time_s"] is None:
                    record["first_unexpected_time_s"] = float(timestamp)
            elif classification in ("allowed_finger_target", "allowed_finger_closure") and depth > contact_tolerance:
                record["excessive_allowed_contact_frames"] += 1
        stats["frames_with_unexpected_collision"] += unexpected_this_frame
        if frame % 500 == 0:
            print(f"Audited {frame}/{len(times)} recorded samples, {len(pairs)} arm pairs", flush=True)
    records.sort(key=lambda item: item["maximum_penetration_m"], reverse=True)
    unexpected_records = [record for record in records if record["unexpected_frames"]]
    return {
        "trajectory": str(trajectory.resolve()), "scene": str(scene.resolve()),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(), "mujoco_version": mujoco.__version__,
        "frames": len(times), "first_time_s": float(times[0]), "last_time_s": float(times[-1]),
        "method": "Independent MjModel/MjData; mj_forward and signed mj_geomDistance; no mj_step or changed masks",
        "distance_query_backend": "15-axis OBB SAT for boxes; original MuJoCo zero-margin penetration and <=5mm near queries for other geoms; original solver settings unchanged",
        "rules": {
            "selected": "Physical geoms named elbow/forearm/wrist/gripper against all physical geoms; includes filtered pairs",
            "allowed": "Same component, same-side directly parented arm interfaces, same-gripper finger closure",
            "allowed_grasp": f"Only gripper fingers against {TARGET_GEOM}, phases {sorted(GRASP_PHASES)}",
            "unexpected": "All other arm/environment/self intersections, including filtered nonadjacent fused-body pairs",
            "distance_limit_m": distance_limit, "distance_limit_note": "Distances returned at cap are lower bounds, not measured clearances",
            "soft_contact_limit_m": contact_tolerance,
            "scope_limit": "Discrete saved frames, collision proxies (meshes use MuJoCo collision representation); excludes camera/visual-only geoms",
        },
        "summary": {
            "arm_pairs": len(pairs), "unexpected_collision_pairs": len(unexpected_records),
            "maximum_unexpected_penetration_m": max((r["maximum_penetration_m"] for r in unexpected_records), default=0.0),
            "filtered_unexpected_pairs": sum(bool(r["filter_reasons"]) for r in unexpected_records),
        },
        "per_phase": per_phase, "unexpected_collisions": unexpected_records, "all_pairs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance-limit", type=float, default=.10)
    parser.add_argument("--contact-tolerance", type=float, default=.002)
    args = parser.parse_args()
    if args.distance_limit <= 0 or args.contact_tolerance < 0:
        parser.error("distance-limit must be positive and contact-tolerance nonnegative")
    report = audit(args.trajectory, args.scene, distance_limit=args.distance_limit,
                   contact_tolerance=args.contact_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved arm collision audit: {args.output}")


if __name__ == "__main__":
    main()
