from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
from any4hdmi.utils.mini3_real_motor import mini3_motor_type

import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.assets.asset_cfg import (
    ActuatorCfg,
    AssetCfg,
    ContactSensorCfg,
    InitialStateCfg,
    MjlabCollisionCfg,
)
from active_adaptation.registry import Registry


registry = Registry.instance()

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MINI3_ASSET_ROOT = REPOSITORY_ROOT / "any4hdmi" / "assets" / "robots" / "mini3_mjlab"
MINI3_MJCF_PATH = MINI3_ASSET_ROOT / "mini3.xml"
MINI3_URDF_PATH = MINI3_ASSET_ROOT / "urdf" / "mini3.urdf"

MINI3_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
)

MINI3_BODY_NAMES = (
    "base_link",
    "imu_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_pitch_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_pitch_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_pitch_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_pitch_link",
)

MINI3_STIFFNESS = {
    "left_hip_pitch_joint": 60.0,
    "left_hip_roll_joint": 55.0,
    "left_hip_yaw_joint": 25.0,
    "left_knee_pitch_joint": 60.0,
    "left_ankle_pitch_joint": 50.0,
    "left_ankle_roll_joint": 45.0,
    "right_hip_pitch_joint": 60.0,
    "right_hip_roll_joint": 55.0,
    "right_hip_yaw_joint": 25.0,
    "right_knee_pitch_joint": 60.0,
    "right_ankle_pitch_joint": 50.0,
    "right_ankle_roll_joint": 45.0,
    "waist_yaw_joint": 65.0,
    "left_shoulder_pitch_joint": 30.0,
    "left_shoulder_roll_joint": 25.0,
    "left_shoulder_yaw_joint": 30.0,
    "left_elbow_pitch_joint": 20.0,
    "right_shoulder_pitch_joint": 30.0,
    "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 30.0,
    "right_elbow_pitch_joint": 20.0,
}

MINI3_DAMPING = {
    "left_hip_pitch_joint": 4.5,
    "left_hip_roll_joint": 2.8,
    "left_hip_yaw_joint": 1.1,
    "left_knee_pitch_joint": 4.5,
    "left_ankle_pitch_joint": 1.2,
    "left_ankle_roll_joint": 1.2,
    "right_hip_pitch_joint": 4.5,
    "right_hip_roll_joint": 2.8,
    "right_hip_yaw_joint": 1.1,
    "right_knee_pitch_joint": 4.5,
    "right_ankle_pitch_joint": 1.2,
    "right_ankle_roll_joint": 1.2,
    "waist_yaw_joint": 3.0,
    "left_shoulder_pitch_joint": 1.0,
    "left_shoulder_roll_joint": 2.0,
    "left_shoulder_yaw_joint": 1.0,
    "left_elbow_pitch_joint": 1.0,
    "right_shoulder_pitch_joint": 1.0,
    "right_shoulder_roll_joint": 2.0,
    "right_shoulder_yaw_joint": 1.0,
    "right_elbow_pitch_joint": 1.0,
}


def _joint_group_value(joint_name: str, *, body: float, ankle: float, arm: float) -> float:
    if "ankle" in joint_name:
        return ankle
    if "shoulder" in joint_name or "elbow" in joint_name:
        return arm
    return body


MINI3_EFFORT_LIMIT = {
    name: _joint_group_value(name, body=27.0, ankle=25.0, arm=12.5)
    for name in MINI3_JOINT_NAMES
}
MINI3_VELOCITY_LIMIT = {
    name: _joint_group_value(name, body=10.0, ankle=45.0, arm=45.0)
    for name in MINI3_JOINT_NAMES
}
MINI3_ARMATURE = {
    name: _joint_group_value(
        name,
        body=0.04595206677913666,
        ankle=0.01,
        arm=0.0019,
    )
    for name in MINI3_JOINT_NAMES
}
MINI3_FRICTION = {
    name: _joint_group_value(
        name,
        body=1.0523104667663574,
        ankle=0.7,
        arm=0.7,
    )
    for name in MINI3_JOINT_NAMES
}


@dataclass(kw_only=True, frozen=True)
class Mini3RealMotorAssetCfg(ActuatorCfg):
    """Backend-neutral wrapper for the MJLab Mini3 real-motor actuator."""

    motor_type: Literal["4340p", "4310p"]
    parallel_ankle_side: Literal["left", "right"] | None = None
    stiffness_by_joint: dict[str, float] | None = None
    damping_by_joint: dict[str, float] | None = None
    torque_response_enabled: bool = True
    torque_response_kp: float = 0.0
    torque_response_ki: float = 90.6769527429
    torque_response_plant_tau_s: float = 0.00393417593548
    torque_response_delay_steps: float = 1.0
    tn_torque_limit_enabled: bool = True
    tn_limit_after_response: bool = True
    kt_output_model_enabled: bool = True
    ankle_motor_torque_limit: float = 12.5

    def mjlab(self):
        from mimic_lite.assets.mini3_real_motor import (
            Mini3ParallelAnkleRealMotorActuatorCfg,
            Mini3RealMotorActuatorCfg,
        )

        target_names_expr = (
            tuple(self.joint_names_expr)
            if isinstance(self.joint_names_expr, list)
            else (self.joint_names_expr,)
        )
        kwargs = {
            "target_names_expr": target_names_expr,
            "effort_limit": float(self.effort_limit),
            "stiffness": float(self.stiffness),
            "damping": float(self.damping),
            "frictionloss": float(self.friction),
            "armature": float(self.armature),
            "motor_type": self.motor_type,
            "stiffness_by_joint": self.stiffness_by_joint,
            "damping_by_joint": self.damping_by_joint,
            "torque_response_enabled": self.torque_response_enabled,
            "torque_response_kp": self.torque_response_kp,
            "torque_response_ki": self.torque_response_ki,
            "torque_response_plant_tau_s": self.torque_response_plant_tau_s,
            "torque_response_delay_steps": self.torque_response_delay_steps,
            "tn_torque_limit_enabled": self.tn_torque_limit_enabled,
            "tn_limit_after_response": self.tn_limit_after_response,
            "kt_output_model_enabled": self.kt_output_model_enabled,
        }
        if self.parallel_ankle_side is None:
            return Mini3RealMotorActuatorCfg(**kwargs)
        return Mini3ParallelAnkleRealMotorActuatorCfg(
            **kwargs,
            side=self.parallel_ankle_side,
            ankle_motor_torque_limit=self.ankle_motor_torque_limit,
        )


def _serial_real_motor_actuator(joint_name: str) -> Mini3RealMotorAssetCfg:
    return Mini3RealMotorAssetCfg(
        joint_names_expr=joint_name,
        effort_limit=MINI3_EFFORT_LIMIT[joint_name],
        velocity_limit=MINI3_VELOCITY_LIMIT[joint_name],
        stiffness=MINI3_STIFFNESS[joint_name],
        damping=MINI3_DAMPING[joint_name],
        friction=MINI3_FRICTION[joint_name],
        armature=MINI3_ARMATURE[joint_name],
        motor_type=mini3_motor_type(joint_name),
    )


def _parallel_ankle_actuator(
    side: Literal["left", "right"],
) -> Mini3RealMotorAssetCfg:
    joint_names = [
        f"{side}_ankle_pitch_joint",
        f"{side}_ankle_roll_joint",
    ]
    return Mini3RealMotorAssetCfg(
        joint_names_expr=joint_names,
        # This is the serial-coordinate force range. Each physical 4310P is
        # limited separately to 12.5 Nm before J.T maps back to the joint pair.
        effort_limit=25.0,
        velocity_limit=45.0,
        stiffness=MINI3_STIFFNESS[joint_names[0]],
        damping=MINI3_DAMPING[joint_names[0]],
        friction=0.7,
        armature=0.01,
        motor_type="4310p",
        parallel_ankle_side=side,
        stiffness_by_joint={name: MINI3_STIFFNESS[name] for name in joint_names},
        damping_by_joint={name: MINI3_DAMPING[name] for name in joint_names},
        ankle_motor_torque_limit=12.5,
    )


def _real_motor_actuators() -> dict[str, Mini3RealMotorAssetCfg]:
    """Build groups whose flattened target order is the 21-joint contract."""

    actuators: dict[str, Mini3RealMotorAssetCfg] = {}
    skip = set()
    for joint_name in MINI3_JOINT_NAMES:
        if joint_name in skip:
            continue
        if joint_name == "left_ankle_pitch_joint":
            actuators["left_parallel_ankle"] = _parallel_ankle_actuator("left")
            skip.add("left_ankle_roll_joint")
        elif joint_name == "right_ankle_pitch_joint":
            actuators["right_parallel_ankle"] = _parallel_ankle_actuator("right")
            skip.add("right_ankle_roll_joint")
        else:
            actuators[joint_name] = _serial_real_motor_actuator(joint_name)
    return actuators


def _validate_source_contract() -> None:
    expected = list(MINI3_JOINT_NAMES)
    if len(expected) != 21 or len(set(expected)) != 21:
        raise ValueError("Mini3 must define exactly 21 unique joints")
    for label, values in (
        ("stiffness", MINI3_STIFFNESS),
        ("damping", MINI3_DAMPING),
        ("effort", MINI3_EFFORT_LIMIT),
        ("velocity", MINI3_VELOCITY_LIMIT),
        ("armature", MINI3_ARMATURE),
        ("friction", MINI3_FRICTION),
    ):
        if list(values) != expected:
            raise ValueError(
                f"Mini3 {label} must cover the canonical joint order exactly; "
                f"expected={expected}, actual={list(values)}"
            )
    for path in (MINI3_MJCF_PATH, MINI3_URDF_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Mini3 asset file not found: {path}")

    model = mujoco.MjModel.from_xml_path(str(MINI3_MJCF_PATH))
    joint_names = [
        model.joint(joint_id).name
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if joint_names != expected:
        raise ValueError(
            "Mini3 MJCF joint order differs from the policy contract: "
            f"expected={expected}, actual={joint_names}"
        )
    actuator_targets = [
        model.joint(int(model.actuator_trnid[actuator_id, 0])).name
        for actuator_id in range(model.nu)
    ]
    if actuator_targets != expected:
        raise ValueError(
            "Mini3 actuator transmission targets must match the policy joint order: "
            f"expected={expected}, actual={actuator_targets}"
        )
    if (model.nq, model.nv, model.nu) != (28, 27, 21):
        raise ValueError(
            "Mini3 MJCF must compile to nq/nv/nu=28/27/21, got "
            f"{model.nq}/{model.nv}/{model.nu}"
        )


_validate_source_contract()

MINI3_INIT_STATE = InitialStateCfg(
    pos=(0.0, 0.0, 0.46305),
    joint_pos={name: 0.0 for name in MINI3_JOINT_NAMES},
    joint_vel={name: 0.0 for name in MINI3_JOINT_NAMES},
)

MINI3_CFG = AssetCfg(
    mjcf_path=MINI3_MJCF_PATH,
    usd_path=MINI3_URDF_PATH,
    init_state=MINI3_INIT_STATE,
    self_collisions=True,
    actuators=_real_motor_actuators(),
    mjlab_remove_xml_actuators=True,
    sensors_mjlab=[
        ContactSensorCfg(
            name="contact_forces",
            primary_contact_match_mode="subtree",
            primary_contact_match_pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            primary_contact_match_entity="robot",
            secondary_contact_match_mode="body",
            secondary_contact_match_pattern="terrain",
            track_air_time=True,
            history_length=4,
            reduce="netforce",
        ),
        ContactSensorCfg(
            name="self_collision",
            primary_contact_match_mode="subtree",
            primary_contact_match_pattern="base_link",
            primary_contact_match_entity="robot",
            secondary_contact_match_mode="subtree",
            secondary_contact_match_pattern="base_link",
            secondary_contact_match_entity="robot",
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    ],
    mjlab_collisions=[
        MjlabCollisionCfg(
            geom_names_expr=(".*_collision",),
            disable_other_geoms=False,
        ),
    ],
    joint_names_simulation=list(MINI3_JOINT_NAMES),
    body_names_simulation=list(MINI3_BODY_NAMES),
    joint_symmetry_mapping=symmetry_utils.mirrored(
        {
            "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
            "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
            "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
            "left_knee_pitch_joint": (1, "right_knee_pitch_joint"),
            "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
            "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
            "waist_yaw_joint": (-1, "waist_yaw_joint"),
            "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
            "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
            "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
            "left_elbow_pitch_joint": (1, "right_elbow_pitch_joint"),
        }
    ),
    spatial_symmetry_mapping=symmetry_utils.mirrored(
        {
            "base_link": "base_link",
            "imu_link": "imu_link",
            "left_hip_pitch_link": "right_hip_pitch_link",
            "left_hip_roll_link": "right_hip_roll_link",
            "left_hip_yaw_link": "right_hip_yaw_link",
            "left_knee_pitch_link": "right_knee_pitch_link",
            "left_ankle_pitch_link": "right_ankle_pitch_link",
            "left_ankle_roll_link": "right_ankle_roll_link",
            "waist_yaw_link": "waist_yaw_link",
            "head_link": "head_link",
            "left_shoulder_pitch_link": "right_shoulder_pitch_link",
            "left_shoulder_roll_link": "right_shoulder_roll_link",
            "left_shoulder_yaw_link": "right_shoulder_yaw_link",
            "left_elbow_pitch_link": "right_elbow_pitch_link",
        }
    ),
    strict_joint_contract=True,
)

registry.register("asset", "mini3-mesh", MINI3_CFG)
