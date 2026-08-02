from __future__ import annotations

from sim2real.config.robots.base import PROJECT_ROOT, RobotCfg


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


def _mapping(values: tuple[float, ...]) -> dict[str, float]:
    if len(values) != len(MINI3_JOINT_NAMES):
        raise ValueError(
            f"Mini3 parameter vector must have {len(MINI3_JOINT_NAMES)} values, got {len(values)}"
        )
    return dict(zip(MINI3_JOINT_NAMES, values, strict=True))


MINI3_CFG = RobotCfg(
    name="mini3",
    joint_names=MINI3_JOINT_NAMES,
    body_names=MINI3_BODY_NAMES,
    joint_pos_lower_limit=_mapping(
        (
            -1.8326, -0.8727, -1.9199, -0.3491, -0.7, -0.518,
            -1.8326, -1.3963, -1.9199, -0.3491, -0.7, -0.518,
            -2.618,
            -3.4907, -0.1745, -1.5708, -0.576,
            -3.4907, -3.1416, -1.5708, -0.576,
        )
    ),
    joint_pos_upper_limit=_mapping(
        (
            1.8326, 1.3963, 1.9199, 1.9199, 0.7, 0.518,
            1.8326, 0.8727, 1.9199, 1.9199, 0.7, 0.518,
            2.618,
            1.0472, 3.1416, 1.5708, 3.8397,
            1.0472, 0.1745, 1.5708, 3.8397,
        )
    ),
    joint_velocity_limit=_mapping(
        (
            10.0, 10.0, 10.0, 10.0, 45.0, 45.0,
            10.0, 10.0, 10.0, 10.0, 45.0, 45.0,
            10.0,
            45.0, 45.0, 45.0, 45.0,
            45.0, 45.0, 45.0, 45.0,
        )
    ),
    joint_effort_limit=_mapping(
        (
            27.0, 27.0, 27.0, 27.0, 25.0, 25.0,
            27.0, 27.0, 27.0, 27.0, 25.0, 25.0,
            27.0,
            12.5, 12.5, 12.5, 12.5,
            12.5, 12.5, 12.5, 12.5,
        )
    ),
    joint_armature=_mapping(
        (
            0.04595206677913666, 0.04595206677913666, 0.04595206677913666,
            0.04595206677913666, 0.01, 0.01,
            0.04595206677913666, 0.04595206677913666, 0.04595206677913666,
            0.04595206677913666, 0.01, 0.01,
            0.04595206677913666,
            0.0019, 0.0019, 0.0019, 0.0019,
            0.0019, 0.0019, 0.0019, 0.0019,
        )
    ),
    joint_frictionloss=_mapping(
        (
            1.0523104667663574, 1.0523104667663574, 1.0523104667663574,
            1.0523104667663574, 0.7, 0.7,
            1.0523104667663574, 1.0523104667663574, 1.0523104667663574,
            1.0523104667663574, 0.7, 0.7,
            1.0523104667663574,
            0.7, 0.7, 0.7, 0.7,
            0.7, 0.7, 0.7, 0.7,
        )
    ),
    mjcf_path=PROJECT_ROOT.parent / "any4hdmi" / "assets" / "robots" / "mini3_mjlab" / "mini3.xml",
    default_qpos=(
        0.0, 0.0, 0.46305,
        1.0, 0.0, 0.0, 0.0,
        *([0.0] * 21),
    ),
    publish_hz=50.0,
    root_joint_names=("floating_base",),
    viewer_track_body_names=("base_link",),
    elastic_band_attach_body_names=("waist_yaw_link", "base_link"),
    strict_joint_contract=True,
)
