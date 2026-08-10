from __future__ import annotations

from sim2real.config.robots.base import PROJECT_ROOT, RobotCfg
from sim2real.config.robots.mini3 import (
    MINI3_BODY_NAMES,
    MINI3_CFG,
    MINI3_JOINT_NAMES,
    _mapping,
)


MINI3_V031_CFG = RobotCfg(
    name="mini3-v031",
    joint_names=MINI3_JOINT_NAMES,
    body_names=MINI3_BODY_NAMES,
    joint_pos_lower_limit=_mapping(
        (
            -2.7925, -0.8, -2.7925, -1.0, -0.5236, -0.2618,
            -2.7925, -1.3963, -2.7925, -1.0, -0.5236, -0.2618,
            -2.7053,
            -3.6652, 0.0, -1.5708, -0.7854,
            -3.6652, -3.1416, -1.5708, -0.7854,
        )
    ),
    joint_pos_upper_limit=_mapping(
        (
            1.5, 1.3963, 2.7925, 2.3, 0.5236, 0.2618,
            1.5, 0.8, 2.7925, 2.3, 0.5236, 0.2618,
            2.7053,
            1.5708, 3.1416, 1.5708, 2.2,
            1.5708, 0.0, 1.5708, 2.2,
        )
    ),
    joint_velocity_limit=MINI3_CFG.joint_velocity_limit,
    joint_effort_limit=MINI3_CFG.joint_effort_limit,
    joint_armature=MINI3_CFG.joint_armature,
    joint_frictionloss=MINI3_CFG.joint_frictionloss,
    mjcf_path=PROJECT_ROOT.parent / "any4hdmi" / "assets" / "robots" / "mini3_v031" / "mjcf" / "mini3.xml",
    default_qpos=(0.0, 0.0, 0.43625, 1.0, 0.0, 0.0, 0.0, *([0.0] * 21)),
    publish_hz=MINI3_CFG.publish_hz,
    root_joint_names=MINI3_CFG.root_joint_names,
    viewer_track_body_names=MINI3_CFG.viewer_track_body_names,
    elastic_band_attach_body_names=MINI3_CFG.elastic_band_attach_body_names,
    strict_joint_contract=True,
    real_motor=MINI3_CFG.real_motor,
)
