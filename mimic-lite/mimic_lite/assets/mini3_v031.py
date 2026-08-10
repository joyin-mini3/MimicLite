from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco

from active_adaptation.assets.asset_cfg import InitialStateCfg
from active_adaptation.registry import Registry
from mimic_lite.assets.mini3 import MINI3_CFG, MINI3_JOINT_NAMES


registry = Registry.instance()

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MINI3_V031_ASSET_ROOT = REPOSITORY_ROOT / "any4hdmi" / "assets" / "robots" / "mini3_v031"
MINI3_V031_MJCF_PATH = MINI3_V031_ASSET_ROOT / "mjcf" / "mini3.xml"
MINI3_V031_URDF_PATH = MINI3_V031_ASSET_ROOT / "urdf" / "mini3.urdf"


def _validate_source_contract() -> None:
    for path in (MINI3_V031_MJCF_PATH, MINI3_V031_URDF_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Mini3 v0.3.1 asset file not found: {path}")

    model = mujoco.MjModel.from_xml_path(str(MINI3_V031_MJCF_PATH))
    joint_names = [
        model.joint(joint_id).name
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if joint_names != list(MINI3_JOINT_NAMES):
        raise ValueError(
            "Mini3 v0.3.1 MJCF joint order differs from the policy contract: "
            f"expected={list(MINI3_JOINT_NAMES)}, actual={joint_names}"
        )
    actuator_targets = [
        model.joint(int(model.actuator_trnid[actuator_id, 0])).name
        for actuator_id in range(model.nu)
    ]
    if actuator_targets != list(MINI3_JOINT_NAMES):
        raise ValueError(
            "Mini3 v0.3.1 actuator targets differ from the policy contract: "
            f"expected={list(MINI3_JOINT_NAMES)}, actual={actuator_targets}"
        )
    if (model.nq, model.nv, model.nu) != (28, 27, 21):
        raise ValueError(
            "Mini3 v0.3.1 MJCF must compile to nq/nv/nu=28/27/21, got "
            f"{model.nq}/{model.nv}/{model.nu}"
        )


_validate_source_contract()

MINI3_V031_INIT_STATE = InitialStateCfg(
    pos=(0.0, 0.0, 0.43625),
    joint_pos={name: 0.0 for name in MINI3_JOINT_NAMES},
    joint_vel={name: 0.0 for name in MINI3_JOINT_NAMES},
)

# The motor and symmetry contract is unchanged. Only source geometry and the
# neutral base height are version-specific; this variant is trained from new
# checkpoints and never aliases the legacy mini3 asset.
MINI3_V031_CFG = replace(
    MINI3_CFG,
    mjcf_path=MINI3_V031_MJCF_PATH,
    usd_path=MINI3_V031_URDF_PATH,
    init_state=MINI3_V031_INIT_STATE,
)

registry.register("asset", "mini3-v031-mesh", MINI3_V031_CFG)
