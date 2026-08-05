from mimic_lite.tasks.command import RobotTracking
from mimic_lite.tasks.actions import JointPosition

from active_adaptation.envs.mdp.observations.base import Observation as BaseObservation
from active_adaptation.envs.utils import find_bodies

import torch
from typing import cast, List

TrackObservation = BaseObservation[RobotTracking]


def _select_available_body_names(
    asset,
    available_body_names: list[str],
    body_names: List[str] | str,
) -> tuple[list[int], list[str]]:
    _, matched_body_names = find_bodies(asset, body_names)
    matched_name_set = set(matched_body_names)
    selected_body_names = [
        body_name for body_name in available_body_names if body_name in matched_name_set
    ]
    if not selected_body_names:
        raise ValueError("No tracking body matched for observation.")
    selected_body_indices = [
        available_body_names.index(body_name) for body_name in selected_body_names
    ]
    return selected_body_indices, selected_body_names


class _tracking_future_step_observation(TrackObservation):
    def __init__(
        self,
        env,
        future_steps: List[int] | int | None = None,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        if future_steps is None:
            future_steps = self.command_manager.future_steps.tolist()
        elif isinstance(future_steps, int):
            future_steps = [future_steps]

        available_future_steps = [
            int(step) for step in self.command_manager.future_steps.tolist()
        ]
        future_step_indices = []
        for step in future_steps:
            step = int(step)
            if step not in available_future_steps:
                raise ValueError(
                    f"future step {step} not in command.future_steps={available_future_steps}"
                )
            future_step_indices.append(available_future_steps.index(step))

        self.future_step_indices = torch.as_tensor(
            future_step_indices, dtype=torch.long, device=self.device
        )

    def _select_future_steps(self, x: torch.Tensor) -> torch.Tensor:
        return torch.index_select(x, 1, self.future_step_indices)


class ref_joint_pos_future(_tracking_future_step_observation, namespace="mimic_lite"):
    def __init__(self, env, noise_std=0.0, **kwargs):
        super().__init__(env, **kwargs)
        self.noise_std = noise_std
        
    def compute(self):
        joint_pos = self._select_future_steps(
            self.command_manager.ref_joint_pos_future_
        ).reshape(self.num_envs, -1)
        if self.noise_std > 0.0:
            joint_pos += (
                torch.randn_like(joint_pos).clamp(-3.0, 3.0) * self.noise_std
            )
        return joint_pos


class ref_joint_vel_future(_tracking_future_step_observation, namespace="mimic_lite"):
    def compute(self):
        return self._select_future_steps(
            self.command_manager.ref_joint_vel_future_
        ).reshape(self.num_envs, -1)


class ref_joint_action(TrackObservation, namespace="mimic_lite"):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        action_manager = cast(JointPosition, self.env.action_manager)
        self.action_joint_ids = action_manager.joint_ids
        self.action_indices_motion = [
            self.command_manager.dataset.joint_names.index(joint_name)
            for joint_name in action_manager.joint_names
        ]

        self.action_scaling = action_manager.action_scaling
        self.default_joint_pos = action_manager.default_joint_pos[
            :, self.action_joint_ids
        ]

    def compute(self):
        ref_joint_pos = self.command_manager.current_ref_motion.joint_pos[
            :, self.action_indices_motion
        ]
        ref_joint_action = (
            ref_joint_pos - self.default_joint_pos
        ) / self.action_scaling
        return ref_joint_action

# root_diff_obs

class ref_root_pos_future_b(TrackObservation, namespace="mimic_lite"):
    """
    Reference root position in robot root frame
    """

    def compute(self):
        return self.command_manager.ref_root_pos_future_b.view(self.num_envs, -1)


class ref_root_ori_future_b(_tracking_future_step_observation, namespace="mimic_lite"):
    """
    Reference root orientation in robot root frame
    """

    def __init__(self, env, noise_std=0.0, **kwargs):
        super().__init__(env, **kwargs)
        self.noise_std = noise_std

    def compute(self):
        ref_root_ori_future_b = self._select_future_steps(
            self.command_manager.ref_root_ori_future_b_matrix
        )
        if self.noise_std > 0.0:
            ref_root_ori_future_b = ref_root_ori_future_b.clone()
            ref_root_ori_future_b += (
                torch.randn_like(ref_root_ori_future_b).clamp(-3.0, 3.0) * self.noise_std
            )
        return ref_root_ori_future_b[:, :, :2, :].reshape(self.num_envs, -1)


class ref_root_lin_vel_future_local(
    _tracking_future_step_observation, namespace="mimic_lite"
):
    """Reference-root linear velocity in the current reference yaw frame."""

    def __init__(self, env, noise_std=0.0, **kwargs):
        super().__init__(env, **kwargs)
        self.noise_std = noise_std

    def compute(self):
        ref_root_lin_vel_future_local = self._select_future_steps(
            self.command_manager.ref_root_lin_vel_future_local
        ).reshape(self.num_envs, -1)
        if self.noise_std > 0.0:
            ref_root_lin_vel_future_local += (
                torch.randn_like(ref_root_lin_vel_future_local).clamp(-3.0, 3.0)
                * self.noise_std
            )
        return ref_root_lin_vel_future_local


# motion_local_obs

class _tracking_body_future_observation(TrackObservation):
    available_body_names_attr = "tracking_body_names"
    available_future_steps_attr = "future_steps"

    def __init__(
        self,
        env,
        body_names: List[str] | str | None = None,
        future_steps: List[int] | int | None = None,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        available_body_names = list(
            getattr(self.command_manager, self.available_body_names_attr)
        )
        if body_names is None:
            body_names = available_body_names
        if future_steps is None:
            future_steps = getattr(
                self.command_manager, self.available_future_steps_attr
            ).tolist()
        elif isinstance(future_steps, int):
            future_steps = [future_steps]

        body_indices_tracking, matched_body_names = _select_available_body_names(
            self.command_manager.asset,
            available_body_names,
            body_names,
        )

        available_future_steps = [
            int(step)
            for step in getattr(
                self.command_manager, self.available_future_steps_attr
            ).tolist()
        ]
        future_step_indices = []
        for step in future_steps:
            step = int(step)
            if step not in available_future_steps:
                raise ValueError(
                    f"future step {step} not in command.{self.available_future_steps_attr}={available_future_steps}"
                )
            future_step_indices.append(available_future_steps.index(step))

        self.body_indices_tracking = torch.as_tensor(
            body_indices_tracking, dtype=torch.long, device=self.device
        )
        self.future_step_indices = torch.as_tensor(
            future_step_indices, dtype=torch.long, device=self.device
        )

    def _select_body_future(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.index_select(x, 1, self.future_step_indices)
        return torch.index_select(x, 2, self.body_indices_tracking)


class _motion_local_body_future_observation(_tracking_body_future_observation):
    available_body_names_attr = "obs_body_names"


class ref_body_pos_future_local(
    _motion_local_body_future_observation, namespace="mimic_lite"
):
    """
    Reference body position in the projected-yaw anchor frame.
    """
    def __init__(self, env, noise_std=0.0, **kwargs):
        super().__init__(env, **kwargs)
        self.noise_std = noise_std

    def compute(self):
        ref_body_pos_future_local = self._select_body_future(
            self.command_manager.ref_body_pos_future_local
        ).reshape(self.num_envs, -1)
        if self.noise_std > 0.0:
            ref_body_pos_future_local += (
                torch.randn_like(ref_body_pos_future_local).clamp(-3.0, 3.0) * self.noise_std
            )
        return ref_body_pos_future_local


class ref_body_ori_future_local(
    _motion_local_body_future_observation, namespace="mimic_lite"
):
    """
    Reference body orientation in the projected-yaw anchor frame.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.ref_body_ori_future_local_matrix
        )[:, :, :, :2, :].reshape(self.num_envs, -1)

# body_local_diff_obs

class _diff_body_future_observation(_tracking_body_future_observation):
    available_future_steps_attr = "diff_future_steps"


class diff_body_pos_future_local(
    _diff_body_future_observation, namespace="mimic_lite"
):
    """
    Reference body position in the projected-yaw anchor frame minus robot body position in the projected-yaw anchor frame.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.diff_body_pos_future_local
        ).reshape(self.num_envs, -1)


class diff_body_lin_vel_future(
    _diff_body_future_observation, namespace="mimic_lite"
):
    """
    Reference body linear velocity minus robot body linear velocity.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.diff_body_lin_vel_future
        ).reshape(self.num_envs, -1)


class diff_body_ori_future_local(
    _diff_body_future_observation, namespace="mimic_lite"
):
    """
    Reference body orientation in the projected-yaw anchor frame minus robot body orientation in the projected-yaw anchor frame.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.diff_body_ori_future_local_matrix
        )[:, :, :, :2, :].reshape(self.num_envs, -1)


class diff_body_ang_vel_future(
    _diff_body_future_observation, namespace="mimic_lite"
):
    """
    Reference body angular velocity minus robot body angular velocity.
    """

    def compute(self):
        return self._select_body_future(
            self.command_manager.diff_body_ang_vel_future
        ).reshape(self.num_envs, -1)


class ref_motion_phase(TrackObservation, namespace="mimic_lite"):
    def compute(self):
        return (self.command_manager.obs_motion_t / self.command_manager.motion_len).unsqueeze(1)


class motion_length(TrackObservation, namespace="mimic_lite"):
    def compute(self):
        return self.command_manager.motion_len.to(torch.float32).unsqueeze(1)


class tracking_filter_diagnostics(TrackObservation, namespace="mimic_lite"):
    """Current tracking errors used to explain checkpoint-filter failures.

    The stable columns contain the termination-frame tracking errors, argmax
    indices, and reference/robot root poses used in the filter artifact.
    """

    columns = (
        "motion_t",
        "root_pos_error",
        "root_ori_error",
        "body_pos_error_local_max",
        "body_pos_error_local_argmax",
        "body_ori_error_local_max",
        "body_ori_error_local_argmax",
        "joint_pos_error_max",
        "joint_pos_error_argmax",
        "applied_action_abs_max",
        "applied_torque_abs_max",
        "reference_root_pos_x",
        "reference_root_pos_y",
        "reference_root_pos_z",
        "reference_root_quat_w",
        "reference_root_quat_x",
        "reference_root_quat_y",
        "reference_root_quat_z",
        "robot_root_pos_x",
        "robot_root_pos_y",
        "robot_root_pos_z",
        "robot_root_quat_w",
        "robot_root_quat_x",
        "robot_root_quat_y",
        "robot_root_quat_z",
    )

    def __init__(self, env, root_body_name: str, **kwargs):
        super().__init__(env, **kwargs)
        self.asset = self.command_manager.asset
        self.action_manager = cast(JointPosition, self.env.action_manager)
        try:
            self.root_body_index = self.command_manager.tracking_body_names.index(
                root_body_name
            )
        except ValueError as error:
            raise ValueError(
                f"Filter diagnostic root body {root_body_name!r} is not tracked"
            ) from error

    def compute(self):
        body_pos_max, body_pos_argmax = self.command_manager.body_pos_error_local.max(
            dim=1
        )
        body_ori_max, body_ori_argmax = self.command_manager.body_ori_error_local.max(
            dim=1
        )
        joint_pos_max, joint_pos_argmax = self.command_manager.joint_pos_error.max(
            dim=1
        )
        applied_action_abs_max = self.action_manager.applied_action.abs().max(
            dim=1
        ).values
        if self.env.backend == "isaaclab":
            applied_torque = self.asset.data.applied_torque
        else:
            applied_torque = self.asset.data.actuator_force
        applied_torque_abs_max = applied_torque[
            :, self.command_manager.tracking_joint_indices_asset
        ].abs().max(dim=1).values
        return torch.stack(
            (
                self.command_manager.obs_motion_t.to(torch.float32),
                self.command_manager.body_pos_error[:, self.root_body_index],
                self.command_manager.body_ori_error[:, self.root_body_index],
                body_pos_max,
                body_pos_argmax.to(torch.float32),
                body_ori_max,
                body_ori_argmax.to(torch.float32),
                joint_pos_max,
                joint_pos_argmax.to(torch.float32),
                applied_action_abs_max,
                applied_torque_abs_max,
                *self.command_manager.ref_body_pos_w[
                    :, self.root_body_index
                ].unbind(dim=1),
                *self.command_manager.ref_body_quat_w[
                    :, self.root_body_index
                ].unbind(dim=1),
                *self.command_manager.robot_body_link_pos_w[
                    :, self.root_body_index
                ].unbind(dim=1),
                *self.command_manager.robot_body_link_quat_w[
                    :, self.root_body_index
                ].unbind(dim=1),
            ),
            dim=1,
        )
