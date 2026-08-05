from __future__ import annotations

from collections.abc import Sequence

import torch

from any4hdmi.dataset.base import BaseDataset, MotionData, MotionSample
from any4hdmi.dataset.fk_cache import FKCacheEntry


_MOTION_DATA_FIELD_NAMES = (
    "motion_id",
    "step",
    "body_pos_w",
    "body_lin_vel_w",
    "body_quat_w",
    "body_ang_vel_w",
    "joint_pos",
    "joint_vel",
)
_FLOAT_MOTION_DATA_FIELD_NAMES = _MOTION_DATA_FIELD_NAMES[2:]
_INACTIVE_MOTION_LEN = 1 << 30


class SequentialWindowedMotionDataset(BaseDataset):
    """Stream complete motions through fixed per-environment device windows.

    This runtime is evaluator-only. ``sample_motion`` consumes a deterministic
    source-motion queue and returns environment route IDs, while ``get_slice``
    accepts absolute source-motion time. Moving a window never changes the
    route ID or resets simulator, policy, or actuator state.
    """

    dataset_kind = "sequential_windowed"

    def __init__(
        self,
        *,
        entry: FKCacheEntry,
        num_envs: int,
        window_frames: int = 512,
        device: torch.device | str | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if window_frames <= 1:
            raise ValueError(
                f"window_frames must be greater than one, got {window_frames}"
            )
        if not entry.motion_paths:
            raise ValueError("SequentialWindowedMotionDataset requires motions")

        self.body_names = list(entry.body_names)
        self.joint_names = list(entry.joint_names)
        self.motion_paths = list(entry.motion_paths)
        self._storage_cpu = dict(entry.storage_fields)
        self._storage_starts_cpu = torch.as_tensor(entry.starts, dtype=torch.long)
        self._storage_ends_cpu = torch.as_tensor(entry.ends, dtype=torch.long)
        self._storage_lengths_cpu = (
            self._storage_ends_cpu - self._storage_starts_cpu
        )
        self._storage_total_length = int(self._storage_cpu["motion_id"].shape[0])
        self._num_envs = int(num_envs)
        self.window_frames = int(window_frames)
        self._window_offsets_cpu = torch.arange(self.window_frames, dtype=torch.long)
        self._motion_queue: list[int] = []
        self._queue_cursor = 0
        self.to(device if device is not None else torch.device("cpu"))

    @classmethod
    def from_cache_entry(
        cls,
        entry: FKCacheEntry,
        *,
        num_envs: int,
        window_frames: int = 512,
        device: torch.device | str | None = None,
    ) -> SequentialWindowedMotionDataset:
        return cls(
            entry=entry,
            num_envs=num_envs,
            window_frames=window_frames,
            device=device,
        )

    @property
    def num_steps(self) -> int:
        return self._storage_total_length

    @property
    def sample_id_span(self) -> int:
        return self._num_envs

    @property
    def env_source_motion_ids(self) -> torch.Tensor:
        return self._env_source_motion_id

    @property
    def env_active(self) -> torch.Tensor:
        return self._env_active

    @property
    def pending_count(self) -> int:
        return len(self._motion_queue) - self._queue_cursor

    @property
    def active_count(self) -> int:
        return int(self._env_active.sum().item())

    @property
    def evaluation_complete(self) -> bool:
        return self.pending_count == 0 and self.active_count == 0

    def set_motion_queue(self, motion_ids: Sequence[int]) -> None:
        queue = [int(motion_id) for motion_id in motion_ids]
        invalid = [
            motion_id
            for motion_id in queue
            if motion_id < 0 or motion_id >= self.num_motions
        ]
        if invalid:
            raise ValueError(
                "Sequential evaluation motion IDs are out of range: "
                f"{invalid[:8]}"
            )
        if len(queue) != len(set(queue)):
            raise ValueError("Sequential evaluation motion IDs must be unique")
        if self.active_count:
            raise RuntimeError("Cannot replace the motion queue while envs are active")
        self._motion_queue = queue
        self._queue_cursor = 0

    def _allocate_window_pool(self, device: torch.device) -> MotionData:
        shape = (self._num_envs, self.window_frames)
        body_count = len(self.body_names)
        joint_count = len(self.joint_names)
        pool = MotionData(
            motion_id=torch.full(shape, -1, dtype=torch.long, device=device),
            step=torch.zeros(shape, dtype=torch.long, device=device),
            body_pos_w=torch.empty(
                (*shape, body_count, 3), dtype=torch.float32, device=device
            ),
            body_lin_vel_w=torch.empty(
                (*shape, body_count, 3), dtype=torch.float32, device=device
            ),
            body_quat_w=torch.empty(
                (*shape, body_count, 4), dtype=torch.float32, device=device
            ),
            body_ang_vel_w=torch.empty(
                (*shape, body_count, 3), dtype=torch.float32, device=device
            ),
            joint_pos=torch.empty(
                (*shape, joint_count), dtype=torch.float32, device=device
            ),
            joint_vel=torch.empty(
                (*shape, joint_count), dtype=torch.float32, device=device
            ),
            batch_size=shape,
            device=device,
        )
        pool.zero_()
        pool.body_pos_w[..., 2] = 1.0
        pool.body_quat_w[..., 0] = 1.0
        return pool

    def to(self, device: torch.device | str) -> SequentialWindowedMotionDataset:
        target_device = torch.device(device)
        self.device = target_device
        self.starts = self._storage_starts_cpu.to(target_device)
        self.ends = self._storage_ends_cpu.to(target_device)
        self.lengths = self._storage_lengths_cpu.to(target_device)
        self._current_window = self._allocate_window_pool(target_device)
        self._env_source_motion_id = torch.full(
            (self._num_envs,), -1, dtype=torch.long, device=target_device
        )
        self._env_motion_len = torch.ones(
            (self._num_envs,), dtype=torch.long, device=target_device
        )
        self._env_window_start = torch.zeros(
            (self._num_envs,), dtype=torch.long, device=target_device
        )
        self._env_active = torch.zeros(
            (self._num_envs,), dtype=torch.bool, device=target_device
        )

        # RobotTracking materializes reference buffers during construction,
        # before the evaluator installs its queue. A valid placeholder makes
        # those reads safe; the first environment reset replaces every route.
        all_env_ids = torch.arange(self._num_envs, device=target_device)
        self._env_source_motion_id.fill_(0)
        self._load_windows(all_env_ids, torch.zeros_like(all_env_ids))
        return self

    def _load_windows(
        self, env_ids: torch.Tensor, window_starts: torch.Tensor
    ) -> None:
        if env_ids.numel() == 0:
            return
        env_ids_device = env_ids.to(device=self.device, dtype=torch.long)
        source_ids_device = self._env_source_motion_id.index_select(
            0, env_ids_device
        )
        source_ids_cpu = source_ids_device.detach().to(device="cpu")
        window_starts_cpu = window_starts.detach().to(
            device="cpu", dtype=torch.long
        )

        source_lengths = self._storage_lengths_cpu.index_select(0, source_ids_cpu)
        max_window_starts = (source_lengths - self.window_frames).clamp_min(0)
        window_starts_cpu = torch.minimum(
            window_starts_cpu.clamp_min(0), max_window_starts
        )
        global_starts = self._storage_starts_cpu.index_select(0, source_ids_cpu)
        global_indices = (
            global_starts.add(window_starts_cpu).unsqueeze(1)
            + self._window_offsets_cpu.unsqueeze(0)
        )
        global_ends = (
            self._storage_ends_cpu.index_select(0, source_ids_cpu).sub(1).unsqueeze(1)
        )
        flat_indices = torch.minimum(global_indices, global_ends).reshape(-1)
        batch_size = int(env_ids_device.numel())

        for field_name in _MOTION_DATA_FIELD_NAMES:
            source = self._storage_cpu[field_name]
            field = source.index_select(0, flat_indices).reshape(
                batch_size, self.window_frames, *source.shape[1:]
            )
            destination = getattr(self._current_window, field_name)
            destination.index_copy_(
                0,
                env_ids_device,
                field.to(
                    device=self.device,
                    dtype=(
                        torch.float32
                        if field_name in _FLOAT_MOTION_DATA_FIELD_NAMES
                        else field.dtype
                    ),
                ),
            )

        self._env_window_start.index_copy_(
            0,
            env_ids_device,
            window_starts_cpu.to(device=self.device),
        )

    def _draw_next_sources(
        self, count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end = min(self._queue_cursor + count, len(self._motion_queue))
        selected = self._motion_queue[self._queue_cursor : end]
        self._queue_cursor = end
        active_count = len(selected)
        if active_count < count:
            selected.extend([0] * (count - active_count))
        source_ids = torch.as_tensor(selected, dtype=torch.long, device=self.device)
        active = torch.arange(count, device=self.device) < active_count
        return source_ids, active

    def sample_motion(
        self,
        env_ids: torch.Tensor,
        *,
        terminated_t: torch.Tensor,
        rewind_mask: torch.Tensor,
        rewind_steps: torch.Tensor,
    ) -> MotionSample:
        del terminated_t, rewind_steps
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        rewind_mask = rewind_mask.to(device=self.device, dtype=torch.bool)
        if bool(torch.any(rewind_mask).item()):
            raise RuntimeError("Sequential evaluation does not support rewind")
        if env_ids.numel() == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return MotionSample(motion_id=empty, motion_len=empty, start_t=empty)

        source_ids, active = self._draw_next_sources(int(env_ids.numel()))
        self._env_source_motion_id.index_copy_(0, env_ids, source_ids)
        self._env_active.index_copy_(0, env_ids, active)
        source_lengths = self.lengths.index_select(0, source_ids).long()
        motion_lengths = torch.where(
            active,
            source_lengths,
            torch.full_like(source_lengths, _INACTIVE_MOTION_LEN),
        )
        self._env_motion_len.index_copy_(0, env_ids, motion_lengths)
        self._load_windows(env_ids, torch.zeros_like(env_ids))
        return MotionSample(
            motion_id=env_ids,
            motion_len=motion_lengths,
            start_t=torch.zeros_like(env_ids),
        )

    def get_slice(
        self,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        steps: torch.Tensor,
        *,
        profile_name: str | None = None,
    ) -> MotionData:
        del profile_name
        route_ids = motion_ids.to(device=self.device, dtype=torch.long)
        starts = starts.to(device=self.device, dtype=torch.long)
        steps = steps.to(device=self.device, dtype=torch.long)
        if route_ids.numel() == 0:
            return self._current_window[route_ids.unsqueeze(1), steps.unsqueeze(0)]
        if bool(torch.any(self._env_source_motion_id[route_ids] < 0).item()):
            raise RuntimeError("Sequential evaluation route has no source motion")

        source_ids = self._env_source_motion_id.index_select(0, route_ids)
        source_lengths = self.lengths.index_select(0, source_ids).long()
        absolute_indices = starts.unsqueeze(1) + steps.unsqueeze(0)
        absolute_indices.clamp_min_(0)
        absolute_indices = torch.minimum(
            absolute_indices,
            source_lengths.sub(1).unsqueeze(1),
        )
        requested_min = absolute_indices.min(dim=1).values
        requested_max = absolute_indices.max(dim=1).values
        if bool(torch.any(requested_max - requested_min >= self.window_frames).item()):
            raise ValueError(
                "Requested reference span does not fit the sequential window; "
                f"window_frames={self.window_frames}, steps={steps.tolist()}"
            )

        window_start = self._env_window_start.index_select(0, route_ids)
        window_end = window_start + self.window_frames - 1
        reload_mask = (requested_min < window_start) | (requested_max > window_end)
        if bool(torch.any(reload_mask).item()):
            reload_positions = torch.nonzero(
                reload_mask, as_tuple=False
            ).squeeze(-1)
            reload_env_ids = route_ids.index_select(0, reload_positions)
            new_starts = requested_min.index_select(0, reload_positions)
            self._load_windows(reload_env_ids, new_starts)
            window_start = self._env_window_start.index_select(0, route_ids)

        local_indices = absolute_indices - window_start.unsqueeze(1)
        return self._current_window[route_ids.unsqueeze(1), local_indices]
