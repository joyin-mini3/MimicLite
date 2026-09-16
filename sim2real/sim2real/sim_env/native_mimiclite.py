"""Desktop MuJoCo playback using the project's exported tracking-policy runtime."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from queue import SimpleQueue

import mujoco
import numpy as np

from sim2real.sim_env.integrated_sim2sim import IntegratedSim2Sim, IntegratedSim2SimArgs


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}; stopping MuJoCo playback")


class _Playback:
    """Advance policy and physics together; keep duration independent of resets."""

    def __init__(self, runtime: IntegratedSim2Sim, *, loop: bool, paused: bool):
        self.runtime = runtime
        self.loop = loop
        self.paused = paused
        self.finished = False
        self.sim_steps = 0
        self.policy_steps = 0
        self.episode_steps = 0
        self.reset_count = 0

    @property
    def elapsed(self) -> float:
        return self.sim_steps * self.runtime.args.sim_dt

    def reset(self) -> None:
        sim = self.runtime.sim
        mujoco.mj_resetData(sim.mj_model, sim.mj_data)
        sim.has_received_command = False
        sim.torques.fill(0.0)
        for name in ("cmd_q", "cmd_dq", "cmd_tau", "cmd_kp", "cmd_kd"):
            getattr(sim, name).fill(0.0)
        # Restores motion frame zero, motor state, observation histories and action.
        self.runtime._reset_playback()
        self.finished = False
        self.episode_steps = 0
        self.reset_count += 1

    def step(self, max_sim_steps: int | None = None) -> int:
        if self.paused:
            return 0
        if self.finished:
            if not self.loop:
                return 0
            self.reset()

        runtime = self.runtime
        sim = runtime.sim
        runtime._sync_policy_state_from_sim()
        command = runtime.policy.step()
        if command is None:
            raise RuntimeError("Policy inference failed; stopping MuJoCo playback")
        _require_finite("policy action", runtime.policy.state_dict["action"])
        command_names = (
            "target position", "target velocity", "feedforward torque", "kp", "kd"
        )
        for name, value in zip(command_names, command, strict=True):
            _require_finite(name, value)
        sim.apply_command(*command)
        runtime.policy.total_inference_cnt += 1
        self.policy_steps += 1

        steps = runtime.args.decimation
        if max_sim_steps is not None:
            steps = min(steps, max_sim_steps)
        advanced = 0
        for _ in range(steps):
            sim.sim_step()
            _require_finite("motor torque", sim.torques)
            _require_finite("qpos", sim.mj_data.qpos)
            _require_finite("qvel", sim.mj_data.qvel)
            self.sim_steps += 1
            self.episode_steps += 1
            advanced += 1
            runtime.headless_elapsed_s = self.episode_steps * runtime.args.sim_dt
            runtime._append_trajectory_frame()
            if runtime.args.stop_on_tracking_failure and runtime._tracking_failure_detected:
                break

        # MotionState.update increments before inference. Start after the first
        # frame-zero policy tick so a zero initial pause cannot skip that frame.
        runtime._maybe_start_motion(sim_elapsed_s=runtime.headless_elapsed_s)
        self.finished = (
            runtime.playback_started
            and int(runtime.state_processor.motion_t[0])
            >= int(runtime.state_processor.motion_length) - 1
        )
        return advanced


def _draw_reference(viewer, runtime: IntegratedSim2Sim) -> None:
    """Draw tracking-body positions without adding bodies to the physics model."""
    state = runtime.state_processor
    motion = state.motion_data
    current = int(np.flatnonzero(state.motion_future_steps == 0)[0])
    names = runtime.policy.policy_config["evaluation"]["tracking_body_names"]
    scene = viewer.user_scn
    scene.ngeom = 0
    for name in names:
        if name not in state.motion_body_names or scene.ngeom >= scene.maxgeom:
            continue
        body_index = state.motion_body_names.index(name)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, 0.018, dtype=np.float64),
            np.asarray(motion.body_pos_w[0, current, body_index], dtype=np.float64),
            np.eye(3).ravel(),
            np.asarray([0.05, 0.9, 1.0, 0.8], dtype=np.float32),
        )
        scene.ngeom += 1


def _close_viewer(viewer) -> None:
    viewer.close()
    # MuJoCo 3.11 close() only signals its daemon render thread. Exiting Python
    # before it destroys the GLFW window can crash; wait for its weak reference
    # to disappear, without keeping the native Simulate object alive ourselves.
    simulate_ref = getattr(viewer, "_sim", None)
    deadline = time.monotonic() + 3.0
    if simulate_ref is not None:
        while simulate_ref() is not None and time.monotonic() < deadline:
            time.sleep(0.01)


def run_native(
    args: IntegratedSim2SimArgs,
    *,
    duration: float,
    headless: bool,
    loop: bool,
    start_paused: bool = False,
) -> None:
    """Run CPU MuJoCo and ONNX inference, optionally with a native desktop viewer.

    ``duration`` is total simulated seconds across manual resets and motion loops;
    zero runs until the viewer closes (or the motion ends with ``loop=False``).
    """
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and non-negative")
    if headless and start_paused:
        raise ValueError("A headless run cannot start paused")
    if headless and duration == 0 and loop:
        raise ValueError("A looping headless run requires a positive duration")
    if not math.isfinite(args.initial_pause_s) or args.initial_pause_s < 0:
        raise ValueError("initial_pause_s must be finite and non-negative")

    # The shared runtime stays headless so it does not start a Viser server.
    runtime = IntegratedSim2Sim(replace(args, headless=True))
    playback = _Playback(runtime, loop=loop, paused=start_paused)
    key_events: SimpleQueue[int] = SimpleQueue()
    viewer = None
    follow = True
    limit = math.ceil(duration / args.sim_dt - 1e-9) if duration else None
    report_at = 1.0
    try:
        if not headless:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(
                runtime.sim.mj_model,
                runtime.sim.mj_data,
                key_callback=key_events.put,
            )
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = runtime.sim.pelvis_body_id
                viewer.cam.distance = 2.2
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -20.0
            print(
                "Keys: P pause/resume, R or 0 reset, F follow camera, Esc close. "
                "Cyan dots: reference bodies."
            )

        while runtime.sim.is_running() and (viewer is None or viewer.is_running()):
            started = time.perf_counter()
            close_requested = False
            while not key_events.empty():
                key = key_events.get()
                if key in (ord("P"), ord("p")):
                    playback.paused = not playback.paused
                    print("Paused" if playback.paused else "Resumed")
                elif key in (ord("R"), ord("r"), ord("0")):
                    playback.reset()
                elif key in (ord("F"), ord("f")):
                    follow = not follow
                elif key == 256:  # GLFW_KEY_ESCAPE
                    close_requested = True
            if close_requested or (limit is not None and playback.sim_steps >= limit):
                break

            remaining = None if limit is None else limit - playback.sim_steps
            playback.step(remaining)
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.type = (
                        mujoco.mjtCamera.mjCAMERA_TRACKING
                        if follow else mujoco.mjtCamera.mjCAMERA_FREE
                    )
                    _draw_reference(viewer, runtime)
                viewer.sync()

            if playback.elapsed >= report_at:
                reference_pos, _ = runtime._motion_root_state()
                address = runtime.sim.root_qpos_adr
                actual_pos = runtime.sim.mj_data.qpos[address : address + 3]
                error = float(np.linalg.norm(actual_pos - reference_pos))
                frame = int(runtime.state_processor.motion_t[0])
                print(
                    f"t={playback.elapsed:.2f}s "
                    f"frame={frame}/{runtime.state_processor.motion_length - 1} "
                    f"root_error={error:.3f}m"
                )
                report_at = math.floor(playback.elapsed) + 1.0
            if (playback.finished and not loop) or (
                args.stop_on_tracking_failure and runtime._tracking_failure_detected
            ):
                break
            if viewer is not None:
                time.sleep(max(0.0, args.env_dt - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            _close_viewer(viewer)
        runtime.policy.report_obs_profile()
        runtime._save_root_trajectory()
        runtime._save_trajectory()
        runtime.policy.save_recording()
        runtime.sim.stop()
        print(
            f"Finished: simulated={playback.elapsed:.3f}s "
            f"policy_steps={playback.policy_steps} physics_steps={playback.sim_steps} "
            f"resets={playback.reset_count}"
        )
