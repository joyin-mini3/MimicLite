from __future__ import annotations

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from sim2real.sim_env.integrated_sim2sim import IntegratedMotionState
from sim2real.sim_env.native_mimiclite import _Playback, _draw_reference, run_native


class _Sim:
    def __init__(self):
        self.mj_model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep="0.002" gravity="0 0 0"/>'
            '<worldbody><body><freejoint/><geom type="sphere" size="0.1"/>'
            '</body></worldbody></mujoco>'
        )
        self.mj_data = mujoco.MjData(self.mj_model)
        self.has_received_command = False
        for name in ("torques", "cmd_q", "cmd_dq", "cmd_tau", "cmd_kp", "cmd_kd"):
            setattr(self, name, np.zeros(1))

    def apply_command(self, *command):
        self.has_received_command = True
        names = ("cmd_q", "cmd_dq", "cmd_tau", "cmd_kp", "cmd_kd")
        for name, value in zip(names, command, strict=True):
            getattr(self, name)[:] = value

    def sim_step(self):
        mujoco.mj_step(self.mj_model, self.mj_data)


class _Runtime:
    def __init__(self, *, length=3, initial_pause=0.0):
        self.args = SimpleNamespace(
            sim_dt=0.002,
            env_dt=0.02,
            decimation=10,
            initial_pause_s=initial_pause,
            stop_on_tracking_failure=False,
        )
        self.sim = _Sim()
        self.state_processor = IntegratedMotionState.__new__(IntegratedMotionState)
        self.state_processor.motion_t = np.array([0])
        self.state_processor.motion_length = length
        self.state_processor._update_motion_data = lambda: None
        self.frames = []
        self.policy = SimpleNamespace(
            state_dict={"action": np.zeros(1), "paused": True},
            total_inference_cnt=0,
            step=self._policy_step,
        )
        self.playback_started = False
        self.headless_elapsed_s = 0.0
        self.saved_frames = 0
        self.resets = 0

    def _policy_step(self):
        self.state_processor.update(self.policy.state_dict)
        self.frames.append(int(self.state_processor.motion_t[0]))
        return tuple(np.zeros(1) for _ in range(5))

    def _sync_policy_state_from_sim(self):
        pass

    def _append_trajectory_frame(self):
        self.saved_frames += 1

    def _maybe_start_motion(self, *, sim_elapsed_s):
        if not self.playback_started and sim_elapsed_s >= self.args.initial_pause_s:
            self.playback_started = True
            self.policy.state_dict["paused"] = False

    def _reset_playback(self):
        self.state_processor.reset()
        self.policy.state_dict = {"action": np.zeros(1), "paused": True}
        self.playback_started = False
        self.headless_elapsed_s = 0.0
        self.resets += 1


class NativeMimicLiteTest(unittest.TestCase):
    def test_once_infers_frame_zero_and_each_frame_exactly_once(self):
        runtime = _Runtime()
        playback = _Playback(runtime, loop=False, paused=False)
        for _ in range(5):
            playback.step()
        self.assertEqual(runtime.frames, [0, 1, 2])
        self.assertTrue(playback.finished)
        self.assertEqual(playback.sim_steps, 30)
        self.assertEqual(runtime.saved_frames, 30)

    def test_initial_pause_advances_with_simulation_time(self):
        runtime = _Runtime(initial_pause=0.04)
        playback = _Playback(runtime, loop=False, paused=False)
        playback.step()
        self.assertFalse(runtime.playback_started)
        playback.step()
        self.assertTrue(runtime.playback_started)
        playback.step()
        self.assertEqual(runtime.frames, [0, 0, 1])

    def test_loop_resets_frame_without_resetting_total_duration(self):
        runtime = _Runtime(length=2)
        playback = _Playback(runtime, loop=True, paused=False)
        for _ in range(3):
            playback.step()
        self.assertEqual(runtime.frames, [0, 1, 0])
        self.assertEqual(runtime.resets, 1)
        self.assertAlmostEqual(playback.elapsed, 0.06)
        self.assertAlmostEqual(runtime.sim.mj_data.time, 0.02)

    def test_reset_clears_mujoco_warmstart_and_pending_motor_command(self):
        runtime = _Runtime()
        playback = _Playback(runtime, loop=True, paused=False)
        playback.step()
        runtime.sim.mj_data.qacc_warmstart[:] = 100.0
        runtime.sim.mj_data.qvel[:] = 5.0
        runtime.sim.cmd_q[:] = 1.0
        runtime.sim.torques[:] = 2.0
        playback.reset()
        self.assertFalse(runtime.sim.has_received_command)
        self.assertEqual(runtime.sim.mj_data.time, 0.0)
        np.testing.assert_array_equal(runtime.sim.mj_data.qacc_warmstart, 0.0)
        np.testing.assert_array_equal(runtime.sim.mj_data.qvel, 0.0)
        np.testing.assert_array_equal(runtime.sim.cmd_q, 0.0)
        np.testing.assert_array_equal(runtime.sim.torques, 0.0)
        self.assertEqual(playback.episode_steps, 0)
        self.assertEqual(playback.sim_steps, 10)
        self.assertEqual(runtime.resets, 1)

    def test_pause_freezes_physics_and_policy(self):
        runtime = _Runtime()
        playback = _Playback(runtime, loop=True, paused=True)
        self.assertEqual(playback.step(), 0)
        self.assertEqual(runtime.frames, [])
        self.assertEqual(runtime.sim.mj_data.time, 0.0)
        playback.paused = False
        self.assertEqual(playback.step(max_sim_steps=3), 3)
        self.assertAlmostEqual(playback.elapsed, 0.006)

    def test_nonfinite_action_stops_before_physics(self):
        runtime = _Runtime()
        runtime.policy.state_dict["action"][:] = np.nan
        playback = _Playback(runtime, loop=False, paused=False)
        with self.assertRaisesRegex(FloatingPointError, "policy action"):
            playback.step()
        self.assertEqual(runtime.sim.mj_data.time, 0.0)
        self.assertFalse(runtime.sim.has_received_command)

    def test_inference_failure_does_not_reuse_previous_motor_command(self):
        runtime = _Runtime()
        runtime.policy.step = lambda: None
        playback = _Playback(runtime, loop=False, paused=False)
        with self.assertRaisesRegex(RuntimeError, "Policy inference failed"):
            playback.step()
        self.assertEqual(playback.sim_steps, 0)

    def test_headless_pause_is_rejected_without_loading_a_model(self):
        with self.assertRaisesRegex(ValueError, "headless run cannot start paused"):
            run_native(None, duration=1.0, headless=True, loop=True, start_paused=True)

    def test_reference_markers_use_current_frame_and_body_name(self):
        runtime = _Runtime()
        state = runtime.state_processor
        state.motion_future_steps = np.array([-1, 0, 1])
        state.motion_body_names = ["unused", "base"]
        positions = np.arange(18, dtype=np.float64).reshape(1, 3, 2, 3)
        state.motion_data = SimpleNamespace(body_pos_w=positions)
        runtime.policy.policy_config = {
            "evaluation": {"tracking_body_names": ["base"]}
        }
        viewer = SimpleNamespace(
            user_scn=mujoco.MjvScene(runtime.sim.mj_model, maxgeom=5)
        )
        _draw_reference(viewer, runtime)
        self.assertEqual(viewer.user_scn.ngeom, 1)
        np.testing.assert_array_equal(viewer.user_scn.geoms[0].pos, positions[0, 1, 1])
        self.assertEqual(runtime.sim.mj_model.nbody, 2)


if __name__ == "__main__":
    unittest.main()
