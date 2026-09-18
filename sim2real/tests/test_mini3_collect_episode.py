from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

import test_mini3_pick_carry_policy as fixtures
from mini3_collect_episode import ControlRecorder, collect_episode, onnx_session_threads


class Mini3ControlRecordingTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture = fixtures.PolicyTaskControlTest()
        fixture.setUp()
        self.scene = fixture.scene
        self.task = fixture.make_task()
        self.scene.state.motion_future_steps = np.array([-8, 0, 4])
        self.scene.policy.state_dict["command"] = np.arange(15, dtype=np.float32)
        self.scene.policy.state_dict["policy"] = np.arange(21, dtype=np.float32)

    def test_hook_records_pre_action_state_and_exact_commands_before_all_motor_substeps(self) -> None:
        scene, task = self.scene, self.task
        scene.extra_integral[3:] = [.05, -.04, .03]
        scene.extra_reference = scene.extra_target + [.01, -.01, .01, .02, -.02, .02]
        scene.collision_phase = "LOWER"
        scene.openings[:] = [.031, .024]
        policy_values = tuple(np.linspace(.01 + index, .21 + index, 21) for index in range(5))
        before_qpos, before_qvel = scene.data.qpos.copy(), scene.data.qvel.copy()
        recorder = ControlRecorder(task)
        calls = []

        def observe(runtime, commands):
            calls.append("observe")
            self.assertEqual(runtime.data.time, 0.)
            np.testing.assert_array_equal(runtime.data.qpos, before_qpos)
            np.testing.assert_array_equal(runtime.data.qvel, before_qvel)
            for name, expected in zip(("policy_q", "policy_dq", "policy_effort", "policy_kp", "policy_kd"),
                                      policy_values):
                np.testing.assert_array_equal(commands[name], expected)
                self.assertFalse(np.shares_memory(commands[name], expected))
                self.assertFalse(commands[name].flags.writeable)
            np.testing.assert_allclose(commands["extra_q"], scene.extra_target + 2 * scene.extra_integral)
            self.assertGreater(np.max(np.abs(commands["extra_q"] - scene.extra_target)), .04)
            recorder.observe(runtime, commands)

        scene.control_observer = observe

        def original_motor(q, position, velocity, *, target_vel, effort, kp, kd):
            self.assertEqual(calls[0], "observe")
            calls.append("original_motor")
            for actual, expected in zip((q, target_vel, effort, kp, kd), policy_values):
                np.testing.assert_array_equal(actual, expected)
            return np.zeros(21)

        def added_motor(q, position, velocity, *, effort):
            np.testing.assert_array_equal(q, recorder.rows[0]["extra_q"])
            np.testing.assert_array_equal(effort, recorder.rows[0]["extra_effort"])
            return np.zeros(6)

        def integrate(model, data):
            np.testing.assert_array_equal(data.ctrl[scene.grippers], recorder.rows[0]["gripper_opening"])
            data.qpos[scene.qids] += .0001
            data.qvel[scene.vids] += .001
            data.time += model.opt.timestep

        with ExitStack() as stack:
            stack.enter_context(patch.object(scene.policy, "step", return_value=policy_values))
            motor = stack.enter_context(patch.object(scene.motor, "compute", side_effect=original_motor))
            extra = stack.enter_context(patch.object(scene.extra_motor, "compute", side_effect=added_motor))
            physics = stack.enter_context(patch("mujoco.mj_step", side_effect=integrate))
            stack.enter_context(patch.object(scene, "check_arm_contacts"))
            scene.step()
        recorder.complete_step(scene)
        self.assertEqual(motor.call_count, 10)
        self.assertEqual(extra.call_count, 10)
        self.assertEqual(physics.call_count, 10)
        self.assertEqual(len(recorder.rows), 1)
        row = recorder.rows[0]
        np.testing.assert_array_equal(row["qpos"], before_qpos)
        np.testing.assert_array_equal(row["qvel"], before_qvel)
        np.testing.assert_array_equal(row["next_qpos"], scene.data.qpos)
        np.testing.assert_array_equal(row["next_qvel"], scene.data.qvel)
        self.assertEqual(row["time"], 0.)
        self.assertAlmostEqual(row["next_time"], .02)
        self.assertTrue(row["action_executed"])
        self.assertEqual(row["executed_substeps"], 10)
        np.testing.assert_array_equal(row["policy_input_command"], scene.policy.state_dict["command"])
        scene.reset()
        self.assertIsNone(scene.last_control_command)

    def test_gated_reference_and_proprioception_match_inputs_without_modifying_state(self) -> None:
        scene, task = self.scene, self.task
        task.reference.allowed_frame = 6
        task.reference.root_velocity_correction[:] = [.1, -.2, 0.]
        scene.state.motion_t[:] = 4
        scene.data.qpos[scene.rootq + 3:scene.rootq + 7] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
        scene.data.qvel[scene.rootv:scene.rootv + 6] = [1., 2., 3., .1, .2, .3]
        mujoco.mj_forward(scene.model, scene.data)
        before = scene.data.qpos.copy()
        recorder = ControlRecorder(task)
        recorder.observe(scene, {"extra_q": np.arange(6.)})
        row = recorder.rows[0]
        np.testing.assert_array_equal(row["reference_step"], [0, 4, 6])
        np.testing.assert_array_equal(row["reference_qpos"], task.reference.qpos[4])
        expected = task.reference.get_slice(np.array([0]), np.array([4]), np.array([-8, 0, 4]))
        for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "joint_pos"):
            np.testing.assert_array_equal(row["reference_" + name], getattr(expected, name)[0])
        np.testing.assert_allclose(row["root_linear_velocity_body"], [2., -1., 3.], atol=1e-12)
        np.testing.assert_array_equal(row["root_angular_velocity_body"], [.1, .2, .3])
        np.testing.assert_array_equal(scene.data.qpos, before)
        self.assertFalse(row["tool_target_active"])
        self.assertTrue(np.isnan(row["tool_target"]).all())

    def test_failed_partial_action_is_saved_and_marked_incomplete(self) -> None:
        scene, task = self.scene, self.task
        recorder = ControlRecorder(task)
        task.failure = "Collision stopped physics after three substeps"
        recorder.observe(scene, {"extra_q": np.arange(6.)})
        scene.data.time = .006
        scene.data.qpos[scene.qids[0]] += .01
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = recorder.save(output, wall_seconds=1.2, onnx_threads=1)
            with np.load(output / "control_samples.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["executed_substeps"].tolist(), [3])
                self.assertEqual(arrays["action_executed"].tolist(), [False])
                self.assertEqual(arrays["phase"].tolist(), ["APPROACH"])
                np.testing.assert_array_equal(arrays["terminal_qpos"], scene.data.qpos)
                np.testing.assert_array_equal(arrays["next_qpos"][0], scene.data.qpos)
                np.testing.assert_array_equal(arrays["reference_step_offsets"], [-8, 0, 4])
            self.assertFalse(metadata["success"])
            self.assertEqual(metadata["complete_actions"], 0)
            saved = json.loads((output / "control_metadata.json").read_text())
            self.assertEqual(saved["policy_joint_names"], scene.names)
            self.assertEqual(saved["reference_body_names"], task.reference.body_names)
            self.assertEqual(saved["policy_qpos_indices"], scene.qids.tolist())
            self.assertEqual(saved["failure"], task.failure)

    def test_inference_failure_does_not_emit_an_action_or_integrate(self) -> None:
        recorder = ControlRecorder(self.task)
        self.scene.control_observer = recorder.observe
        with patch.object(self.scene.policy, "step", return_value=None), patch("mujoco.mj_step") as physics:
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                self.scene.step()
        physics.assert_not_called()
        self.assertEqual(recorder.rows, [])
        self.assertIsNone(self.scene.last_control_command)

    def test_existing_attempt_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = output / "report.json"
            report.write_text('{"success": true}\n')
            with patch("mini3_collect_episode.limit_compute_threads") as limits:
                with self.assertRaises(FileExistsError):
                    collect_episode(seed=42, output=output)
            limits.assert_not_called()
            self.assertEqual(report.read_text(), '{"success": true}\n')

    def test_onnx_thread_override_is_limited_to_the_collector_context(self) -> None:
        import onnxruntime as ort
        original = ort.InferenceSession
        with patch.object(ort, "InferenceSession") as factory:
            with onnx_session_threads(2):
                ort.InferenceSession("unused.onnx", providers=["CPUExecutionProvider"])
            self.assertIs(ort.InferenceSession, factory)
            options = factory.call_args.args[1]
            self.assertEqual(options.intra_op_num_threads, 2)
            self.assertEqual(options.inter_op_num_threads, 1)
            self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)
        self.assertIs(ort.InferenceSession, original)


if __name__ == "__main__":
    unittest.main()
