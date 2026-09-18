from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

import mujoco
import numpy as np

from test_mini3_pick_carry import PolicyDouble, ROOT, compatible_policy_config
from mini3_pick_carry_policy import AddedJointPlanner, GatedReference, PolicyOnlyScene, PolicyPickCarryTask


SCENE = ROOT / "any4hdmi/assets/robots/mini3_mjlab/scene_pick_carry_7dof.xml"


class PolicyTaskControlTest(unittest.TestCase):
    def setUp(self) -> None:
        with patch("sim2real.sim_env.integrated_sim2sim.IntegratedPolicyRuntime", PolicyDouble):
            with patch("builtins.open", mock_open(read_data=json.dumps(compatible_policy_config()))):
                self.scene = PolicyOnlyScene(SCENE, Path("unused.yaml"), Path("unused.npz"))

    def make_task(self):
        """A short synthetic reference isolates gates from private policy data."""
        scene, state = self.scene, self.scene.state
        state.joint_names = scene.names
        state.motion_ids = np.zeros(1, dtype=np.int64)
        state.motion_t = np.zeros(1, dtype=np.int64)
        state._update_motion_data = lambda: None
        pose = np.r_[scene.data.qpos[:7], scene.data.qpos[scene.qids]]
        poses = np.tile(pose, (151, 1))
        # Different poses beyond each gate make leaked future samples observable.
        poses[50:, scene.names.index("right_elbow_pitch_joint") + 7] += .1
        poses[100:, 0] += .1
        phases = {"APPROACH": 0., "CLEAR_ARM": .1, "REACH": .2, "LOWER": .3,
                  "CLOSE": .5, "LIFT": 1., "CARRY": 2., "PLACE": 2.2,
                  "RELEASE": 2.5, "SUCCESS": 3.}

        def sample(time):
            phase = max((key for key in phases if phases[key] <= time + 1e-9),
                        key=phases.get)
            return {"qpos": poses[min(round(time / .02), 150)].copy(),
                    "extra_command": scene.stow_angles.copy(), "body_source_time": time,
                    "arm_source_time": time, "phase": phase}

        plan = SimpleNamespace(qpos=poses, duration=3., sample=sample,
                               sample_source_qpos=lambda time: scene.data.qpos.copy(),
                               events=[{"phase": key, "time": value} for key, value in phases.items()])
        with patch("builtins.print"):
            return PolicyPickCarryTask(scene, plan)

    def test_every_original_command_channel_passes_through_all_ten_motor_substeps(self) -> None:
        scene = self.scene
        policy_values = tuple(np.linspace(.01 + index, .21 + index, 21)
                              for index in range(5))
        for array in policy_values:
            array.flags.writeable = False
        before_values = [value.copy() for value in policy_values]
        reference_pos = np.arange(84, dtype=float).reshape(4, 21) / 100
        reference_vel = np.full((4, 21), .17)
        scene.state.motion_dataset = SimpleNamespace(
            _storage={"joint_pos": reference_pos, "joint_vel": reference_vel}, num_steps=4)
        reference_before = reference_pos.copy()
        scene.arm_target = np.ones(3) * 100  # Legacy override state must be ignored.
        scene.arm_command = np.ones(7) * 100
        scene.arm_integral[:] = .12
        scene.arm_kp, scene.arm_kd = 1000, 1000
        scene.tool_target = scene.ik.position(scene.data).copy()
        planned_extra = scene.data.qpos[scene.ik.qids] + [.01, -.02, .03]
        observations = []

        def motor(q, position, velocity, *, target_vel, effort, kp, kd):
            for actual, expected in zip((q, target_vel, effort, kp, kd), policy_values):
                np.testing.assert_array_equal(actual, expected)
            observations.append((position.copy(), velocity.copy()))
            np.testing.assert_array_equal(position, scene.data.qpos[scene.qids])
            np.testing.assert_array_equal(velocity, scene.data.qvel[scene.vids])
            return np.linspace(-.1, .1, 21)

        def integrate(model, data):
            data.qpos[scene.qids] += .0001
            data.qvel[scene.vids] += .001
            data.time += model.opt.timestep

        with ExitStack() as stack:
            inference = stack.enter_context(patch.object(scene.policy, "step", return_value=policy_values))
            original = stack.enter_context(patch.object(scene.motor, "compute", side_effect=motor))
            added = stack.enter_context(patch.object(scene.extra_motor, "compute", return_value=np.zeros(6)))
            stack.enter_context(patch.object(scene.ik, "solve", return_value=planned_extra))
            stack.enter_context(patch.object(scene, "check_arm_contacts"))
            physics = stack.enter_context(patch("mujoco.mj_step", side_effect=integrate))
            forward = stack.enter_context(patch("mujoco.mj_forward"))
            scene.step()
        self.assertEqual(inference.call_count, 1)
        self.assertEqual(original.call_count, 10)
        self.assertEqual(added.call_count, 10)
        self.assertEqual(physics.call_count, 10)
        forward.assert_not_called()
        self.assertEqual(scene.policy_output_checks, 10)
        self.assertEqual(scene.policy_output_override_max, 0)
        self.assertAlmostEqual(scene.data.time, .02)
        self.assertFalse(np.array_equal(observations[0][0], observations[-1][0]))
        for actual, expected in zip(policy_values, before_values):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(scene.last_policy_command, policy_values[0])
        np.testing.assert_array_equal(scene.last_motor_command, policy_values[0])
        np.testing.assert_array_equal(reference_pos, reference_before)
        np.testing.assert_array_equal(reference_vel, np.full((4, 21), .17))

    def test_added_planner_variables_and_all_fk_iterations_exclude_original_joints(self) -> None:
        scene, planner = self.scene, self.scene.ik
        self.assertIsInstance(planner, AddedJointPlanner)
        self.assertEqual(planner.names, scene.extra_names[3:])
        self.assertEqual(set(planner.qids) & set(scene.qids), set())
        original_state = {name: getattr(scene.data, name).copy()
                          for name in ("qpos", "qvel", "ctrl", "qacc", "qfrc_applied", "xfrc_applied")}
        original_time = float(scene.data.time)
        planner.preferred_posture = scene.data.qpos[planner.qids].copy()
        planner.orientation = scene.data.site_xmat[planner.site].reshape(3, 3).copy()
        original_fk = mujoco.mj_kinematics
        calls = []

        def inspect_fk(model, data):
            self.assertIs(data, planner.scratch)
            np.testing.assert_array_equal(data.qpos[scene.qids], original_state["qpos"][scene.qids])
            calls.append(data.qpos[planner.qids].copy())
            original_fk(model, data)

        with patch("mujoco.mj_kinematics", side_effect=inspect_fk):
            with patch("mujoco.mj_step") as integrate:
                solution = planner.solve(scene.data, planner.position(scene.data).copy())
        integrate.assert_not_called()
        self.assertGreater(len(calls), 3)
        self.assertEqual(solution.shape, (3,))
        self.assertTrue(np.all(solution >= planner.lower))
        self.assertTrue(np.all(solution <= planner.upper))
        for name, before in original_state.items():
            np.testing.assert_array_equal(getattr(scene.data, name), before)
        self.assertEqual(scene.data.time, original_time)

    def test_held_object_scratch_fk_does_not_move_physical_cube_or_original_arm(self) -> None:
        scene, planner = self.scene, self.scene.ik
        planner.held_body_id = scene.model.body("pick_cube_3").id
        planner.allow_target_contact = True
        planner.orientation = scene.data.site_xmat[planner.site].reshape(3, 3).copy()
        before = scene.data.qpos.copy()
        target = planner.position(scene.data).copy() + [.005, 0, .005]
        planner.solve(scene.data, target)
        np.testing.assert_array_equal(scene.data.qpos, before)
        np.testing.assert_array_equal(planner.scratch.qpos[scene.qids], before[scene.qids])

    def test_inference_failure_never_integrates_or_reuses_old_commands(self) -> None:
        with patch.object(self.scene.policy, "step", return_value=None):
            with patch("mujoco.mj_step") as integrate:
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    self.scene.step()
        integrate.assert_not_called()

    def test_reset_clears_added_planning_and_original_command_audit(self) -> None:
        scene = self.scene
        scene.extra_integral[:] = .11
        scene.extra_reference[:] = .23
        scene.tool_target = np.ones(3)
        scene.last_policy_command[:] = .31
        scene.last_motor_command[:] = .31
        scene.policy_output_checks = 77
        scene.policy_output_override_max = .05
        scene.ik.last_solution = np.ones(3)
        scene.reset()
        np.testing.assert_array_equal(scene.extra_integral, np.zeros(6))
        np.testing.assert_array_equal(scene.extra_reference, scene.stow_angles)
        self.assertIsNone(scene.tool_target)
        self.assertIsNone(scene.ik.last_solution)
        np.testing.assert_array_equal(scene.last_policy_command, np.zeros(21))
        np.testing.assert_array_equal(scene.last_motor_command, np.zeros(21))
        self.assertEqual(scene.policy_output_checks, 0)
        self.assertEqual(scene.policy_output_override_max, 0)
        self.assertFalse(scene.data.eq_active[scene.model.equality("base_support_weld").id])

    def test_future_samples_are_clamped_at_unverified_grasp_for_every_observation_field(self) -> None:
        task = self.make_task()
        reference = task.reference
        gate = round(task.phase_times["LIFT"] / .02) - 1
        sampled = reference.get_slice(np.array([0, 0]), np.array([gate - 1, gate]),
                                      np.array([-100, 0, 1, 20, 1000]))
        expected = np.clip(np.array([gate - 1, gate])[:, None] + [-100, 0, 1, 20, 1000], 0, gate)
        for name, values in reference._storage.items():
            np.testing.assert_array_equal(getattr(sampled, name), values[expected])
        self.assertFalse(task.grasp_verified)
        self.assertEqual(reference.allowed_frame, gate)

    def test_unaligned_grasp_holds_clock_and_does_not_begin_closure(self) -> None:
        task, scene = self.make_task(), self.scene
        task.reference_time = task.phase_times["CLOSE"]
        with patch.object(task, "grasp_alignment", return_value=(False, np.array([.03, 0, 0]))):
            with patch.object(task, "set_extra_plan"), patch.object(scene, "step"), patch("builtins.print"):
                task.step()
        self.assertEqual(task.phase, "CLOSE")
        self.assertIsNone(task.close_started)
        self.assertFalse(task.grasp_verified)
        self.assertEqual(task.reference_time, task.phase_times["CLOSE"])
        self.assertAlmostEqual(scene.openings[1], .035)
        self.assertEqual(task.reference.allowed_frame, round(task.phase_times["LIFT"] / .02) - 1)

    def test_a_newly_aligned_grasp_cannot_unlock_lift_on_existing_contacts_alone(self) -> None:
        task, scene = self.make_task(), self.scene
        task.reference_time = task.phase_times["LIFT"] - .02
        scene.data.time = 8.
        with patch.object(task, "grasp_alignment", return_value=(True, np.zeros(3))):
            with patch.object(task, "contacts", return_value=np.ones(2)):
                with patch.object(task, "set_extra_plan"), patch.object(scene, "step"), patch("builtins.print"):
                    task.step()
        self.assertEqual(task.close_started, 8.)
        self.assertFalse(task.grasp_verified)
        self.assertEqual(task.reference_time, task.phase_times["LIFT"] - .02)
        self.assertEqual(task.reference.allowed_frame, round(task.phase_times["LIFT"] / .02) - 1)

    def test_opposing_contacts_unlock_lift_but_future_walk_waits_for_actual_lift_gate(self) -> None:
        task, scene = self.make_task(), self.scene
        task.reference_time = task.phase_times["LIFT"] - .02
        task.close_started = 0.
        scene.data.time = 2.
        with patch.object(task, "contacts", return_value=np.ones(2)):
            with patch.object(task, "set_extra_plan"), patch.object(scene, "step"), patch("builtins.print"):
                task.step()
        self.assertTrue(task.grasp_verified)
        carry_gate = round(task.phase_times["CARRY"] / .02) - 1
        self.assertEqual(task.reference.allowed_frame, carry_gate)
        future = task.reference.get_slice(np.array([0]), np.array([49]), np.array([200]))
        self.assertEqual(int(future.step[0, 0]), carry_gate)

    def test_single_finger_contact_cannot_unlock_lift_after_closing_delay(self) -> None:
        task, scene = self.make_task(), self.scene
        task.reference_time = task.phase_times["LIFT"] - .02
        task.close_started = 0.
        scene.data.time = 2.
        with patch.object(task, "contacts", return_value=np.array([1., 0.])):
            with patch.object(task, "set_extra_plan"), patch.object(scene, "step"), patch("builtins.print"):
                task.step()
        self.assertFalse(task.grasp_verified)
        self.assertEqual(task.reference_time, task.phase_times["LIFT"] - .02)
        self.assertEqual(task.reference.allowed_frame, round(task.phase_times["LIFT"] / .02) - 1)

    def test_future_carry_requires_measured_height_and_both_finger_contacts(self) -> None:
        task, scene = self.make_task(), self.scene
        task.reference_time = task.phase_times["CARRY"]
        task.grasp_verified = True
        task.phase = "LIFT"
        carry_gate = round(task.phase_times["CARRY"] / .02) - 1
        task.reference.allowed_frame = carry_gate
        original_height = float(task.initial_cube[2])
        cube_address = int(scene.model.joint("pick_cube_3_free").qposadr[0])
        for height, contacts, ready in ((.079, [1., 1.], False),
                                        (.081, [1., .029], False),
                                        (.081, [.031, .031], True)):
            with self.subTest(height=height, contacts=contacts):
                # Only the test fixture changes physical object position; the
                # gate itself must read its measured height and contact forces.
                scene.data.qpos[cube_address + 2] = original_height + height
                mujoco.mj_forward(scene.model, scene.data)
                with patch.object(task, "contacts", return_value=np.array(contacts)):
                    with patch.object(task, "set_extra_plan"), patch.object(scene, "step"), patch("builtins.print"):
                        task.step()
                self.assertEqual(task.lift_verified, ready)
                self.assertEqual(task.phase, "CARRY" if ready else "LIFT")
                self.assertEqual(task.reference.allowed_frame,
                                 round(task.phase_times["RELEASE"] / .02) - 1 if ready else carry_gate)
                self.assertAlmostEqual(task.reference_time, task.phase_times["CARRY"] + (.02 if ready else 0))

    def test_release_gate_holds_fingers_closed_until_cube_is_over_basket(self) -> None:
        task, scene = self.make_task(), self.scene
        task.reference_time = task.phase_times["RELEASE"]
        task.grasp_verified = True
        task.phase = "PLACE"
        task.reference.allowed_frame = round(task.phase_times["RELEASE"] / .02) - 1
        scene.openings[1] = .02
        with patch.object(task, "release_ready", return_value=False):
            with patch.object(task, "contacts", return_value=np.ones(2)):
                with patch.object(task, "closing_command", return_value=.02):
                    with patch.object(task, "set_extra_plan"), patch.object(scene, "step"), patch("builtins.print"):
                        task.step()
        self.assertEqual(task.phase, "PLACE")
        self.assertAlmostEqual(scene.openings[1], .02)
        self.assertEqual(task.reference_time, task.phase_times["RELEASE"])
        self.assertEqual(task.reference.allowed_frame, round(task.phase_times["RELEASE"] / .02) - 1)


if __name__ == "__main__":
    unittest.main()
