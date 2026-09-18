from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from test_mini3_pick_carry import PolicyDouble, ROOT, compatible_policy_config
from mini3_pick_carry_policy import AddedJointPlanner, GatedReference, PolicyOnlyScene, PolicyPickCarryTask


SCENE = ROOT / "any4hdmi/assets/robots/mini3_mjlab/scene_pick_carry_7dof.xml"


class PolicyTaskControlTest(unittest.TestCase):
    def setUp(self) -> None:
        with patch("sim2real.sim_env.integrated_sim2sim.IntegratedPolicyRuntime", PolicyDouble):
            with patch("builtins.open", mock_open(read_data=json.dumps(compatible_policy_config()))):
                self.scene = PolicyOnlyScene(SCENE, Path("unused.yaml"), Path("unused.npz"))

    def make_task(self, target_body="pick_cube_3", *, phases=None):
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
        if phases is None:
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
            return PolicyPickCarryTask(scene, plan, target_body=target_body)

    def test_target_switch_updates_task_planner_and_collision_monitor(self):
        scene = self.scene
        for index in (1, 2, 3):
            with self.subTest(target=index):
                task = self.make_task(f"pick_cube_{index}")
                self.assertEqual(task.cube_geom, scene.model.geom(f"pick_cube_{index}_geom").id)
                for other in (1, 2, 3):
                    cube = scene.model.geom(f"pick_cube_{other}_geom").id
                    for part in ("positive", "negative"):
                        finger = scene.model.geom(f"right_gripper_finger_{part}_geom").id
                        pair = tuple(sorted((cube, finger)))
                        self.assertEqual(pair in scene.unexpected_pairs, other != index)
                        self.assertEqual(pair in scene.pregrasp_pairs, other == index)
                        planned = set(map(tuple, scene.ik.collision_pairs[:scene.ik.target_pair_start]))
                        self.assertEqual(pair in planned, other != index)
                    palm = scene.model.geom("right_gripper_palm_geom").id
                    self.assertIn(tuple(sorted((cube, palm))), scene.unexpected_pairs)

    def test_basket_success_uses_selected_cube_contact_and_velocity(self):
        scene = self.scene
        task = self.make_task("pick_cube_1")
        address = int(scene.model.joint("pick_cube_1_free").qposadr[0])
        scene.data.qpos[address:address + 3] = scene.data.xpos[task.basket] + [0., 0., .0299]
        mujoco.mj_forward(scene.model, scene.data)
        self.assertTrue(task.cube_in_basket())
        velocity = int(scene.model.joint("pick_cube_1_free").dofadr[0])
        scene.data.qvel[velocity] = .1
        self.assertFalse(task.cube_in_basket())

    def test_every_original_command_channel_passes_through_all_ten_motor_substeps(self) -> None:
        scene = self.scene
        task = self.make_task()
        task.phase, task.reference_time = "LOWER", 1.
        task.set_extra_plan(task.plan.sample(task.reference_time))
        self.assertIsNone(task.episode)
        self.assertGreater(np.linalg.norm(task.reference.root_velocity_correction), 0.)
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

    def test_randomized_lowering_captures_alignment_without_unlocking_lift(self) -> None:
        for ready in (False, True):
            with self.subTest(ready=ready):
                self.scene.reset()
                task, scene = self.make_task(), self.scene
                task.episode = {}
                task.reference_time = .4  # LOWER, before scheduled CLOSE at .5.
                scene.data.time = 8.
                with patch.object(task, "grasp_alignment", return_value=(ready, np.zeros(3))):
                    with patch.object(task, "set_extra_plan"), patch.object(scene, "step"), patch("builtins.print"):
                        task.step()
                self.assertEqual(task.close_started, 8. if ready else None)
                self.assertEqual(task.phase, "CLOSE" if ready else "LOWER")
                self.assertAlmostEqual(task.reference_time, .42)
                self.assertFalse(task.grasp_verified)
                self.assertEqual(task.reference.allowed_frame, round(task.phase_times["LIFT"] / .02) - 1)

    def test_fixed_layout_keeps_original_closure_schedule(self) -> None:
        task, scene = self.make_task(), self.scene
        task.reference_time = .4
        with patch.object(task, "grasp_alignment", return_value=(True, np.zeros(3))) as align:
            with patch.object(task, "set_extra_plan"), patch.object(scene, "step"), patch("builtins.print"):
                task.step()
        align.assert_not_called()
        self.assertEqual(task.phase, "LOWER")
        self.assertIsNone(task.close_started)

    def make_tool_transition(self, phase: str):
        self.scene.reset()
        task, scene = self.make_task(), self.scene
        source = scene.data.qpos.copy()
        task.plan.sample_source_qpos = lambda time: source.copy()
        scene.data.qpos[:3] += [.11, -.04, .025]
        scene.data.qpos[3:7] = np.roll(Rotation.from_euler("z", .25).as_quat(), 1)
        mujoco.mj_forward(scene.model, scene.data)
        sample = task.plan.sample(task.phase_times[phase])
        # Obtain the steady command in the new frame before applying a switch.
        task.phase = phase
        task.set_extra_plan(sample)
        destination = scene.tool_target.copy()
        destination_rotation = scene.ik.orientation.copy()
        previous = destination + [.07, -.03, .045]
        previous_rotation = Rotation.from_euler("xyz", [.2, -.3, .4]).as_matrix() @ destination_rotation
        scene.tool_target = previous.copy()
        scene.ik.orientation = previous_rotation.copy()
        scene.data.time = 10.
        with patch("builtins.print"):
            task.enter(phase)
        return task, sample, previous, previous_rotation, destination, destination_rotation

    def test_tool_frame_switch_is_continuous_and_finishes_at_new_frame(self) -> None:
        for phase in ("REACH", "CARRY", "PLACE"):
            with self.subTest(phase=phase):
                task, sample, previous, previous_rotation, destination, rotation = self.make_tool_transition(phase)
                scene = self.scene
                state_before = {name: getattr(scene.data, name).copy()
                                for name in ("qpos", "qvel", "ctrl")}
                task.set_extra_plan(sample)
                np.testing.assert_allclose(scene.tool_target, previous, atol=1e-12)
                np.testing.assert_allclose(scene.ik.orientation, previous_rotation, atol=1e-12)
                # Repeated planning at the same timestamp must not re-anchor.
                task.set_extra_plan(sample)
                np.testing.assert_allclose(scene.tool_target, previous, atol=1e-12)
                scene.data.time = 10.2
                task.set_extra_plan(sample)
                np.testing.assert_allclose(scene.tool_target, .5 * (previous + destination), atol=1e-12)
                relative = Rotation.from_matrix(previous_rotation @ rotation.T).as_rotvec()
                halfway = Rotation.from_rotvec(.5 * relative).as_matrix() @ rotation
                np.testing.assert_allclose(scene.ik.orientation, halfway, atol=1e-12)
                np.testing.assert_allclose(scene.ik.orientation.T @ scene.ik.orientation, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(scene.ik.orientation), 1.)
                for elapsed in (.4, .8):
                    scene.data.time = 10. + elapsed
                    task.set_extra_plan(sample)
                    np.testing.assert_allclose(scene.tool_target, destination, atol=1e-12)
                    np.testing.assert_allclose(scene.ik.orientation, rotation, atol=1e-12)
                for name, before in state_before.items():
                    np.testing.assert_array_equal(getattr(scene.data, name), before)
                np.testing.assert_array_equal(scene.extra_reference, sample["extra_command"])

    def test_tool_transition_eases_at_endpoints_and_advances_while_reference_is_held(self) -> None:
        task, sample, previous, previous_rotation, destination, rotation = self.make_tool_transition("REACH")
        scene = self.scene
        held_reference_time = task.reference_time
        task.set_extra_plan(sample)
        distance = np.linalg.norm(destination - previous)
        angle = Rotation.from_matrix(previous_rotation @ rotation.T).magnitude()
        for elapsed, position_endpoint, rotation_endpoint in (
                (.02, previous, previous_rotation), (.38, destination, rotation)):
            scene.data.time = 10. + elapsed
            task.set_extra_plan(sample)
            self.assertLess(np.linalg.norm(scene.tool_target - position_endpoint), .002 * distance)
            self.assertLess(Rotation.from_matrix(scene.ik.orientation @ rotation_endpoint.T).magnitude(), .002 * angle)
        scene.data.time = 10.4
        task.set_extra_plan(sample)
        self.assertEqual(task.reference_time, held_reference_time)
        np.testing.assert_allclose(scene.tool_target, destination, atol=1e-12)
        np.testing.assert_allclose(scene.ik.orientation, rotation, atol=1e-12)

    def test_approach_without_previous_tool_target_does_not_reuse_transition_offsets(self) -> None:
        task, sample, *_ = self.make_tool_transition("REACH")
        scene = self.scene
        task.set_extra_plan(sample)
        self.assertGreater(np.linalg.norm(task.tool_position_offset), 0.)
        with patch("builtins.print"):
            task.enter("APPROACH")
        task.set_extra_plan(task.plan.sample(0.))
        self.assertIsNone(scene.tool_target)
        with patch("builtins.print"):
            task.enter("REACH")
        task.set_extra_plan(sample)
        first_position, first_rotation = scene.tool_target.copy(), scene.ik.orientation.copy()
        scene.data.time += .4
        task.set_extra_plan(sample)
        np.testing.assert_allclose(scene.tool_target, first_position, atol=1e-12)
        np.testing.assert_allclose(scene.ik.orientation, first_rotation, atol=1e-12)

    def make_descent_task(self, episode=None):
        task = self.make_task(phases={
            "APPROACH": 0., "CLEAR_ARM": .1, "REACH": .2, "LOWER": .3,
            "CLOSE": 1.4, "LIFT": 2., "CARRY": 2.2, "PLACE": 2.4,
            "RELEASE": 2.6, "SUCCESS": 3.,
        })
        task.episode = episode
        task.reference_time = .92
        return task

    def test_descent_alignment_holds_open_gripper_and_future_reference_then_resumes(self) -> None:
        for episode in (None, {}):
            with self.subTest(randomized=episode is not None):
                self.scene.reset()
                self.assert_descent_alignment_holds_then_resumes(episode)

    def assert_descent_alignment_holds_then_resumes(self, episode) -> None:
        task, scene = self.make_descent_task(episode), self.scene
        held_time = task.reference_time
        with patch.object(task, "horizontal_alignment", side_effect=(False, True)):
            with patch.object(task, "grasp_alignment", return_value=(False, np.zeros(3))):
                with patch.object(task, "set_extra_plan"), patch.object(scene, "step") as integrate:
                    with patch("builtins.print"):
                        task.step()
                        self.assertEqual(task.phase, "LOWER")
                        self.assertFalse(task.descent_aligned)
                        self.assertEqual(task.reference_time, held_time)
                        self.assertIsNone(task.close_started)
                        self.assertAlmostEqual(scene.openings[1], .035)
                        self.assertAlmostEqual(task.gate_waits["ALIGN"], .02)
                        gate = round(held_time / .02)
                        self.assertEqual(task.reference.allowed_frame, gate)
                        future = task.reference.get_slice(np.array([0]), np.array([gate]),
                                                          np.array([0, 1, 1000]))
                        for name, values in task.reference._storage.items():
                            np.testing.assert_array_equal(getattr(future, name), values[[[gate] * 3]])
                        task.step()
        self.assertEqual(integrate.call_count, 2, "Physics must continue while the reference waits")
        self.assertTrue(task.descent_aligned)
        self.assertAlmostEqual(task.reference_time, held_time + .02)
        self.assertEqual(task.reference.allowed_frame, round(task.phase_times["LIFT"] / .02) - 1)
        self.assertEqual(task.gate_wait, 0.)
        self.assertAlmostEqual(task.gate_waits["ALIGN"], .02)
        self.assertFalse(task.grasp_verified)
        self.assertIsNone(task.close_started)

    def test_unaligned_descent_timeout_reports_align_gate_without_closing(self) -> None:
        task, scene = self.make_descent_task(), self.scene
        task.gate_timeout = .03
        with patch.object(task, "horizontal_alignment", return_value=False):
            with patch.object(task, "grasp_alignment", return_value=(False, np.array([.03, 0, 0]))):
                with patch.object(task, "set_extra_plan"), patch.object(scene, "step") as integrate:
                    with patch("builtins.print"):
                        task.step()
                        task.step()
        self.assertTrue(task.done)
        self.assertFalse(task.success)
        self.assertIn("Measured ALIGN gate not reached", task.failure)
        self.assertAlmostEqual(task.gate_waits["ALIGN"], .04)
        self.assertEqual(task.gate_waits["CLOSE"], 0.)
        self.assertIsNone(task.close_started)
        self.assertAlmostEqual(scene.openings[1], .035)
        self.assertEqual(integrate.call_count, 1)

    def test_alignment_velocity_is_bounded_smoothed_and_changes_only_reference_body_motion(self) -> None:
        for episode in (None, {}):
            with self.subTest(randomized=episode is not None):
                self.scene.reset()
                self.assert_alignment_velocity_changes_only_reference_body_motion(episode)

    def assert_alignment_velocity_changes_only_reference_body_motion(self, episode) -> None:
        task, scene = self.make_task(), self.scene
        task.episode, task.phase, task.reference_time = episode, "LOWER", 1.
        cube_qpos = int(scene.model.jnt_qposadr[scene.model.body_jntadr[task.cube]])
        scene.data.qpos[cube_qpos:cube_qpos + 3] = scene.ik.position(scene.data) + [.03, -.03, 0.]
        mujoco.mj_forward(scene.model, scene.data)
        pose, velocity = scene.data.qpos.copy(), scene.data.qvel.copy()
        stored = {name: values.copy() for name, values in task.reference._storage.items()}
        sample = task.plan.sample(task.reference_time)
        task.set_extra_plan(sample)
        np.testing.assert_allclose(scene.tool_target[:2], scene.data.xpos[task.cube, :2])
        np.testing.assert_allclose(task.reference.root_velocity_correction, [.012, -.012, 0.])
        task.set_extra_plan(sample)
        np.testing.assert_allclose(task.reference.root_velocity_correction, [.0228, -.0228, 0.])
        for _ in range(100):
            task.set_extra_plan(sample)
        self.assertTrue(np.all(np.abs(task.reference.root_velocity_correction) <= .12))
        correction = task.reference.root_velocity_correction.copy()
        steps = np.array([0, 1, 10])
        actual = task.reference.get_slice(np.array([0]), np.array([0]), steps)
        for name, values in stored.items():
            np.testing.assert_array_equal(task.reference._storage[name], values)
            expected = values[steps][None].copy()
            if name == "body_pos_w":
                expected += (steps * .02)[None, :, None, None] * correction
            elif name == "body_lin_vel_w":
                expected += correction
            np.testing.assert_array_equal(getattr(actual, name), expected)
        np.testing.assert_array_equal(scene.data.qpos, pose)
        np.testing.assert_array_equal(scene.data.qvel, velocity)
        task.close_started = 1.
        task.set_extra_plan(sample)
        np.testing.assert_allclose(task.reference.root_velocity_correction, .9 * correction)

    def test_placement_centers_selected_cube_in_rotated_basket_without_moving_state(self) -> None:
        for episode in (None, {}):
            with self.subTest(randomized=episode is not None):
                self.scene.reset()
                self.assert_placement_centers_selected_cube_without_moving_state(episode)

    def assert_placement_centers_selected_cube_without_moving_state(self, episode) -> None:
        task, scene = self.make_task("pick_cube_1"), self.scene
        task.episode, task.phase = episode, "PLACE"
        scene.model.body_quat[task.basket] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
        mujoco.mj_forward(scene.model, scene.data)
        rotation = scene.data.xmat[task.basket].reshape(3, 3).copy()
        cube_qpos = int(scene.model.jnt_qposadr[scene.model.body_jntadr[task.cube]])
        scene.data.qpos[cube_qpos:cube_qpos + 3] = (
            scene.data.xpos[task.basket] + rotation @ [.09, -.08, .2])
        mujoco.mj_forward(scene.model, scene.data)
        pose, velocity = scene.data.qpos.copy(), scene.data.qvel.copy()
        hand = scene.ik.position(scene.data).copy()
        task.set_extra_plan(task.plan.sample(2.2))
        expected_delta = rotation @ [-.04, .03, 0.]
        np.testing.assert_allclose(scene.tool_target[:2], hand[:2] + expected_delta[:2])
        np.testing.assert_allclose(task.reference.root_velocity_correction, [-.012, -.012, 0.])
        np.testing.assert_array_equal(scene.data.qpos, pose)
        np.testing.assert_array_equal(scene.data.qvel, velocity)
        scene.data.qpos[cube_qpos:cube_qpos + 3] = (
            scene.data.xpos[task.basket] + rotation @ [.02, -.03, .2])
        mujoco.mj_forward(scene.model, scene.data)
        task.set_extra_plan(task.plan.sample(2.2))
        np.testing.assert_allclose(scene.tool_target[:2], hand[:2])
        np.testing.assert_allclose(task.reference.root_velocity_correction, [-.0108, -.0108, 0.])

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
