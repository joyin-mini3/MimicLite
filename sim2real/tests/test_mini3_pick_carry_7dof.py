from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

import mujoco
import numpy as np

from test_mini3_pick_carry import PolicyDouble, ROOT, compatible_policy_config
from mini3_extra_arm_motor import EXTRA_ARM_JOINT_NAMES
from mini3_pick_carry_7dof import ArticulatedPickCarryTask, ArticulatedPolicyScene


SCENE = ROOT / "any4hdmi/assets/robots/mini3_mjlab/scene_pick_carry_7dof.xml"


class Mini3SevenJointCarryTest(unittest.TestCase):
    def setUp(self) -> None:
        with patch("sim2real.sim_env.integrated_sim2sim.IntegratedPolicyRuntime", PolicyDouble):
            with patch("builtins.open", mock_open(read_data=json.dumps(compatible_policy_config()))):
                self.scene = ArticulatedPolicyScene(SCENE, Path("unused.yaml"), Path("unused.npz"))

    def test_added_dofs_and_actuators_stay_outside_the_trained_policy_contract(self) -> None:
        scene = self.scene
        self.assertEqual((scene.model.nq, scene.model.nv, scene.model.nu), (59, 55, 29))
        self.assertEqual(tuple(scene.extra_names), EXTRA_ARM_JOINT_NAMES)
        original = mujoco.MjModel.from_xml_path(str(SCENE.with_name("mini3.xml")))
        expected_names = [original.joint(index).name for index in range(1, original.njnt)]
        self.assertEqual(scene.names, expected_names)
        self.assertEqual(set(scene.qids) & set(scene.extra_qids), set())
        self.assertEqual(set(scene.aids) & set(scene.extra_aids), set())
        self.assertEqual(set(scene.grippers) & (set(scene.aids) | set(scene.extra_aids)), set())
        for name, joint_id, actuator_id in zip(scene.extra_names, scene.extra_jids, scene.extra_aids):
            self.assertEqual(scene.model.joint(name).id, joint_id)
            self.assertEqual(scene.model.actuator_trnid[actuator_id, 0], joint_id)
        expected = np.linspace(-0.1, 0.1, 21)
        scene.data.qpos[scene.qids] = expected
        scene.data.qpos[scene.extra_qids] = np.linspace(0.2, 0.4, 6)
        scene.data.qvel[:] = np.arange(scene.model.nv) * 0.01
        scene.sync()
        self.assertEqual(scene.state.qpos.shape, (28,))
        self.assertEqual(scene.state.qvel.shape, (27,))
        np.testing.assert_allclose(scene.state.joint_pos, expected)
        np.testing.assert_allclose(scene.state.joint_vel, scene.data.qvel[scene.vids])

    def test_seven_joint_commands_stream_only_four_policy_columns_and_use_both_motor_chains(self) -> None:
        scene = self.scene
        self.assertEqual(len(scene.ik.names), 7)
        self.assertEqual(scene.ik.names[4:], scene.extra_names[3:])
        self.assertEqual([scene.names[index] for index in scene.arm_indices], scene.ik.names[:4])
        joint_positions = np.arange(8 * 21, dtype=float).reshape(8, 21) * 0.001
        joint_velocities = np.full((8, 21), 0.5)
        scene.state.motion_dataset = SimpleNamespace(
            num_steps=8, _storage={"joint_pos": joint_positions, "joint_vel": joint_velocities},
        )
        scene.state.motion_t = np.array([1])
        before_reference = joint_positions.copy()
        before_qpos, before_qvel = scene.data.qpos.copy(), scene.data.qvel.copy()
        scene.arm_target = scene.ik.position(scene.data).copy()
        goal = scene.data.qpos[scene.ik.qids] + np.linspace(0.01, 0.07, 7)
        original_torque, extra_torque = np.linspace(-0.5, 0.5, 21), np.linspace(-0.6, 0.6, 6)
        scene.openings[:] = [0.035, 0]
        with patch.object(scene.ik, "solve", return_value=goal):
            with patch.object(scene.motor, "compute", return_value=original_torque) as original:
                with patch.object(scene.extra_motor, "compute", return_value=extra_torque) as extra:
                    with patch("mujoco.mj_step") as integrate:
                        scene.step()
        self.assertEqual(original.call_count, 10)
        self.assertEqual(extra.call_count, 10)
        self.assertEqual(integrate.call_count, 10)
        np.testing.assert_array_equal(scene.data.qpos, before_qpos)
        np.testing.assert_array_equal(scene.data.qvel, before_qvel)
        np.testing.assert_array_equal(joint_positions[:2], before_reference[:2])
        other_columns = np.setdiff1d(np.arange(21), scene.arm_indices)
        np.testing.assert_array_equal(joint_positions[:, other_columns], before_reference[:, other_columns])
        np.testing.assert_array_equal(joint_positions[2:, scene.arm_indices],
                                      np.tile(scene.arm_command[:4], (6, 1)))
        np.testing.assert_array_equal(joint_velocities[2:, scene.arm_indices], np.zeros((6, 4)))
        self.assertEqual(original.call_args.args[0].shape, (21,))
        self.assertEqual(extra.call_args.args[0].shape, (6,))
        np.testing.assert_array_equal(original.call_args.args[1], before_qpos[scene.qids])
        np.testing.assert_array_equal(extra.call_args.args[1], before_qpos[scene.extra_qids])
        np.testing.assert_array_equal(extra.call_args.args[0][:3],
                                      getattr(scene, "stow_angles", np.zeros(6))[:3])
        np.testing.assert_array_equal(extra.call_args.kwargs["effort"][3:],
                                      scene.data.qfrc_bias[scene.ik.vids[4:]])
        np.testing.assert_array_equal(original.call_args.kwargs["effort"][scene.arm_indices],
                                      scene.data.qfrc_bias[scene.ik.vids[:4]])
        np.testing.assert_array_equal(original.call_args.kwargs["kp"][scene.arm_indices],
                                      np.full(4, scene.arm_kp))
        np.testing.assert_array_equal(original.call_args.kwargs["kd"][scene.arm_indices],
                                      np.full(4, scene.arm_kd))
        # The independent motor uses its configured gains, without inheriting
        # the original arm's gains through per-step overrides.
        self.assertNotIn("kp", extra.call_args.kwargs)
        self.assertNotIn("kd", extra.call_args.kwargs)
        np.testing.assert_allclose(scene.data.ctrl[scene.aids], original_torque)
        np.testing.assert_allclose(scene.data.ctrl[scene.extra_aids], extra_torque)
        np.testing.assert_allclose(scene.data.ctrl[scene.grippers], scene.openings)

    def test_seven_joint_ik_preserves_physics_and_obeys_joint_limits(self) -> None:
        scene = self.scene
        before_qpos, before_qvel = scene.data.qpos.copy(), scene.data.qvel.copy()
        # A target generated from the actual model is reachable independently
        # of the current hand approach targets used by the task state machine.
        scene.ik.orientation = scene.data.site_xmat[scene.ik.site].reshape(3, 3).copy()
        target = scene.ik.position(scene.data).copy()
        solution = scene.ik.solve(scene.data, target)
        lower, upper = scene.model.jnt_range[scene.ik.jids].T
        self.assertTrue(np.all(solution >= lower))
        self.assertTrue(np.all(solution <= upper))
        self.assertLess(scene.ik.last_error, 1e-5)
        self.assertLess(scene.ik.last_orientation_error, 1e-5)
        np.testing.assert_array_equal(scene.data.qpos, before_qpos)
        np.testing.assert_array_equal(scene.data.qvel, before_qvel)

    def test_reset_clears_extra_motor_and_arm_state_and_disables_support(self) -> None:
        scene = self.scene
        scene.extra_target[:] = 0.1
        scene.step()
        self.assertAlmostEqual(float(scene.data.time), 0.02)
        self.assertGreater(float(np.linalg.norm(scene.extra_motor.applied_torque)), 0)
        self.assertTrue(scene.extra_motor.response_enabled)
        self.assertTrue(scene.extra_motor.tn_enabled)
        self.assertTrue(scene.extra_motor.kt_enabled)
        scene.arm_integral[:] = 0.1
        scene.arm_command = np.ones(7)
        scene.arm_target = np.ones(3)
        scene.ik.last_solution = np.ones(7)
        scene.ik.allow_posture_reseed = False
        scene.ik.held_body_id = scene.model.body("pick_cube_3").id
        scene.ik.allow_target_contact = True
        scene.ik.orientation[:] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        scene.data.eq_active[scene.model.equality("base_support_weld").id] = True
        scene.reset()
        expected = mujoco.MjData(scene.model)
        mujoco.mj_resetDataKeyframe(scene.model, expected, scene.model.key("home").id)
        expected.qpos[scene.extra_qids] = getattr(scene, "stow_angles", np.zeros(6))
        mujoco.mj_forward(scene.model, expected)
        np.testing.assert_array_equal(scene.extra_target, getattr(scene, "stow_angles", np.zeros(6)))
        np.testing.assert_array_equal(scene.arm_integral, np.zeros(7))
        np.testing.assert_array_equal(scene.extra_motor.response_torque, np.zeros(6))
        np.testing.assert_array_equal(scene.extra_motor.applied_torque, np.zeros(6))
        np.testing.assert_allclose(scene.data.qpos, expected.qpos)
        np.testing.assert_allclose(scene.data.cam_xpos, expected.cam_xpos)
        self.assertIsNone(scene.arm_command)
        self.assertIsNone(scene.arm_target)
        self.assertIsNone(scene.ik.last_solution)
        self.assertTrue(scene.ik.allow_posture_reseed)
        self.assertIsNone(scene.ik.held_body_id)
        self.assertFalse(scene.ik.allow_target_contact)
        np.testing.assert_array_equal(scene.ik.orientation, np.eye(3))
        self.assertFalse(scene.data.eq_active[scene.model.equality("base_support_weld").id])

    def test_enabled_pelvis_support_stops_before_any_physics_step(self) -> None:
        scene = self.scene
        scene.data.eq_active[scene.model.equality("base_support_weld").id] = True
        with patch("mujoco.mj_step") as integrate:
            with self.assertRaisesRegex(RuntimeError, "support must stay disabled"):
                scene.step()
        integrate.assert_not_called()

    def make_grasp_task(self, *, lateral_offset=0.0, orientation_error=0.0):
        from scipy.spatial.transform import Rotation

        scene = self.scene
        # Isolate the gate from the walking stow pose and all private motion
        # data. Only this test fixture repositions its free cube.
        scene.data.qpos[scene.ik.qids] = 0
        mujoco.mj_forward(scene.model, scene.data)
        rotation = scene.data.site_xmat[scene.ik.site].reshape(3, 3).copy()
        scene.ik.orientation = rotation @ Rotation.from_rotvec([orientation_error, 0, 0]).as_matrix()
        center = scene.ik.position(scene.data).copy()
        cube_joint = scene.model.joint("pick_cube_3_free")
        address = int(cube_joint.qposadr[0])
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        scene.data.qpos[address:address + 3] = center + rotation @ [0, lateral_offset, 0]
        scene.data.qpos[address + 3:address + 7] = quaternion
        mujoco.mj_forward(scene.model, scene.data)
        task = object.__new__(ArticulatedPickCarryTask)
        task.r = scene
        task.cube = scene.model.body("pick_cube_3").id
        task.cube_geom = scene.model.geom("pick_cube_3_geom").id
        task.phase, task.phase_started = "LOWER", 0.0
        task.events = []
        task.done, task.success, task.failure = False, False, None
        return task

    def set_cube_rotation_in_pad_frame(self, relative_rotation):
        scene = self.scene
        rotation = scene.data.site_xmat[scene.ik.site].reshape(3, 3) @ relative_rotation
        address = int(scene.model.joint("pick_cube_3_free").qposadr[0])
        mujoco.mju_mat2Quat(scene.data.qpos[address + 3:address + 7], rotation.ravel())
        mujoco.mj_forward(scene.model, scene.data)

    def test_closure_projects_all_cube_corners_and_reaches_preload_after_one_second(self) -> None:
        from itertools import product
        from scipy.spatial.transform import Rotation

        task = self.make_grasp_task()
        scene = self.scene
        signs = np.array(list(product((-1, 1), repeat=3)))
        rotations = (Rotation.identity(), Rotation.from_euler("z", np.pi / 4),
                     Rotation.from_euler("xyz", [0.3, 0.5, -0.7]))
        for rotation in rotations:
            with self.subTest(rotation=rotation.as_rotvec()):
                self.set_cube_rotation_in_pad_frame(rotation.as_matrix())
                hand_rotation = scene.data.site_xmat[scene.ik.site].reshape(3, 3)
                cube_rotation = scene.data.xmat[task.cube].reshape(3, 3)
                # Independently transform all eight corners instead of using
                # the controller's absolute-matrix half-extents shortcut.
                corners = signs * scene.model.geom_size[task.cube_geom]
                pad_corners = corners @ cube_rotation.T @ hand_rotation
                half_width = float(np.ptp(pad_corners[:, 1]) / 2)
                closed = half_width - 0.002
                before_qpos, before_qvel = scene.data.qpos.copy(), scene.data.qvel.copy()
                self.assertAlmostEqual(task.closing_command(0.0), 0.035)
                self.assertAlmostEqual(task.closing_command(0.5), (0.035 + closed) / 2)
                self.assertAlmostEqual(task.closing_command(1.0), closed)
                self.assertAlmostEqual(task.closing_command(1.2), closed)
                samples = [task.closing_command(t) for t in np.linspace(0, 1, 11)]
                self.assertTrue(np.all(np.diff(samples) <= 0))
                np.testing.assert_array_equal(scene.data.qpos, before_qpos)
                np.testing.assert_array_equal(scene.data.qvel, before_qvel)

    def test_held_phases_refresh_opening_as_cube_turns_but_release_does_not_reclose(self) -> None:
        from scipy.spatial.transform import Rotation

        task = self.make_grasp_task()
        task.records = []
        scene = self.scene
        # Isolate the articulated wrapper from walking and contact integration:
        # each successive call still observes a new physical cube orientation.
        with patch("mini3_pick_carry.PickCarryTask.step"):
            for phase in ("LIFT", "CARRY", "PLACE"):
                with self.subTest(phase=phase):
                    task.phase = phase
                    scene.openings[:] = [0.031, 0.007]
                    self.set_cube_rotation_in_pad_frame(Rotation.identity().as_matrix())
                    task.step()
                    aligned = float(scene.openings[1])
                    self.set_cube_rotation_in_pad_frame(Rotation.from_euler("z", np.pi / 4).as_matrix())
                    task.step()
                    self.assertGreater(scene.openings[1], aligned + 0.007)
                    self.assertAlmostEqual(scene.openings[1], task.closing_command(1.0))
                    self.assertAlmostEqual(scene.openings[0], 0.031)
            task.phase = "RELEASE"
            scene.openings[1] = 0.035
            with patch.object(task, "closing_command", side_effect=AssertionError("Release must stay open")):
                task.step()
            self.assertAlmostEqual(scene.openings[1], 0.035)

    def test_target_contact_is_enabled_only_after_the_early_approach(self) -> None:
        task = self.make_grasp_task()
        scene = self.scene
        task.start_tcp = scene.ik.position(scene.data).copy()
        task.clear_target = task.start_tcp + [0, 0, 0.1]
        task.grasp_rotation = scene.ik.orientation.copy()
        before_qpos, before_qvel = scene.data.qpos.copy(), scene.data.qvel.copy()
        for phase in ("APPROACH", "CLEAR_ARM", "REACH", "LOWER", "CLOSE", "LIFT", "CARRY", "PLACE", "RELEASE"):
            with self.subTest(phase=phase):
                task.enter(phase)
                self.assertEqual(scene.collision_phase, phase)
                self.assertEqual(scene.ik.allow_target_contact,
                                 phase not in ("APPROACH", "CLEAR_ARM", "REACH"))
                self.assertEqual(scene.ik.held_body_id,
                                 task.cube if phase in ("LIFT", "CARRY", "PLACE") else None)
        np.testing.assert_array_equal(scene.data.qpos, before_qpos)
        np.testing.assert_array_equal(scene.data.qvel, before_qvel)

    def test_ik_contact_permission_skips_only_target_finger_clearances(self) -> None:
        task = self.make_grasp_task()
        scene, ik = self.scene, self.scene.ik
        finger = scene.model.geom("right_gripper_finger_positive_geom").id
        address = int(scene.model.joint("pick_cube_3_free").qposadr[0])
        scene.data.qpos[address:address + 3] = scene.data.geom_xpos[finger]
        mujoco.mj_forward(scene.model, scene.data)
        before_qpos = scene.data.qpos.copy()
        mujoco.mj_copyData(ik.scratch, scene.model, scene.data)
        ik.allow_target_contact = False
        blocked = ik.clearances()
        target_pairs = ik.collision_pairs[ik.target_pair_start:]
        self.assertEqual(set(map(tuple, target_pairs)), {
            tuple(sorted((task.cube_geom, scene.model.geom(f"right_gripper_finger_{part}_geom").id)))
            for part in ("positive", "negative")
        })
        self.assertLess(float(blocked[ik.target_pair_start:].min()), -0.002)
        ik.allow_target_contact = True
        permitted = ik.clearances()
        np.testing.assert_array_equal(permitted[:ik.target_pair_start], blocked[:ik.target_pair_start])
        self.assertTrue(np.all(permitted[ik.target_pair_start:] > 0.004))
        np.testing.assert_array_equal(scene.data.qpos, before_qpos)

    def test_close_rejects_unsafe_alignment_even_within_original_position_tolerance(self) -> None:
        for lateral, angle in ((0.017, 0.0), (0.0, 0.13)):
            with self.subTest(lateral_offset=lateral, orientation_error=angle):
                task = self.make_grasp_task(lateral_offset=lateral, orientation_error=angle)
                center = self.scene.ik.position(self.scene.data)
                self.assertLess(np.linalg.norm(center - self.scene.data.xpos[task.cube]), 0.018)
                opening = self.scene.openings.copy()
                before_qpos = self.scene.data.qpos.copy()
                task.enter("CLOSE")
                self.assertEqual(task.phase, "FAILED")
                self.assertTrue(task.done)
                self.assertIn("safely centred", task.failure)
                self.assertFalse(self.scene.freeze_arm_command)
                np.testing.assert_array_equal(self.scene.openings, opening)
                np.testing.assert_array_equal(self.scene.data.qpos, before_qpos)
                self.assertAlmostEqual(task.grasp_alignment["orientation_error_rad"], angle)
                self.assertAlmostEqual(task.grasp_alignment["cube_in_pad_frame_m"][1], lateral)

    def test_close_accepts_cube_centred_between_correctly_oriented_pads(self) -> None:
        task = self.make_grasp_task()
        before_qpos = self.scene.data.qpos.copy()
        task.enter("CLOSE")
        self.assertEqual(task.phase, "CLOSE")
        self.assertFalse(task.done)
        self.assertIsNone(task.failure)
        self.assertTrue(self.scene.freeze_arm_command)
        self.assertEqual(task.events[-1]["phase"], "CLOSE")
        np.testing.assert_allclose(task.grasp_alignment["cube_in_pad_frame_m"], np.zeros(3), atol=1e-12)
        self.assertLess(task.grasp_alignment["orientation_error_rad"], 1e-12)
        np.testing.assert_array_equal(self.scene.data.qpos, before_qpos)

    def test_release_target_accounts_for_rotated_basket_cube_and_grasp_offset(self) -> None:
        from itertools import product
        from scipy.spatial.transform import Rotation

        task = self.make_grasp_task()
        scene = self.scene
        task.basket = scene.model.body("pick_basket").id
        rotation = Rotation.from_euler("z", 0.7).as_matrix()
        cube_rotation = rotation @ Rotation.from_euler("z", np.pi / 4).as_matrix()
        pad = scene.ik.position(scene.data).copy()
        cube = pad + [0.005, -0.007, 0.004]
        initial_local_cube = np.array([-0.23, 0.02, 0.18])
        basket = cube - rotation @ initial_local_cube
        scene.model.body_pos[task.basket] = basket
        mujoco.mju_mat2Quat(scene.model.body_quat[task.basket], rotation.ravel())
        address = int(scene.model.joint("pick_cube_3_free").qposadr[0])
        scene.data.qpos[address:address + 3] = cube
        mujoco.mju_mat2Quat(scene.data.qpos[address + 3:address + 7], cube_rotation.ravel())
        mujoco.mj_forward(scene.model, scene.data)
        before_qpos, before_qvel = scene.data.qpos.copy(), scene.data.qvel.copy()
        before_bodies = scene.data.xpos.copy()
        target = task.release_target_position()
        released_cube = target + (cube - pad)
        np.testing.assert_allclose(task.release_cube_target, released_cube)
        local_cube = rotation.T @ (released_cube - basket)
        # Check all eight transformed cube corners, independently of the
        # controller's projected-half-extents calculation.
        corners = np.array(list(product((-1, 1), repeat=3))) * scene.model.geom_size[task.cube_geom]
        local_corners = (released_cube + corners @ cube_rotation.T - basket) @ rotation
        self.assertGreaterEqual(float(local_corners[:, 0].min()), -0.12 + 0.03 - 1e-12)
        self.assertLessEqual(float(local_corners[:, 0].max()), 0.12 - 0.03 + 1e-12)
        self.assertGreaterEqual(float(local_corners[:, 1].min()), -0.11 + 0.03 - 1e-12)
        self.assertLessEqual(float(local_corners[:, 1].max()), 0.11 - 0.03 + 1e-12)
        self.assertGreaterEqual(float(local_corners[:, 2].min()), 0.12 + 0.02 - 1e-12)
        # Projection changes only the out-of-bounds coordinate, stopping at
        # the nearest safe edge rather than unnecessarily seeking the centre.
        self.assertAlmostEqual(float(local_corners[:, 0].min()), -0.09)
        self.assertAlmostEqual(float(local_cube[1]), float(initial_local_cube[1]))
        self.assertAlmostEqual(float(local_cube[2]), 0.18)
        self.assertGreater(np.linalg.norm(target - task.release_cube_target), 0.008)
        np.testing.assert_array_equal(scene.data.qpos, before_qpos)
        np.testing.assert_array_equal(scene.data.qvel, before_qvel)
        np.testing.assert_array_equal(scene.data.xpos, before_bodies)

    def test_release_target_rejects_basket_too_small_for_cube_and_margin(self) -> None:
        task = self.make_grasp_task()
        scene = self.scene
        task.basket = scene.model.body("pick_basket").id
        scene.model.geom("pick_basket_front").pos[0] = 0.04
        scene.model.geom("pick_basket_back").pos[0] = -0.04
        before_qpos = scene.data.qpos.copy()
        with self.assertRaisesRegex(ValueError, "too small"):
            task.release_target_position()
        self.assertFalse(hasattr(task, "release_cube_target"))
        np.testing.assert_array_equal(scene.data.qpos, before_qpos)

    def test_held_cube_follows_each_ik_candidate_only_in_scratch_state(self) -> None:
        from scipy.spatial.transform import Rotation

        task = self.make_grasp_task(lateral_offset=0.006)
        scene, ik = self.scene, self.scene.ik
        hand_rotation = scene.data.site_xmat[ik.site].reshape(3, 3).copy()
        cube_address = int(scene.model.joint("pick_cube_3_free").qposadr[0])
        relative_rotation = Rotation.from_euler("z", 0.2).as_matrix()
        mujoco.mju_mat2Quat(scene.data.qpos[cube_address + 3:cube_address + 7],
                           (hand_rotation @ relative_rotation).ravel())
        mujoco.mj_forward(scene.model, scene.data)
        local_offset = hand_rotation.T @ (scene.data.xpos[task.cube] - ik.position(scene.data))
        before_qpos, before_qvel = scene.data.qpos.copy(), scene.data.qvel.copy()
        before_bodies = scene.data.xpos.copy()
        ik.held_body_id = task.cube
        original_clearances = ik.clearances

        def checked_clearances():
            # The transform must already be current when collision distances
            # are evaluated, including intermediate optimization candidates.
            rotation = ik.scratch.site_xmat[ik.site].reshape(3, 3)
            cube_rotation = ik.scratch.xmat[task.cube].reshape(3, 3)
            np.testing.assert_allclose(rotation.T @ (ik.scratch.xpos[task.cube] - ik.position(ik.scratch)),
                                       local_offset, atol=1e-10)
            np.testing.assert_allclose(rotation.T @ cube_rotation, relative_rotation, atol=1e-10)
            return original_clearances()

        with patch.object(ik, "clearances", side_effect=checked_clearances) as collision_check:
            ik.solve(scene.data, ik.position(scene.data) + [0.04, -0.03, 0.02])
        self.assertGreater(collision_check.call_count, 1)
        self.assertGreater(np.linalg.norm(ik.scratch.xpos[task.cube] - scene.data.xpos[task.cube]), 0.001)
        np.testing.assert_array_equal(scene.data.qpos, before_qpos)
        np.testing.assert_array_equal(scene.data.qvel, before_qvel)
        np.testing.assert_array_equal(scene.data.xpos, before_bodies)

    def test_held_body_must_have_a_free_joint(self) -> None:
        scene = self.scene
        scene.ik.held_body_id = scene.model.body("pick_basket").id
        before_qpos = scene.data.qpos.copy()
        with self.assertRaisesRegex(ValueError, "must have a free joint"):
            scene.ik.solve(scene.data, scene.ik.position(scene.data))
        np.testing.assert_array_equal(scene.data.qpos, before_qpos)


if __name__ == "__main__":
    unittest.main()
