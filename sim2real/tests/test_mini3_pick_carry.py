from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "any4hdmi/assets/robots/mini3_mjlab/scene_pick_carry.xml"
spec = importlib.util.spec_from_file_location("mini3_pick_carry_tested", ROOT / "mini3_pick_carry.py")
assert spec is not None and spec.loader is not None
carry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(carry)
carry.setup_paths()


def compatible_policy_config():
    """Public observation contract of the locally exported Mini3 tracker."""
    return {
        "observation": {
            "command": {
                "ref_root_pos_future_local": {
                    "_target_": "mimic_lite.ref_body_pos_future_local", "body_names": "base_link",
                },
                "ref_root_ori_future_b": {"_target_": "mimic_lite.ref_root_ori_future_b"},
                "ref_joint_pos_future": {"_target_": "mimic_lite.ref_joint_pos_future"},
                "ref_root_lin_vel_future_local": {"_target_": "mimic_lite.ref_root_lin_vel_future_local"},
            },
            "policy": {
                "root_ang_vel_history": {"_target_": "mimic_lite.root_ang_vel_history"},
                "projected_gravity_history": {"_target_": "mimic_lite.projected_gravity_history"},
                "joint_pos_history": {"_target_": "mimic_lite.joint_pos_history"},
                "joint_vel_history": {"_target_": "mimic_lite.joint_vel_history"},
                "prev_actions": {"_target_": "mimic_lite.prev_actions"},
            },
        },
        "motion": {"root_body_name": "base_link", "anchor_body_name": "base_link"},
    }


class PolicyDouble:
    """Keep the controller tests independent of private model and motion files."""

    def __init__(self, *, args, robot_cfg):
        count = len(robot_cfg.joint_names)
        qpos = np.zeros(7 + count, dtype=np.float32)
        qvel = np.zeros(6 + count, dtype=np.float32)
        self.state_processor = SimpleNamespace(
            qpos=qpos, qvel=qvel, joint_pos=qpos[7:], joint_vel=qvel[6:],
            joint_torque=np.zeros(count, dtype=np.float32), low_state_tick=-1,
            restart_motion=lambda: None,
        )
        self.joint_kp_unitree = np.full(count, 40.0)
        self.joint_kd_unitree = np.full(count, 2.0)
        self.policy_config = compatible_policy_config()
        self.state_dict = {}
        self.target = np.zeros(count)

    def reset(self):
        self.state_dict["paused"] = True

    def step(self):
        return (self.target.copy(), np.zeros_like(self.target), np.zeros_like(self.target),
                self.joint_kp_unitree.copy(), self.joint_kd_unitree.copy())


class Mini3PickCarryTest(unittest.TestCase):
    def setUp(self) -> None:
        with patch("sim2real.sim_env.integrated_sim2sim.IntegratedPolicyRuntime", PolicyDouble):
            with patch("builtins.open", mock_open(read_data=json.dumps(compatible_policy_config()))):
                self.scene = carry.PolicyScene(SCENE, Path("unused.yaml"), Path("unused.npz"))

    def make_task(self, **waits):
        """Exercise real reference construction with a public synthetic clip."""
        state = self.scene.state
        state.joint_names = self.scene.names
        state.motion_ids = np.zeros(1, dtype=np.int64)
        state.motion_t = np.zeros(1, dtype=np.int64)
        state._update_motion_data = lambda: None
        pose = np.r_[self.scene.data.qpos[:7], self.scene.data.qpos[self.scene.qids]]
        frames = np.tile(pose, (200, 1))
        frames[:, 0] += np.linspace(0, 3.2, len(frames))
        frames[:, 7:] += 0.04 * np.sin(np.linspace(0, 8 * np.pi, len(frames)))[:, None]
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            (dataset / "manifest.json").write_text(json.dumps({"timestep": 0.02}))
            (dataset / "motions").mkdir()
            motion = dataset / "motions" / "synthetic.npz"
            np.savez(motion, qpos=frames)
            return carry.PickCarryTask(self.scene, motion, **waits)

    def test_wait_changes_preserve_walk_samples_and_carry_start_blend(self) -> None:
        from mini3_motion_reference import shorten_walk

        references = []
        # Include zero and a fractional control period to catch empty padding
        # and early switching caused by truncating the requested wait.
        for wait, frames in ((0.0, 0), (0.021, 2), (1.0, 50), (3.0, 150)):
            task = self.make_task(approach_wait=wait, basket_wait=wait)
            approach = task.reference.qpos.copy()
            self.assertEqual(task.reference.dt, 0.02)
            task.start_carry()
            walking, _ = shorten_walk(task.walk, task.carry_distance)
            self.assertEqual(len(task.reference.qpos) - len(walking) - frames, 50)
            references.append((frames, approach, task.reference.qpos.copy()))

        for frames, approach, carrying in references[1:]:
            with self.subTest(hold_frames=frames):
                for actual, original in ((approach, references[0][1]),
                                         (carrying, references[0][2])):
                    self.assertEqual(len(actual), len(original) + frames)
                    np.testing.assert_array_equal(actual[:len(original)], original)
                    np.testing.assert_array_equal(actual[len(original):],
                                                  np.tile(original[-1], (frames, 1)))

    def test_lift_wait_changes_transition_time_without_speeding_up_arm(self) -> None:
        for wait in (0.3, 1.0):
            with self.subTest(lift_wait=wait):
                self.scene.reset()
                task = self.make_task(lift_wait=wait)
                task.phase, task.phase_started = "LIFT", 0.0
                task.lift_start = self.scene.ik.position(self.scene.data).copy()
                task.lift_goal = task.lift_start + [0, 0, 0.17]
                # Supply a successful grasp independently of MuJoCo contact
                # dynamics; this test isolates the scheduling contract.
                task.initial_cube[2] -= 0.17
                with patch.object(task, "contacts", return_value=np.ones(2)):
                    with patch.object(self.scene.ik, "position", return_value=self.scene.data.xpos[task.cube]):
                        with patch.object(self.scene, "step"), patch.object(task, "start_carry") as start:
                            self.scene.data.time = 1.5
                            task.step()
                            np.testing.assert_allclose(self.scene.arm_target,
                                                       (task.lift_start + task.lift_goal) / 2)
                            self.scene.data.time = 3.0 + wait - 0.02
                            task.step()
                            np.testing.assert_allclose(self.scene.arm_target, task.lift_goal)
                            start.assert_not_called()
                            self.scene.data.time = 3.0 + wait + 0.001
                            task.step()
                            start.assert_called_once()

    def test_carry_transition_uses_overridable_release_waypoint(self) -> None:
        task = self.make_task()
        scene = self.scene
        basket = scene.data.xpos[task.basket].copy()
        np.testing.assert_array_equal(task.release_target_position(), basket + [0, 0, 0.18])
        target = basket + [-0.04, 0.02, 0.18]
        task.phase = "CARRY"
        task.carry_offset = np.zeros(3)
        task.carry_height = float(scene.ik.position(scene.data)[2])
        scene.state.motion_t[:] = scene.state.motion_length - 1
        scene.ik.last_error = 0.0
        with patch.object(task, "release_target_position", return_value=target) as waypoint:
            with patch.object(task, "hold_reference"), patch.object(task, "contacts", return_value=np.ones(2)):
                with patch.object(scene.ik, "position", return_value=scene.data.xpos[task.cube]):
                    with patch.object(scene.ik, "solve") as solve, patch.object(scene, "step"):
                        task.step()
        waypoint.assert_called_once_with()
        np.testing.assert_array_equal(solve.call_args.args[1], target)
        np.testing.assert_array_equal(task.release_target, target)
        self.assertEqual(task.phase, "PLACE")
        self.assertFalse(task.done)

    def test_policy_state_keeps_original_joints_and_excludes_scene_dofs(self) -> None:
        model, data = self.scene.model, self.scene.data
        original = mujoco.MjModel.from_xml_path(str(SCENE.with_name("mini3.xml")))
        expected_names = [original.joint(index).name for index in range(1, original.njnt)]
        self.assertEqual(self.scene.names, expected_names)
        expected = np.linspace(-0.1, 0.1, 21)
        for name, value in zip(expected_names, expected):
            data.qpos[model.joint(name).qposadr[0]] = value
        data.qvel[:] = np.arange(model.nv) * 0.01
        for side in ("left", "right"):
            for finger in ("finger", "follower"):
                data.qpos[model.joint(f"{side}_gripper_{finger}_joint").qposadr[0]] = 0.017
        data.time = 0.12
        self.scene.sync()
        self.assertEqual(self.scene.state.qpos.shape, (28,))
        self.assertEqual(self.scene.state.qvel.shape, (27,))
        np.testing.assert_allclose(self.scene.state.joint_pos, expected)
        for index, name in enumerate(expected_names):
            joint = model.joint(name)
            self.assertAlmostEqual(float(self.scene.state.joint_vel[index]),
                                   float(data.qvel[joint.dofadr[0]]), places=6)
            self.assertEqual(int(model.actuator_trnid[self.scene.aids[index], 0]), joint.id)
        self.assertEqual(self.scene.state.low_state_tick, 120)

    def test_motor_and_gripper_commands_do_not_reposition_simulated_bodies(self) -> None:
        scene = self.scene
        before_position, before_velocity = scene.data.qpos.copy(), scene.data.qvel.copy()
        scene.openings[:] = [0.035, 0.0]
        scene.arm_target = np.array([scene.data.qpos[0] + 0.23, scene.data.qpos[1] - 0.24,
                                     scene.data.xpos[scene.model.body("pick_cube_3").id, 2] + 0.008])
        torque = np.linspace(-0.5, 0.5, 21)
        # Removing the physics integrator must remove every source of body motion.
        # IK and phase controls may set commands, but cannot teleport a robot/cube.
        with patch.object(scene.motor, "compute", return_value=torque) as compute:
            with patch("mujoco.mj_step") as integrate:
                scene.step()
        self.assertEqual(compute.call_count, 10)
        self.assertEqual(integrate.call_count, 10)
        np.testing.assert_array_equal(scene.data.qpos, before_position)
        np.testing.assert_array_equal(scene.data.qvel, before_velocity)
        np.testing.assert_allclose(scene.data.ctrl[scene.aids], torque)
        self.assertAlmostEqual(float(scene.data.ctrl[scene.model.actuator("left_gripper_ctrl").id]), 0.035)
        self.assertEqual(float(scene.data.ctrl[scene.model.actuator("right_gripper_ctrl").id]), 0.0)
        np.testing.assert_array_equal(compute.call_args.args[1], before_position[scene.qids])
        np.testing.assert_array_equal(compute.call_args.args[2], before_velocity[scene.vids])

    def test_arm_ik_reaches_table_height_without_mutating_physics_state(self) -> None:
        scene = self.scene
        before_position, before_velocity = scene.data.qpos.copy(), scene.data.qvel.copy()
        target = np.array([scene.data.qpos[0] + 0.23, scene.data.qpos[1] - 0.24,
                           scene.data.xpos[scene.model.body("pick_cube_3").id, 2] + 0.008])
        target_joints = scene.ik.solve(scene.data, target)
        verification = mujoco.MjData(scene.model)
        verification.qpos[:] = before_position
        verification.qpos[scene.ik.qids] = target_joints
        mujoco.mj_forward(scene.model, verification)
        # Evaluate the actual two pad geom centres, independently of the TCP
        # offset formula in ArmIK.position(). Their midpoint is the grasp centre.
        finger_ids = [scene.model.geom(f"right_gripper_finger_{side}_geom").id
                      for side in ("positive", "negative")]
        pad_midpoint = verification.geom_xpos[finger_ids].mean(axis=0)
        np.testing.assert_allclose(pad_midpoint, target, atol=0.0001)
        np.testing.assert_allclose(scene.ik.position(verification), pad_midpoint, atol=1e-10)
        low, high = scene.model.jnt_range[scene.ik.jids].T
        self.assertTrue(np.all(target_joints >= low))
        self.assertTrue(np.all(target_joints <= high))
        np.testing.assert_array_equal(scene.data.qpos, before_position)
        np.testing.assert_array_equal(scene.data.qvel, before_velocity)

    def test_task_starts_three_metres_before_blue_with_free_objects_and_cameras(self) -> None:
        model, data = self.scene.model, self.scene.data
        blue = data.xpos[model.body("pick_cube_3").id]
        base = data.xpos[model.body("base_link").id]
        self.assertAlmostEqual(float(blue[0] - base[0]), 3.0)
        self.assertFalse(data.eq_active[model.equality("base_support_weld").id])
        for index in range(1, 4):
            joint = model.joint(f"pick_cube_{index}_free")
            self.assertEqual(int(joint.type[0]), mujoco.mjtJoint.mjJNT_FREE)
        welds = [index for index in range(model.neq)
                 if model.eq_type[index] == mujoco.mjtEq.mjEQ_WELD]
        self.assertEqual(welds, [model.equality("base_support_weld").id])
        for name in ("head_rgb", "left_gripper_rgb", "right_gripper_rgb"):
            self.assertGreaterEqual(model.camera(name).id, 0)

    def test_reset_refreshes_body_and_camera_poses_before_controller_reads_them(self) -> None:
        scene = self.scene
        scene.data.qpos[:3] += [0.5, 0.4, 0.3]
        mujoco.mj_forward(scene.model, scene.data)
        scene.reset()
        expected = mujoco.MjData(scene.model)
        mujoco.mj_resetDataKeyframe(scene.model, expected, scene.model.key("home").id)
        mujoco.mj_forward(scene.model, expected)
        np.testing.assert_allclose(scene.data.xpos, expected.xpos)
        np.testing.assert_allclose(scene.data.site_xpos, expected.site_xpos)
        np.testing.assert_allclose(scene.data.cam_xpos, expected.cam_xpos)

    def test_initialization_cannot_teleport_robot_after_physics_has_started(self) -> None:
        scene = self.scene
        initial = np.r_[scene.data.qpos[:7], scene.data.qpos[scene.qids]]
        scene.initialize_robot(initial)
        scene.step()
        before_position, before_velocity = scene.data.qpos.copy(), scene.data.qvel.copy()
        requested = initial.copy()
        requested[:3] += [1.0, 0.5, 0.2]
        with self.assertRaisesRegex(RuntimeError, "before the first physics step"):
            scene.initialize_robot(requested)
        np.testing.assert_array_equal(scene.data.qpos, before_position)
        np.testing.assert_array_equal(scene.data.qvel, before_velocity)

    def test_arm_integral_cannot_command_beyond_joint_limits(self) -> None:
        scene = self.scene
        limits = scene.model.jnt_range[scene.ik.jids]
        for bound_index, sign in ((0, -1), (1, 1)):
            with self.subTest(bound=bound_index):
                bound = limits[:, bound_index].copy()
                scene.data.qpos[scene.ik.qids] = bound
                mujoco.mj_forward(scene.model, scene.data)
                scene.arm_target = scene.ik.position(scene.data).copy()
                scene.arm_command = bound.copy()
                scene.arm_integral[:] = sign * 0.12
                with patch.object(scene.ik, "solve", return_value=bound):
                    with patch.object(scene.motor, "compute", return_value=np.zeros(21)) as compute:
                        with patch("mujoco.mj_step"):
                            scene.step()
                commanded = compute.call_args.args[0][scene.arm_indices]
                self.assertTrue(np.all(commanded >= limits[:, 0]))
                self.assertTrue(np.all(commanded <= limits[:, 1]))
                np.testing.assert_allclose(commanded, bound)
        scene.arm_target = None
        with patch.object(scene.motor, "compute", return_value=np.zeros(21)):
            with patch("mujoco.mj_step"):
                scene.step()
        np.testing.assert_array_equal(scene.arm_integral, np.zeros(4))
        self.assertIsNone(scene.arm_command)

    def test_real_motor_response_is_active_and_support_is_rejected(self) -> None:
        self.assertTrue(self.scene.motor.response_enabled)
        self.assertTrue(self.scene.motor.tn_enabled)
        self.assertTrue(self.scene.motor.kt_enabled)
        self.scene.policy.target[:] = 0.02
        self.scene.step()
        self.assertGreater(float(np.linalg.norm(self.scene.motor.applied_torque)), 0.0)
        self.assertGreater(float(np.linalg.norm(self.scene.motor.response_torque)), 0.0)
        self.assertAlmostEqual(float(self.scene.data.time), 0.02)
        self.scene.data.eq_active[self.scene.model.equality("base_support_weld").id] = True
        with self.assertRaisesRegex(RuntimeError, "support must stay disabled"):
            self.scene.step()

    def test_basket_success_requires_contained_stationary_cube_on_bottom(self) -> None:
        scene = self.scene
        task = object.__new__(carry.PickCarryTask)
        task.r = scene
        task.cube = scene.model.body("pick_cube_3").id
        task.cube_geom = scene.model.geom("pick_cube_3_geom").id
        task.basket = scene.model.body("pick_basket").id
        joint = scene.model.joint("pick_cube_3_free")
        qpos_address, qvel_address = int(joint.qposadr[0]), int(joint.dofadr[0])
        basket_position = scene.data.xpos[task.basket].copy()
        basket_rotation = scene.data.xmat[task.basket].reshape(3, 3).copy()
        basket_quaternion = scene.data.xquat[task.basket].copy()
        # MuJoCo's compliant resting contact has a tiny penetration; exact
        # mathematical touching can produce no contact due to roundoff.
        rest_height = 0.03 - 1e-5
        cases = (
            ("bottom_rest", (0, 0, rest_height), 0, (0, 0, 0, 0, 0, 0), True),
            ("near_wall_unrotated", (0.097, 0, rest_height), 0, (0, 0, 0, 0, 0, 0), True),
            ("rotated_corner_outside", (0.097, 0, rest_height), np.pi / 4, (0, 0, 0, 0, 0, 0), False),
            ("hovering_five_mm", (0, 0, 0.035), 0, (0, 0, 0, 0, 0, 0), False),
            ("still_translating", (0, 0, rest_height), 0, (0.08, 0, 0, 0, 0, 0), False),
            ("still_rotating", (0, 0, rest_height), 0, (0, 0, 0, 0, 0, 0.5), False),
        )
        for name, relative_position, yaw, velocity, expected in cases:
            with self.subTest(case=name):
                local_quaternion = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
                quaternion = np.empty(4)
                mujoco.mju_mulQuat(quaternion, basket_quaternion, local_quaternion)
                scene.data.qpos[qpos_address:qpos_address + 3] = (
                    basket_position + basket_rotation @ relative_position
                )
                scene.data.qpos[qpos_address + 3:qpos_address + 7] = quaternion
                scene.data.qvel[qvel_address:qvel_address + 6] = velocity
                mujoco.mj_forward(scene.model, scene.data)
                self.assertEqual(task.cube_in_basket(), expected)


class Mini3PickCarrySchemaTest(unittest.TestCase):
    def test_accepts_root_only_tracking_command_with_string_or_list_body_selection(self) -> None:
        for selection in ("base_link", ["base_link"]):
            with self.subTest(selection=selection):
                config = compatible_policy_config()
                config["observation"]["command"]["ref_root_pos_future_local"]["body_names"] = selection
                before = deepcopy(config)
                carry.validate_policy_schema(config)
                self.assertEqual(config, before)

    def test_rejects_reference_body_observations_that_require_updated_arm_fk(self) -> None:
        for selection in (".*", ["base_link", "right_elbow_pitch_link"], "right_elbow_pitch_link"):
            with self.subTest(selection=selection):
                config = compatible_policy_config()
                config["observation"]["command"]["ref_root_pos_future_local"]["body_names"] = selection
                with self.assertRaises(ValueError):
                    carry.validate_policy_schema(config)
        config = compatible_policy_config()
        config["observation"]["policy"]["joint_pos_history"]["_target_"] = "mimic_lite.ref_body_pos_future_local"
        with self.assertRaises(ValueError):
            carry.validate_policy_schema(config)

    def test_rejects_missing_extra_or_mislabelled_command_components(self) -> None:
        config = compatible_policy_config()
        del config["observation"]["command"]["ref_root_lin_vel_future_local"]
        with self.assertRaises(ValueError):
            carry.validate_policy_schema(config)
        config = compatible_policy_config()
        config["observation"]["command"]["extra"] = {"_target_": "mimic_lite.ref_body_ori_future_local"}
        with self.assertRaises(ValueError):
            carry.validate_policy_schema(config)
        config = compatible_policy_config()
        config["observation"]["command"]["ref_joint_pos_future"]["_target_"] = "mimic_lite.ref_body_pos_future_local"
        with self.assertRaises(ValueError):
            carry.validate_policy_schema(config)

    def test_rejects_other_root_or_anchor_frames_and_extra_policy_groups(self) -> None:
        for key in ("root_body_name", "anchor_body_name"):
            with self.subTest(key=key):
                config = compatible_policy_config()
                config["motion"][key] = "right_elbow_pitch_link"
                with self.assertRaises(ValueError):
                    carry.validate_policy_schema(config)
        config = compatible_policy_config()
        config["observation"]["extra"] = {"ref_body_pos": {"_target_": "mimic_lite.ref_body_pos_future_local"}}
        with self.assertRaises(ValueError):
            carry.validate_policy_schema(config)


if __name__ == "__main__":
    unittest.main()
