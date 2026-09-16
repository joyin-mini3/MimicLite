from __future__ import annotations

import builtins
from pathlib import Path
import sys
from threading import Lock
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mini3_pick_carry as carry
from mini3_task_viewer import Mini3TaskViewer


MODEL_XML = """
<mujoco><option timestep="0.002"/><worldbody>
  <body name="base_link" pos="0 0 0.5"><freejoint/>
    <geom type="sphere" size="0.08" mass="1"/>
    <camera name="head_rgb"/><camera name="left_gripper_rgb"/><camera name="right_gripper_rgb"/>
    <body><joint name="hinge"/><geom type="sphere" size="0.02" mass="0.1"/></body>
  </body>
</worldbody><actuator><motor joint="hinge"/></actuator></mujoco>
"""


class FakeGlfw(ModuleType):
    PRESS, RELEASE, REPEAT = 1, 0, 2
    KEY_ESCAPE, KEY_SPACE, KEY_RIGHT, MOD_CONTROL = 256, 32, 262, 2
    KEY_LEFT_SHIFT, KEY_RIGHT_SHIFT = 340, 344
    MOD_SHIFT = 1
    MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT, MOUSE_BUTTON_MIDDLE = 0, 1, 2

    def __init__(self):
        super().__init__("glfw")
        self.callback = None
        self.batches = []
        self.polls = 0
        self.cursor = (400, 300)
        self.size = (800, 600)
        self.now = 1.0
        self.keys, self.buttons = {}, {}

    def set_key_callback(self, window, callback):
        self.window, self.callback = window, callback

    def set_cursor_pos_callback(self, window, callback):
        self.cursor_callback = callback

    def set_mouse_button_callback(self, window, callback):
        self.mouse_callback = callback

    def set_scroll_callback(self, window, callback):
        self.scroll_callback = callback

    def get_cursor_pos(self, window):
        return self.cursor

    def get_framebuffer_size(self, window):
        return self.size

    def get_window_size(self, window):
        return self.size

    def get_time(self):
        return self.now

    def get_key(self, window, key):
        return self.keys.get(key, self.RELEASE)

    def get_mouse_button(self, window, button):
        return self.buttons.get(button, self.RELEASE)

    def poll_events(self):
        self.polls += 1
        if self.batches:
            for key, action, mods in self.batches.pop(0):
                self.callback(self.window, key, 0, action, mods)

    def window_should_close(self, window):
        return window.should_close

    def set_window_should_close(self, window, value):
        window.should_close = value


class PerturbingViewer:
    """Deliberately mutate everything that must remain private to the viewer."""

    instances = []

    def __init__(self, model, data, **kwargs):
        self.model, self.data = model, data
        self.window = SimpleNamespace(should_close=False)
        self.cam, self.vopt = mujoco.MjvCamera(), mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.scn = mujoco.MjvScene(model, maxgeom=100)
        self.viewport = mujoco.MjrRect(0, 0, 800, 600)
        self._gui_lock = Lock()
        self._scale = 1.0
        self._button_left_pressed = self._button_right_pressed = False
        self._button_middle_pressed = False
        self._last_left_click_time = self._last_right_click_time = None
        self._last_mouse_x = self._last_mouse_y = 0
        self.is_alive, self._paused = True, False
        self._overlay = {}
        self.frames, self.forwarded_keys = [], []
        self.close_count = 0
        type(self).instances.append(self)

    def _key_callback(self, window, key, scancode, action, mods):
        self.forwarded_keys.append(key)
        if action == FakeGlfw.RELEASE and key == ord("R"):
            self.model.geom_rgba[:, 3] *= 0.5

    def _create_overlay(self):
        pass

    def render(self):
        assert self._paused is None, "Native pause could trap the caller in a render loop"
        assert self._render_every_frame and self._run_speed == 1
        self._create_overlay()
        self.frames.append(self.data.qpos.copy())
        self.data.qpos[:3] += 10
        self.data.qvel[:] = 99
        self.data.ctrl[:] = 88
        self.data.qfrc_applied[:] = 77
        self.data.xfrc_applied[:] = 66
        self.model.geom_rgba[:] = 0.25

    def close(self):
        self.close_count += 1
        self.is_alive = False


class RuntimeDouble:
    instances = []

    def __init__(self, scene, policy, motion):
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        type(self).instances.append(self)


class TaskDouble:
    instances = []
    finish_after = 2

    def __init__(self, runtime, motion, **kwargs):
        self.r = runtime
        self.done = self.success = False
        self.failure = None
        self.steps, self.saves = 0, 0
        type(self).instances.append(self)

    def step(self):
        self.steps += 1
        self.r.data.time += 0.02
        if self.steps >= self.finish_after:
            self.done = self.success = True

    def fail(self, reason):
        self.failure, self.done = reason, True

    def save(self, output):
        self.saves += 1


class Mini3TaskViewerTest(unittest.TestCase):
    def setUp(self):
        self.glfw = FakeGlfw()
        package = ModuleType("mujoco_viewer")
        package.MujocoViewer = PerturbingViewer
        packages = patch.dict(sys.modules, {"glfw": self.glfw, "mujoco_viewer": package})
        packages.start()
        self.addCleanup(packages.stop)
        original_import = builtins.__import__

        def no_native_viewer(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mujoco.viewer" or (name == "mujoco" and "viewer" in (fromlist or ())):
                raise AssertionError("The native MuJoCo viewer must not be imported")
            return original_import(name, globals, locals, fromlist, level)

        import_guard = patch("builtins.__import__", side_effect=no_native_viewer)
        import_guard.start()
        self.addCleanup(import_guard.stop)
        PerturbingViewer.instances.clear()
        RuntimeDouble.instances.clear()
        TaskDouble.instances.clear()
        TaskDouble.finish_after = 2
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def press(self, key, action=FakeGlfw.PRESS):
        self.glfw.callback(self.glfw.window, key, 0, action, 0)

    def mouse(self, button, action=FakeGlfw.PRESS, mods=0, *, at=None):
        self.glfw.now = self.glfw.now + 0.5 if at is None else at
        self.glfw.buttons[button] = action
        self.glfw.mouse_callback(self.glfw.window, button, action, mods)

    def move(self, x, y):
        self.glfw.cursor = (x, y)
        self.glfw.cursor_callback(self.glfw.window, x, y)

    def prepare_pick_scene(self, viewer):
        display = viewer._viewer
        mujoco.mjv_defaultCamera(display.cam)
        display.cam.lookat[:] = [0, 0, 0.5]
        display.cam.distance, display.cam.azimuth, display.cam.elevation = 2.0, 90, -20
        mujoco.mjv_updateScene(viewer.model, viewer.data, display.vopt, display.pert,
                              display.cam, mujoco.mjtCatBit.mjCAT_ALL, display.scn)

    def double_click(self, button, *, mods=0):
        start = self.glfw.now + 1
        self.mouse(button, mods=mods, at=start)
        self.mouse(button, FakeGlfw.RELEASE, mods=mods, at=start + 0.05)
        self.mouse(button, mods=mods, at=start + 0.1)

    def test_mouse_rotation_pan_and_zoom_use_real_mujoco_camera_api(self):
        viewer = Mini3TaskViewer(self.model, self.data)
        self.addCleanup(viewer.close)
        camera = viewer._viewer.cam
        self.mouse(FakeGlfw.MOUSE_BUTTON_LEFT)
        old_angles = np.array([camera.azimuth, camera.elevation])
        self.move(440, 325)
        self.assertFalse(np.array_equal([camera.azimuth, camera.elevation], old_angles))
        self.mouse(FakeGlfw.MOUSE_BUTTON_LEFT, FakeGlfw.RELEASE)

        self.glfw.keys[FakeGlfw.KEY_LEFT_SHIFT] = FakeGlfw.PRESS
        self.mouse(FakeGlfw.MOUSE_BUTTON_LEFT)
        old_azimuth = camera.azimuth
        self.move(460, 335)
        self.assertNotEqual(camera.azimuth, old_azimuth)
        self.mouse(FakeGlfw.MOUSE_BUTTON_LEFT, FakeGlfw.RELEASE)
        self.glfw.keys.clear()

        for shift in (False, True):
            with self.subTest(right_drag_shift=shift):
                viewer.follow = True
                if shift:
                    self.glfw.keys[FakeGlfw.KEY_RIGHT_SHIFT] = FakeGlfw.PRESS
                self.mouse(FakeGlfw.MOUSE_BUTTON_RIGHT)
                old_lookat = camera.lookat.copy()
                self.move(self.glfw.cursor[0] + 20, self.glfw.cursor[1] + 20)
                self.assertFalse(np.array_equal(camera.lookat, old_lookat))
                self.assertFalse(viewer.follow)
                self.mouse(FakeGlfw.MOUSE_BUTTON_RIGHT, FakeGlfw.RELEASE)
                self.glfw.keys.clear()

        old_distance = camera.distance
        self.glfw.scroll_callback(self.glfw.window, 0, 1)
        self.assertNotEqual(camera.distance, old_distance)
        self.mouse(FakeGlfw.MOUSE_BUTTON_MIDDLE)
        old_distance = camera.distance
        self.move(self.glfw.cursor[0], self.glfw.cursor[1] + 30)
        self.assertNotEqual(camera.distance, old_distance)
        self.mouse(FakeGlfw.MOUSE_BUTTON_MIDDLE, FakeGlfw.RELEASE)
        np.testing.assert_array_equal(self.data.qpos, self.model.qpos0)
        self.assertTrue(viewer.is_running())

    def test_double_click_select_hit_and_miss_use_real_mujoco_selection_api(self):
        viewer = Mini3TaskViewer(self.model, self.data)
        self.addCleanup(viewer.close)
        self.prepare_pick_scene(viewer)
        self.double_click(FakeGlfw.MOUSE_BUTTON_LEFT)
        self.assertEqual(viewer._viewer.pert.select, self.model.body("base_link").id)
        self.assertEqual(viewer._viewer.pert.skinselect, -1)
        self.assertEqual(viewer._viewer.pert.active, 0)
        self.glfw.cursor = (0, 0)
        self.double_click(FakeGlfw.MOUSE_BUTTON_LEFT)
        self.assertEqual(viewer._viewer.pert.select, 0)
        self.assertEqual(viewer._viewer.pert.skinselect, -1)

    def test_double_right_click_focus_disables_follow_and_ctrl_selects_tracking(self):
        viewer = Mini3TaskViewer(self.model, self.data)
        self.addCleanup(viewer.close)
        self.prepare_pick_scene(viewer)
        before = viewer._viewer.cam.lookat.copy()
        self.double_click(FakeGlfw.MOUSE_BUTTON_RIGHT)
        self.assertFalse(viewer.follow)
        self.assertFalse(np.array_equal(before, viewer._viewer.cam.lookat))
        self.prepare_pick_scene(viewer)
        self.double_click(FakeGlfw.MOUSE_BUTTON_RIGHT, mods=FakeGlfw.MOD_CONTROL)
        self.assertEqual(viewer._viewer.cam.type, mujoco.mjtCamera.mjCAMERA_TRACKING)
        self.assertEqual(viewer._viewer.cam.trackbodyid, self.model.body("base_link").id)

    def test_ctrl_mouse_perturbation_only_changes_viewer_copy(self):
        self.data.qfrc_applied[:] = 0.4
        self.data.xfrc_applied[:] = 0.5
        arrays = ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")
        before = {name: getattr(self.data, name).copy() for name in arrays}
        viewer = Mini3TaskViewer(self.model, self.data)
        self.addCleanup(viewer.close)
        self.prepare_pick_scene(viewer)
        self.double_click(FakeGlfw.MOUSE_BUTTON_LEFT)
        self.mouse(FakeGlfw.MOUSE_BUTTON_LEFT, FakeGlfw.RELEASE)
        self.mouse(FakeGlfw.MOUSE_BUTTON_RIGHT, mods=FakeGlfw.MOD_CONTROL)
        self.assertEqual(viewer._viewer.pert.active, mujoco.mjtPertBit.mjPERT_TRANSLATE)
        old_position = viewer._viewer.pert.refpos.copy()
        self.move(460, 335)
        self.assertFalse(np.array_equal(viewer._viewer.pert.refpos, old_position))
        mujoco.mjv_applyPerturbForce(viewer.model, viewer.data, viewer._viewer.pert)
        self.mouse(FakeGlfw.MOUSE_BUTTON_RIGHT, FakeGlfw.RELEASE)
        self.assertEqual(viewer._viewer.pert.active, 0)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(self.data, name), value)

    def test_mouse_callbacks_handle_minimized_window_without_dividing_by_zero(self):
        viewer = Mini3TaskViewer(self.model, self.data)
        self.addCleanup(viewer.close)
        self.glfw.size = (0, 0)
        viewer._viewer.viewport.width = viewer._viewer.viewport.height = 0
        self.mouse(FakeGlfw.MOUSE_BUTTON_LEFT)
        self.move(460, 335)
        self.double_click(FakeGlfw.MOUSE_BUTTON_LEFT)
        self.glfw.scroll_callback(self.glfw.window, 0, 1)
        camera = viewer._viewer.cam
        self.assertTrue(np.isfinite([camera.azimuth, camera.elevation, camera.distance]).all())
        self.assertTrue(np.isfinite(camera.lookat).all())

    def test_main_records_viewer_api_typeerror_and_closes_window(self):
        with patch.object(carry, "setup_paths"), patch.object(carry, "resolve_policy", return_value=Path("policy.yaml")):
            with patch.object(PerturbingViewer, "render", side_effect=TypeError("incompatible mouse API")):
                with patch.object(carry.time, "sleep"), patch.object(sys, "argv", ["mini3_pick_carry.py"]):
                    result = carry.main(runtime_class=RuntimeDouble, task_class=TaskDouble)
        task = TaskDouble.instances[-1]
        self.assertEqual(result, 1)
        self.assertEqual(task.saves, 1)
        self.assertIn("incompatible mouse API", task.failure)
        self.assertEqual(PerturbingViewer.instances[-1].close_count, 1)

    def test_render_and_visual_shortcuts_cannot_modify_physics_or_model(self):
        self.data.ctrl[:] = 0.3
        self.data.qfrc_applied[:] = 0.4
        self.data.xfrc_applied[:] = 0.5
        arrays = ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")
        expected = {name: getattr(self.data, name).copy() for name in arrays}
        rgba = self.model.geom_rgba.copy()
        viewer = Mini3TaskViewer(self.model, self.data)
        self.addCleanup(viewer.close)
        self.assertIsNot(viewer.model, self.model)
        self.assertIsNot(viewer.data, self.data)
        with patch("mujoco.mj_step", side_effect=AssertionError("Rendering must not integrate physics")):
            viewer.render(self.data)
            self.press(ord("R"), FakeGlfw.RELEASE)
            viewer.render(self.data)
        for name, value in expected.items():
            np.testing.assert_array_equal(getattr(self.data, name), value)
        np.testing.assert_array_equal(self.model.geom_rgba, rgba)
        np.testing.assert_array_equal(viewer._viewer.frames[0], expected["qpos"])
        np.testing.assert_array_equal(viewer._viewer.frames[1], expected["qpos"])

    def test_pause_is_owned_by_task_and_key_repeat_does_not_toggle_twice(self):
        viewer = Mini3TaskViewer(self.model, self.data, start_paused=True)
        self.addCleanup(viewer.close)
        viewer.render(self.data)
        self.assertTrue(viewer.paused)
        self.assertIsNone(viewer._viewer._paused)
        self.press(ord("P"))
        self.press(ord("P"), FakeGlfw.REPEAT)
        self.press(ord("P"), FakeGlfw.RELEASE)
        self.assertFalse(viewer.paused)
        for key in (FakeGlfw.KEY_SPACE, FakeGlfw.KEY_RIGHT, ord("D"), ord("S")):
            self.press(key, FakeGlfw.RELEASE)
        self.assertFalse(viewer.paused)
        self.assertEqual(viewer._viewer.forwarded_keys, [])
        viewer._viewer._paused = True
        viewer._viewer._render_every_frame = False
        viewer.render(self.data)
        self.assertEqual(len(viewer._viewer.frames), 2)

    def test_camera_shortcuts_follow_and_escape_leave_simulation_untouched(self):
        viewer = Mini3TaskViewer(self.model, self.data)
        before = self.data.qpos.copy()
        for key, name in ((298, "head_rgb"), (299, "left_gripper_rgb"), (300, "right_gripper_rgb")):
            self.press(key)
            self.assertEqual(viewer._viewer.cam.type, mujoco.mjtCamera.mjCAMERA_FIXED)
            self.assertEqual(viewer._viewer.cam.fixedcamid, viewer.model.camera(name).id)
        self.press(297)
        viewer.render(self.data)
        np.testing.assert_allclose(viewer._viewer.cam.lookat, self.data.qpos[:3] + [0.3, 0, 0])
        self.press(ord("F"))
        lookat = viewer._viewer.cam.lookat.copy()
        self.data.qpos[0] += 0.1
        viewer.render(self.data)
        np.testing.assert_array_equal(viewer._viewer.cam.lookat, lookat)
        self.data.qpos[0] -= 0.1
        np.testing.assert_array_equal(self.data.qpos, before)
        self.press(FakeGlfw.KEY_ESCAPE)
        self.assertFalse(viewer.is_running())
        count = len(viewer._viewer.frames)
        viewer.render(self.data)
        self.assertEqual(len(viewer._viewer.frames), count)
        viewer.close()
        viewer.close()
        self.assertEqual(viewer._viewer.close_count, 1)

    def test_main_pause_resume_and_escape_save_without_an_extra_policy_step(self):
        TaskDouble.finish_after = 100
        self.glfw.batches = [[], [(ord("P"), 1, 0)], [(ord("P"), 1, 0)], [(256, 1, 0)]]
        with patch.object(carry, "setup_paths"), patch.object(carry, "resolve_policy", return_value=Path("policy.yaml")):
            with patch.object(carry.time, "sleep"), patch.object(sys, "argv", ["mini3_pick_carry.py", "--start-paused"]):
                result = carry.main(runtime_class=RuntimeDouble, task_class=TaskDouble)
        task = TaskDouble.instances[-1]
        self.assertEqual(result, 1)
        self.assertEqual(task.steps, 1)
        self.assertEqual(task.saves, 1)
        self.assertEqual(task.failure, "Stopped by viewer/user")
        self.assertEqual(task.visualization, "mujoco_viewer")
        self.assertAlmostEqual(task.r.data.time, 0.02)
        self.assertEqual(len(PerturbingViewer.instances[-1].frames), 3)
        self.assertEqual(PerturbingViewer.instances[-1].close_count, 1)
        np.testing.assert_array_equal(task.r.data.ctrl, np.zeros(task.r.model.nu))

    def test_headless_main_keeps_runtime_injection_and_never_imports_a_viewer(self):
        original_import = builtins.__import__

        def no_viewer(name, globals=None, locals=None, fromlist=(), level=0):
            if name in ("mujoco.viewer", "mujoco_viewer", "mini3_task_viewer", "glfw"):
                raise AssertionError("Headless execution must not load a viewer")
            return original_import(name, globals, locals, fromlist, level)

        with patch.object(carry, "setup_paths"), patch.object(carry, "resolve_policy", return_value=Path("policy.yaml")):
            with patch("builtins.__import__", side_effect=no_viewer):
                with patch.object(sys, "argv", ["mini3_pick_carry.py", "--headless"]):
                    result = carry.main(runtime_class=RuntimeDouble, task_class=TaskDouble)
        task = TaskDouble.instances[-1]
        self.assertEqual(result, 0)
        self.assertEqual(task.steps, 2)
        self.assertEqual(task.saves, 1)
        self.assertEqual(task.visualization, "headless")
        self.assertEqual(PerturbingViewer.instances, [])


if __name__ == "__main__":
    unittest.main()
