"""Lightweight mujoco_viewer display isolated from policy and physics state."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from preview_mini3_pick_scene import CAMERA_SHORTCUTS, configure_camera

if TYPE_CHECKING:
    import mujoco


class Mini3TaskViewer:
    """Render a private model/data copy and leave simulation timing to the caller.

    mujoco_viewer.render() applies mouse perturbations to its data, and visual
    shortcuts can change its model. Neither object is shared with the policy.
    The package's internal pause loop and speed control are disabled so that
    one render call returns after one frame even while the task is paused.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *,
                 camera: str = "overview", start_paused: bool = False) -> None:
        import glfw
        import mujoco
        try:
            import mujoco_viewer
        except ImportError as exc:
            raise RuntimeError("Install mujoco-python-viewer in the active Python environment") from exc

        self._glfw = glfw
        self.model = copy.copy(model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.data, self.model, data)
        mujoco.mj_forward(self.model, self.data)
        self.paused = bool(start_paused)
        self.follow = True
        self.quit_requested = False
        self.task_label = ""
        self._closed = False
        self._viewer = mujoco_viewer.MujocoViewer(
            self.model, self.data, mode="window", width=1280, height=960,
            title="Mini3 pick & carry (mujoco_viewer)", hide_menus=True,
        )
        # The original STL meshes show shadow-map acne. Disable these display
        # passes to keep the forearm geometry visible without changing physics.
        self._viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self._viewer.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        # MuJoCo 3.11 removed the scene argument from mjv_moveCamera; the
        # installed viewer still calls the older signature from mouse events.
        self._camera_uses_scene = "scn:" in (mujoco.mjv_moveCamera.__doc__ or "")
        self._mouse_buttons: set[int] = set()
        self._cursor_position = (0.0, 0.0)
        self._last_click_times: dict[int, float] = {}
        self._default_key_callback = self._viewer._key_callback
        self._disable_internal_timing()
        configure_camera(self._viewer.cam, self.model, self.data, camera)
        self._viewer.cam.distance = 2.4
        self._viewer.vopt.geomgroup[1] = 0
        # GLFW retains the bound callback; keep the adapter alive until close.
        glfw.set_key_callback(self._viewer.window, self._key_callback)
        glfw.set_cursor_pos_callback(self._viewer.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self._viewer.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self._viewer.window, self._scroll_callback)
        self._default_overlay = self._viewer._create_overlay
        self._viewer._create_overlay = self._create_overlay

    def _disable_internal_timing(self) -> None:
        # None also disables the package's Space/right-arrow pause callbacks.
        self._viewer._paused = None
        self._viewer._advance_by_one_step = False
        self._viewer._render_every_frame = True
        self._viewer._run_speed = 1.0

    def _key_callback(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        controlled = {ord("P"), ord("F"), self._glfw.KEY_ESCAPE, *CAMERA_SHORTCUTS}
        if key in controlled:
            if action != self._glfw.PRESS:
                return
            if key == ord("P"):
                self.paused = not self.paused
            elif key == ord("F"):
                self.follow = not self.follow
            elif key in CAMERA_SHORTCUTS:
                configure_camera(self._viewer.cam, self.model, self.data, CAMERA_SHORTCUTS[key])
                if CAMERA_SHORTCUTS[key] == "overview":
                    self._viewer.cam.distance = 2.4
            else:
                self.quit_requested = True
                self._glfw.set_window_should_close(window, True)
            return
        # Preserve camera/mouse/display controls, while ignoring the native
        # playback shortcuts that could block render or change frame pacing.
        if key in (self._glfw.KEY_SPACE, self._glfw.KEY_RIGHT, ord("D")):
            return
        if key == ord("S") and mods != self._glfw.MOD_CONTROL:
            return
        self._default_key_callback(window, key, scancode, action, mods)
        self._disable_internal_timing()

    def _create_overlay(self) -> None:
        import mujoco
        self._default_overlay()
        self._viewer._overlay[mujoco.mjtGridPos.mjGRID_TOPRIGHT] = [
            "Mini3 task\nTarget\nP: pause/resume\nF: follow\nF8 / F9 / F10 / F11\nEsc\n",
            f"{'PAUSED' if self.paused else 'RUNNING'}\n{self.task_label or '-'}\n\n"
            f"{'on' if self.follow else 'off'}\noverview / head / left / right\nexit\n",
        ]

    def _move_camera(self, action: int, dx: float, dy: float) -> None:
        import mujoco
        if self._camera_uses_scene:
            mujoco.mjv_moveCamera(self.model, action, dx, dy, self._viewer.scn, self._viewer.cam)
        else:
            mujoco.mjv_moveCamera(self.model, action, dx, dy, self._viewer.cam)

    def _cursor_pos_callback(self, window, xpos: float, ypos: float) -> None:
        import mujoco
        previous_x, previous_y = self._cursor_position
        self._cursor_position = (xpos, ypos)
        if not self._mouse_buttons:
            return
        width, height = self._glfw.get_window_size(window)
        if width <= 0 or height <= 0:
            return
        shift = any(self._glfw.get_key(window, key) == self._glfw.PRESS for key in
                    (self._glfw.KEY_LEFT_SHIFT, self._glfw.KEY_RIGHT_SHIFT))
        if self._glfw.MOUSE_BUTTON_RIGHT in self._mouse_buttons:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._glfw.MOUSE_BUTTON_LEFT in self._mouse_buttons:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        dx, dy = (xpos - previous_x) / height, (ypos - previous_y) / height
        viewer = self._viewer
        with viewer._gui_lock:
            if viewer.pert.active:
                mujoco.mjv_movePerturb(self.model, self.data, action, dx, dy, viewer.scn, viewer.pert)
            else:
                self._move_camera(action, dx, dy)
                if self._glfw.MOUSE_BUTTON_RIGHT in self._mouse_buttons:
                    # Preserve a manually chosen focus instead of overwriting
                    # the user's pan on the next rendered frame.
                    self.follow = False

    def _scroll_callback(self, window, x_offset: float, y_offset: float) -> None:
        import mujoco
        with self._viewer._gui_lock:
            self._move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * y_offset)

    def _mouse_button_callback(self, window, button: int, action: int, mods: int) -> None:
        import mujoco
        import numpy as np
        glfw, viewer = self._glfw, self._viewer
        self._cursor_position = glfw.get_cursor_pos(window)
        if action == glfw.RELEASE:
            self._mouse_buttons.discard(button)
            viewer.pert.active = 0
            return
        if action != glfw.PRESS:
            return
        if button not in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT, glfw.MOUSE_BUTTON_MIDDLE):
            return
        self._mouse_buttons.add(button)
        control = bool(mods & glfw.MOD_CONTROL)
        perturb = 0
        if control and viewer.pert.select > 0:
            if button == glfw.MOUSE_BUTTON_LEFT:
                perturb = mujoco.mjtPertBit.mjPERT_ROTATE
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                perturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if perturb and not viewer.pert.active:
                mujoco.mjv_initPerturb(self.model, self.data, viewer.scn, viewer.pert)
        viewer.pert.active = perturb

        now = glfw.get_time()
        previous = self._last_click_times.get(button)
        self._last_click_times[button] = now
        threshold = 0.3 if button == glfw.MOUSE_BUTTON_LEFT else 0.2
        if (button == glfw.MOUSE_BUTTON_MIDDLE or previous is None
                or not 0.01 < now - previous < threshold):
            return
        width, height = glfw.get_window_size(window)
        if width <= 0 or height <= 0:
            return
        x, y = self._cursor_position
        point = np.zeros(3)
        geom, flex, skin = (np.full(1, -1, dtype=np.int32) for _ in range(3))
        body = mujoco.mjv_select(self.model, self.data, viewer.vopt, width / height,
                                x / width, 1.0 - y / height, viewer.scn, point, geom, flex, skin)
        if button == glfw.MOUSE_BUTTON_RIGHT:
            if body >= 0:
                viewer.cam.lookat[:] = point
                self.follow = False
            if control and body > 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = body
                viewer.cam.fixedcamid = -1
        elif body >= 0:
            viewer.pert.select = body
            # MuJoCo expects an integer, not the 1x1 NumPy array passed by the
            # package callback. Express the selected point in the body frame.
            viewer.pert.skinselect = int(skin.item())
            viewer.pert.localpos[:] = self.data.xmat[body].reshape(3, 3).T @ (point - self.data.xpos[body])
        else:
            viewer.pert.select = 0
            viewer.pert.skinselect = -1
        viewer.pert.active = 0

    def poll_events(self) -> None:
        if not self._closed:
            self._glfw.poll_events()

    def is_running(self) -> bool:
        return bool(not self._closed and not self.quit_requested and self._viewer.is_alive
                    and not self._glfw.window_should_close(self._viewer.window))

    def render(self, data: mujoco.MjData) -> None:
        import mujoco
        if not self.is_running():
            return
        mujoco.mj_copyData(self.data, self.model, data)
        mujoco.mj_forward(self.model, self.data)
        if self.follow and self._viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
            self._viewer.cam.lookat[:] = self.data.qpos[:3] + [0.3, 0, 0]
        self._disable_internal_timing()
        self._viewer.render()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._viewer.is_alive:
                self._viewer.close()
