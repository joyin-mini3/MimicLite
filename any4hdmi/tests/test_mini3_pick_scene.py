from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np


ANY4HDMI_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ANY4HDMI_ROOT / "assets/robots/mini3_mjlab"
SCENE_XML = ASSET_ROOT / "scene_pick_place.xml"
GENERATOR = ANY4HDMI_ROOT / "scripts/generate_mini3_pick_scene.py"
CUBE_NAMES = tuple(f"pick_cube_{index}" for index in range(1, 4))


class Mini3PickSceneTest(unittest.TestCase):
    def make_scene(self, path: Path = SCENE_XML, *, support: bool = False):
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        if support:
            # Isolate prop dynamics from an uncontrolled, free-standing robot.
            data.eq_active[model.equality("base_support_weld").id] = True
        mujoco.mj_forward(model, data)
        return model, data

    def step(self, model, data, seconds: float) -> None:
        for _ in range(math.ceil(seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertTrue(np.isfinite(data.qvel).all())

    def cube_bottom(self, model, data, name: str) -> float:
        geom = model.geom(f"{name}_geom")
        rotation = data.geom_xmat[geom.id].reshape(3, 3)
        extent = np.abs(rotation[2]) @ model.geom_size[geom.id]
        return float(data.geom_xpos[geom.id, 2] - extent)

    def contacts_for(self, model, data, name: str) -> set[str]:
        geom_id = model.geom(f"{name}_geom").id
        partners = set()
        for contact in data.contact:
            if int(contact.geom1) == geom_id:
                partners.add(model.geom(int(contact.geom2)).name)
            elif int(contact.geom2) == geom_id:
                partners.add(model.geom(int(contact.geom1)).name)
        return partners

    def drop_cube(self, model, data) -> None:
        joint = model.joint("pick_cube_1_free")
        address = int(joint.qposadr[0])
        basket_position = data.xpos[model.body("pick_basket").id]
        data.qpos[address:address + 3] = basket_position + (0.0, 0.0, 0.30)
        data.qpos[address + 3:address + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        self.step(model, data, 1.4)

    def test_home_has_three_free_cubes_and_no_initial_intersections(self) -> None:
        model, data = self.make_scene()
        self.assertFalse(data.eq_active[model.equality("base_support_weld").id])
        world_free_joints = [
            model.joint(index).name
            for index in range(model.njnt)
            if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE
        ]
        self.assertCountEqual(
            world_free_joints,
            ["floating_base", *(f"{name}_free" for name in CUBE_NAMES)],
        )
        for name in CUBE_NAMES:
            self.assertAlmostEqual(float(model.body(name).mass[0]), 0.04)
            np.testing.assert_allclose(model.geom(f"{name}_geom").size, 0.02)
            self.assertGreaterEqual(self.cube_bottom(model, data, name), 0.0)
        for contact in data.contact:
            self.assertGreaterEqual(
                float(contact.dist), -1e-6,
                f"Initial intersection: {model.geom(int(contact.geom1)).name} / "
                f"{model.geom(int(contact.geom2)).name}",
            )

    def test_three_cubes_settle_on_the_ground(self) -> None:
        model, data = self.make_scene(support=True)
        initial_xy = {name: data.xpos[model.body(name).id, :2].copy() for name in CUBE_NAMES}
        self.step(model, data, 2.0)
        for name in CUBE_NAMES:
            with self.subTest(cube=name):
                position = data.xpos[model.body(name).id]
                np.testing.assert_allclose(position[:2], initial_xy[name], atol=0.001)
                self.assertAlmostEqual(float(position[2]), 0.02, delta=0.0005)
                self.assertGreaterEqual(self.cube_bottom(model, data, name), -0.0005)
                self.assertIn("pick_floor", self.contacts_for(model, data, name))
                address = int(model.joint(f"{name}_free").dofadr[0])
                self.assertLess(np.linalg.norm(data.qvel[address:address + 6]), 0.001)

    def test_open_basket_catches_a_falling_cube(self) -> None:
        model, data = self.make_scene(support=True)
        self.drop_cube(model, data)
        position = data.xpos[model.body("pick_cube_1").id]
        np.testing.assert_allclose(position, (0.52, 0.0, 0.03), atol=0.0005)
        self.assertGreaterEqual(self.cube_bottom(model, data, "pick_cube_1"), 0.0095)
        self.assertIn("pick_basket_bottom", self.contacts_for(model, data, "pick_cube_1"))
        self.assertNotIn("pick_floor", self.contacts_for(model, data, "pick_cube_1"))

    def test_each_basket_wall_retains_a_cube(self) -> None:
        directions = {
            "front": (1.0, 0.0),
            "back": (-1.0, 0.0),
            "left": (0.0, 1.0),
            "right": (0.0, -1.0),
        }
        for wall, direction in directions.items():
            with self.subTest(wall=wall):
                model, data = self.make_scene(support=True)
                self.drop_cube(model, data)
                cube_id = model.body("pick_cube_1").id
                data.xfrc_applied[cube_id, :2] = np.asarray(direction) * 0.6
                contacts = set()
                for _ in range(math.ceil(1.4 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                    contacts.update(self.contacts_for(model, data, "pick_cube_1"))
                self.assertTrue(np.isfinite(data.qpos).all())
                self.assertIn(f"pick_basket_{wall}", contacts)
                relative = data.xpos[cube_id] - data.xpos[model.body("pick_basket").id]
                self.assertLess(abs(float(relative[0])), 0.121)
                self.assertLess(abs(float(relative[1])), 0.111)
                self.assertGreater(float(relative[2]), 0.025)
                self.assertLess(float(relative[2]), 0.10)

    def test_cli_spawn_transform_preserves_layout_and_source_robot_files(self) -> None:
        sources = (ASSET_ROOT / "mini3.xml", ASSET_ROOT / "mini3_gripper.xml")
        before = {path: path.read_bytes() for path in sources}
        expected = {
            "pick_cube_1": (0.28, 0.12, 0.0205),
            "pick_cube_2": (0.33, 0.0, 0.0205),
            "pick_cube_3": (0.28, -0.12, 0.0205),
            "pick_basket": (0.52, 0.0, 0.0),
        }
        with tempfile.TemporaryDirectory() as temporary:
            for x, y, yaw in ((1.2, -0.7, 90.0), (-0.5, 0.8, -135.0)):
                with self.subTest(x=x, y=y, yaw=yaw):
                    output = Path(temporary) / "translated_scene.xml"
                    subprocess.run(
                        [sys.executable, str(GENERATOR), "--output", str(output),
                         "--x", str(x), "--y", str(y), "--yaw-deg", str(yaw)],
                        check=True, capture_output=True, text=True,
                    )
                    model, data = self.make_scene(output)
                    base_id = model.body("base_link").id
                    rotation = data.xmat[base_id].reshape(3, 3)
                    np.testing.assert_allclose(data.xpos[base_id, :2], (x, y))
                    for name, local_position in expected.items():
                        world_position = data.xpos[model.body(name).id]
                        relative = rotation.T @ (world_position - (x, y, 0.0))
                        np.testing.assert_allclose(relative, local_position, atol=1e-8)
                        self.assertGreater(float(relative[0]), 0.0)
                    for contact in data.contact:
                        self.assertGreaterEqual(float(contact.dist), -1e-6)
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original, f"Generator changed source {path}")

    def test_start_distance_moves_robot_without_moving_stations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "distant_ground_scene.xml"
            subprocess.run(
                [sys.executable, str(GENERATOR), "--output", str(output),
                 "--x", "1.2", "--y", "-0.7", "--yaw-deg", "90",
                 "--start-distance", "3", "--basket-y", "1"],
                check=True, capture_output=True, text=True,
            )
            model, data = self.make_scene(output)
            base_id = model.body("base_link").id
            rotation = data.xmat[base_id].reshape(3, 3)
            np.testing.assert_allclose(data.xpos[base_id, :2], (1.2, -3.7), atol=1e-8)
            np.testing.assert_allclose(data.mocap_pos[0], data.xpos[base_id], atol=1e-8)
            self.assertFalse(data.eq_active[model.equality("base_support_weld").id])
            expected = {
                "pick_cube_1": (0.28, 0.12, 0.0205),
                "pick_cube_2": (0.33, 0.0, 0.0205),
                "pick_cube_3": (0.28, -0.12, 0.0205),
                "pick_basket": (0.52, 1.0, 0.0),
            }
            for name, position in expected.items():
                relative = rotation.T @ (data.xpos[model.body(name).id] - (1.2, -0.7, 0.0))
                np.testing.assert_allclose(relative, position, atol=1e-8)
            self.assertTrue({"head_rgb", "left_gripper_rgb", "right_gripper_rgb"}.issubset(
                {model.camera(index).name for index in range(model.ncam)},
            ))

    def test_separated_tables_support_free_cubes_and_basket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "table_carry_scene.xml"
            subprocess.run(
                [sys.executable, str(GENERATOR), "--output", str(output),
                 "--start-distance", "2.72", "--surface", "table", "--table-height", "0.45",
                 "--cube-layout", "longitudinal", "--basket-x", "1.3", "--basket-y", "-0.3"],
                check=True, capture_output=True, text=True,
            )
            model, data = self.make_scene(output, support=True)
            for table_name in ("pick_table", "basket_table"):
                table = model.body(table_name)
                self.assertEqual(int(table.jntnum[0]), 0)
                top = model.geom(f"{table_name}_top")
                self.assertNotEqual(int(top.contype[0]), 0)
                self.assertNotEqual(int(top.conaffinity[0]), 0)
                self.assertAlmostEqual(float(data.geom_xpos[top.id, 2] + top.size[2]), 0.45)
            np.testing.assert_allclose(data.xpos[model.body("base_link").id, :2], (-2.72, 0))
            np.testing.assert_allclose(data.xpos[model.body("pick_cube_3").id], (0.28, -0.24, 0.4705))
            top = model.geom("pick_table_top")
            self.assertAlmostEqual(float(data.geom_xpos[top.id, 1] + top.size[1]), -0.19)
            for contact in data.contact:
                self.assertGreaterEqual(float(contact.dist), -1e-6)
            for name in CUBE_NAMES:
                self.assertEqual(int(model.joint(f"{name}_free").type[0]), mujoco.mjtJoint.mjJNT_FREE)
                self.assertAlmostEqual(float(data.xpos[model.body(name).id, 2]), 0.4705)
            self.step(model, data, 1.2)
            for name in CUBE_NAMES:
                self.assertAlmostEqual(float(data.xpos[model.body(name).id, 2]), 0.47, delta=0.0005)
                self.assertIn("pick_table_top", self.contacts_for(model, data, name))
                self.assertNotIn("pick_floor", self.contacts_for(model, data, name))
            self.drop_cube(model, data)
            np.testing.assert_allclose(data.xpos[model.body("pick_cube_1").id],
                                       (1.3, -0.3, 0.48), atol=0.0005)
            self.assertIn("pick_basket_bottom", self.contacts_for(model, data, "pick_cube_1"))

    def test_close_table_stations_share_one_physical_top(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "shared_table_scene.xml"
            subprocess.run(
                [sys.executable, str(GENERATOR), "--output", str(output), "--surface", "table"],
                check=True, capture_output=True, text=True,
            )
            model, data = self.make_scene(output, support=True)
            self.assertEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "basket_table"), -1)
            top = model.geom("pick_table_top")
            basket_id = model.body("pick_basket").id
            relative = data.xpos[basket_id] - data.geom_xpos[top.id]
            self.assertLessEqual(abs(relative[0]) + 0.13, top.size[0])
            self.assertLessEqual(abs(relative[1]) + 0.12, top.size[1])
            self.step(model, data, 1.2)
            for name in CUBE_NAMES:
                self.assertIn("pick_table_top", self.contacts_for(model, data, name))


if __name__ == "__main__":
    unittest.main()
