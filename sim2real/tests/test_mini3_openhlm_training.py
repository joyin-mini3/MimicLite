"""Mini3 VLA transforms must retain this robot's channels and camera meaning."""

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mini3_openhlm_training import CAMERA_KEYS, CHANNELS, Mini3Inputs, Mini3Outputs
from mini3_vla_features import FEATURE_NAMES


class Mini3OpenHLMTransformTest(unittest.TestCase):
    def sample(self):
        frame = {key: np.full((224, 224, 3), 64, np.uint8) for key in CAMERA_KEYS}
        frame.update(state=np.arange(CHANNELS, dtype=np.float32),
                     actions=np.tile(np.arange(CHANNELS, dtype=np.float32), (50, 1)),
                     prompt="Please put the blue square from the tabletop into the basket.")
        return frame

    def test_state_action_names_preserve_mini3_root_channels(self):
        self.assertEqual(CHANNELS, 36)
        self.assertEqual(FEATURE_NAMES[-1], "root_height")
        original = self.sample()
        transformed = Mini3Inputs()(original)
        np.testing.assert_array_equal(transformed["state"], original["state"])
        np.testing.assert_array_equal(transformed["actions"], original["actions"])
        self.assertEqual(transformed["prompt"], original["prompt"])

    def test_float_chw_images_recover_rgb_uint8(self):
        original = self.sample()
        for key in CAMERA_KEYS:
            original[key] = original[key].transpose(2, 0, 1).astype(np.float32) / 255
        transformed = Mini3Inputs()(original)
        for image in transformed["image"].values():
            self.assertEqual(image.shape, (224, 224, 3))
            self.assertEqual(image.dtype, np.uint8)
            self.assertTrue(np.all(image == 64))

    def test_inference_observation_keys_and_camera_masks(self):
        original = self.sample()
        data = {f"observation/{key}": value for key, value in original.items()
                if key in (*CAMERA_KEYS, "state")}
        data["prompt"] = original["prompt"]
        data["observation/head_image_left"][:] = 0
        transformed = Mini3Inputs()(data)
        self.assertNotIn("actions", transformed)
        self.assertTrue(all(transformed["image_mask"].values()))
        self.assertTrue(np.all(transformed["image"]["base_0_rgb"] == 0))

    def test_reject_g1_action_width(self):
        original = self.sample()
        original["actions"] = original["actions"][:, :34]
        with self.assertRaisesRegex(ValueError, "36"):
            Mini3Inputs()(original)

    def test_reject_nonfinite_state(self):
        original = self.sample()
        original["state"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            Mini3Inputs()(original)

    def test_outputs_keep_root_velocity_and_height(self):
        actions = np.tile(np.arange(40, dtype=np.float32), (50, 1))
        result = Mini3Outputs()({"actions": actions})["actions"]
        self.assertEqual(result.shape, (50, 36))
        np.testing.assert_array_equal(result, actions[:, :36])
        self.assertEqual(result[0, -1], 35)

    def test_outputs_reject_g1_width(self):
        with self.assertRaisesRegex(ValueError, "36"):
            Mini3Outputs()({"actions": np.zeros((50, 34))})


if __name__ == "__main__":
    unittest.main()
