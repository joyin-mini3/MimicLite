"""Normalization must use the same future-action boundary contract as training."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validate_mini3_openhlm_dataset import action_windows


class DatasetWindowTest(unittest.TestCase):
    def test_future_windows_repeat_the_terminal_action_without_crossing_episode(self):
        actions = np.arange(4 * 36, dtype=np.float32).reshape(4, 36)
        before = actions.copy()
        windows = action_windows(actions)
        self.assertEqual(windows.shape, (4, 50, 36))
        for frame in range(4):
            expected = actions[np.minimum(np.arange(frame, frame + 50), 3)]
            np.testing.assert_array_equal(windows[frame], expected)
        np.testing.assert_array_equal(actions, before)

    def test_single_action_episode_repeats_only_its_own_action(self):
        first_episode = np.full((1, 36), 1., dtype=np.float32)
        second_episode = np.full((1, 36), -2., dtype=np.float32)
        self.assertTrue(np.all(action_windows(first_episode) == 1.))
        self.assertTrue(np.all(action_windows(second_episode) == -2.))


if __name__ == "__main__":
    unittest.main()
