import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reconstruction/scripts'))
from ndvi_tail_replacement import tail_mask
from observed_remote_benchmark import trajectory_features


class PreprocessingTests(unittest.TestCase):
    def test_suffix_rounding_and_padding(self):
        active = np.array([[1, 1, 0, 1, 1, 1, 0], [0]*7], bool)
        for ratio, expected in [(0, 0), (.1, 1), (.3, 2), (.5, 3), (.7, 4), (1, 5)]:
            mask = tail_mask(active, ratio)
            self.assertEqual(mask[0].sum(), expected)
            self.assertFalse(mask[1].any())
            self.assertFalse((mask & ~active).any())
            if expected:
                self.assertTrue(mask[0, 5])

    def test_bad_suffix_fraction(self):
        with self.assertRaises(ValueError):
            tail_mask(np.ones((2, 12)), 1.1)

    def test_trajectory_preserves_slots_and_summaries(self):
        values = np.arange(12, dtype=float)[None, :]
        valid = np.zeros((1, 12), bool)
        valid[0, :3] = True
        out = trajectory_features(values, valid)
        self.assertEqual(out.shape, (1, 18))
        np.testing.assert_array_equal(out[0, :12], [0, 1, 2, *([0]*9)])
        np.testing.assert_allclose(out[0, 12:], [1, np.std([0,1,2]), 2, 0, 3, 2/11])

    def test_empty_trajectory(self):
        result = trajectory_features(np.full((2, 12), np.nan), np.zeros((2, 12), bool))
        self.assertTrue(np.isfinite(result).all())
        self.assertFalse(result.any())
