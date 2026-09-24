import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from splits import partition


class SplitTests(unittest.TestCase):
    def test_three_blocks(self):
        years = np.arange(1982, 2017)
        evaluation = []
        for cutoff in (2001, 2005, 2009):
            masks = partition(years, cutoff)
            self.assertFalse((masks["inner_train"] & masks["inner_validation"]).any())
            self.assertFalse((masks["final_train"] & masks["evaluation"]).any())
            np.testing.assert_array_equal(years[masks["inner_validation"]], [cutoff-1, cutoff])
            evaluation.extend(years[masks["evaluation"]])
        self.assertEqual(evaluation, [2002, 2003, 2004, 2006, 2007, 2008, *range(2010, 2017)])

    def test_reject_unknown_block(self):
        with self.assertRaises(ValueError):
            partition([2013], 2012)

    def test_reject_fractional_year(self):
        with self.assertRaises(ValueError):
            partition([2001.5], 2001)
