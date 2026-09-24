import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metrics import annual_rmse, mean_annual_rmse, relative_reduction


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "crop": ["maize"] * 4,
            "year": [2001, 2001, 2002, 2002],
            "target": [1.0, 3.0, 2.0, 4.0],
            "prediction": [2.0, 2.0, 2.0, 6.0],
        })

    def test_annual_rmse(self):
        result = annual_rmse(self.frame)
        np.testing.assert_allclose(result.rmse, [1.0, np.sqrt(2.0)])

    def test_equal_year_weight(self):
        value = mean_annual_rmse(self.frame).mean_annual_rmse.iloc[0]
        self.assertAlmostEqual(value, (1.0 + np.sqrt(2.0)) / 2)

    def test_relative_reduction(self):
        candidate = self.frame.copy()
        candidate["prediction"] = candidate.target
        result = relative_reduction(candidate, self.frame)
        self.assertAlmostEqual(result.rmse_reduction_percent.iloc[0], 100.0)

    def test_reject_nonfinite(self):
        invalid = self.frame.copy()
        invalid.loc[0, "prediction"] = np.nan
        with self.assertRaises(ValueError):
            annual_rmse(invalid)

    def test_methods_are_scored_separately(self):
        frame = pd.concat([
            self.frame.assign(method="a"),
            self.frame.assign(method="b", prediction=self.frame.target),
        ], ignore_index=True)
        result = mean_annual_rmse(frame)
        self.assertEqual(result.method.tolist(), ["a", "b"])
        self.assertGreater(result.mean_annual_rmse.iloc[0], 0)
        self.assertEqual(result.mean_annual_rmse.iloc[1], 0)


if __name__ == "__main__":
    unittest.main()
