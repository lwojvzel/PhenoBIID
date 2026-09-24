import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from rebuild_main_table import rebuild, verify


class ReferenceTests(unittest.TestCase):
    def test_main_table_reconstruction(self):
        self.assertEqual(len(verify(rebuild())), 48)

    def test_reject_modified_summary(self):
        table = rebuild()
        table.loc[0, "rmse"] += 0.01
        with self.assertRaises(ValueError):
            verify(table)
