import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
CROPS = {
    "maize": ("gpp",),
    "rice": ("ndvi",),
    "soybean": ("ndvi",),
    "wheat": ("ndvi", "gpp"),
}
CUTOFFS = (10, 30, 50, 70)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


class SampleDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((SAMPLE / "manifest.json").read_text())

    def test_manifest_hashes(self):
        self.assertEqual(len(self.manifest["cases"]), 16)
        for name, expected in self.manifest["files"].items():
            path = SAMPLE / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(digest(path), expected, name)

    def test_input_reference_separation_and_shapes(self):
        identities = {}
        for crop, products in CROPS.items():
            for cutoff in CUTOFFS:
                stem = SAMPLE / "assets" / crop / f"cutoff_{cutoff:03d}"
                inputs = load(Path(f"{stem}_inputs.npz"))
                reference = load(Path(f"{stem}_reference.npz"))

                self.assertNotIn("target", inputs)
                self.assertIn("target", reference)
                self.assertEqual(inputs["history_x"].shape, (24, 20))
                self.assertEqual(inputs["common"].shape, (24, 465))
                self.assertEqual(inputs["active"].shape, (24, 12))
                self.assertEqual(inputs["tail"].shape, (24, 12))
                self.assertEqual(sorted(np.unique(reference["year"]).tolist()), [2006, 2007, 2008])
                self.assertEqual(reference["features"].shape, (24, 465 + 36 * len(products)))

                identity = np.column_stack(
                    (reference["year"], reference["row"], reference["col"])
                )
                if crop in identities:
                    np.testing.assert_array_equal(identity, identities[crop])
                else:
                    identities[crop] = identity

                for product in products:
                    self.assertEqual(inputs[f"{product}_weather"].shape, (24, 12, 13))
                    self.assertEqual(inputs[f"{product}_context"].shape, (24, 5))
                    prefix = inputs[f"{product}_prefix"]
                    self.assertFalse(np.isfinite(prefix[inputs["tail"]]).any())
                    self.assertFalse(np.isfinite(prefix[~inputs["active"]]).any())

    def test_fixed_sampling_statement(self):
        self.assertEqual(self.manifest["years"], [2006, 2007, 2008])
        self.assertEqual(self.manifest["samples_per_crop_year"], 8)
        self.assertEqual(self.manifest["selection_uses_targets"], False)
        self.assertEqual(self.manifest["selection_uses_predictions"], False)


if __name__ == "__main__":
    unittest.main()
