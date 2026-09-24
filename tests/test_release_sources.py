import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SourceTests(unittest.TestCase):
    def test_workflow_hashes_and_syntax(self):
        manifest = json.loads((ROOT / 'workflow/source_manifest.json').read_text())
        for name, entry in manifest.items():
            file = ROOT / 'workflow' / name
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), entry['sha256'], name)
            ast.parse(file.read_text())

    def test_reconstruction_hashes(self):
        manifest = json.loads((ROOT / 'reconstruction/source_manifest.json').read_text())
        for name, entry in manifest.items():
            file = ROOT / 'reconstruction/scripts' / name
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), entry['sha256'], name)

    def test_annual_record_hashes(self):
        folder = ROOT / 'data/reference/annual'
        for name, entry in json.loads((folder/'provenance.json').read_text()).items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(), entry['sha256'])

    def test_feature_imports_without_models(self):
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run([sys.executable, str(ROOT/'scripts/build_features.py'),
                            '--workspace', folder, '--stage', 'check-imports'], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def test_raw_stage_plan_does_not_write(self):
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run([sys.executable, str(ROOT/'scripts/reconstruct_raw.py'),
                            '--workspace', folder, '--stage', 'all', '--dry-run'], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(list(Path(folder).iterdir()), [])
