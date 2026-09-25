import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("download_processed", Path(__file__).resolve().parents[1] / "scripts/download_processed_data.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProcessedDownloadTests(unittest.TestCase):
    def make_archive(self, root, name="Data/example.npy", value=b"test"):
        archive = root / "shard.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo(name)
            info.size = len(value)
            tar.addfile(info, io.BytesIO(value))
        return archive, [{"path": name, "bytes": len(value), "sha256": MODULE.hashlib.sha256(value).hexdigest()}]

    def test_extract_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, records = self.make_archive(root)
            for _ in range(2):
                MODULE.extract_verified(archive, root / "workspace", records)
            self.assertEqual((root / "workspace/Data/example.npy").read_bytes(), b"test")

    def test_different_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, records = self.make_archive(root)
            target = root / "workspace/Data/example.npy"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"user data")
            with self.assertRaises(FileExistsError):
                MODULE.extract_verified(archive, root / "workspace", records)
            self.assertEqual(target.read_bytes(), b"user data")

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, records = self.make_archive(root, "../outside")
            with self.assertRaises(ValueError):
                MODULE.extract_verified(archive, root / "workspace", records)
            self.assertFalse((root / "outside").exists())

    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, records = self.make_archive(root)
            records.append({"path": "Data/missing.npy", "bytes": 0, "sha256": ""})
            with self.assertRaises(ValueError):
                MODULE.extract_verified(archive, root / "workspace", records)

    def test_bad_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, records = self.make_archive(root)
            records[0]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                MODULE.extract_verified(archive, root / "workspace", records)
            self.assertFalse((root / "workspace/Data/example.npy").exists())

    def test_symlink_destination_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, records = self.make_archive(root)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (workspace / "Data").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                MODULE.extract_verified(archive, workspace, records)
            self.assertFalse((outside / "example.npy").exists())

    def test_unlisted_member_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive, _ = self.make_archive(root)
            with self.assertRaises(ValueError):
                MODULE.extract_verified(archive, root / "workspace", [])


if __name__ == "__main__":
    unittest.main()
