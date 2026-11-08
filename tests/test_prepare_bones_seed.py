"""CPU checks for complete, deterministic source inventory and mirror grouping."""

import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("prepare_bones_seed", Path(__file__).parents[1] / "tools/prepare_bones_seed.py")
assert spec is not None and spec.loader is not None
bones = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bones)


def row(name="walk__A001", mirror=False):
    return {
        "filename": name, "move_name": "independent display name", "is_mirror": mirror,
        "actor_uid": "A001", "take_name": "walk", "take_date": 240101,
        "package": "Locomotion", "category": "Baseline", "move_duration_frames": 121,
        "move_soma_uniform_path": f"soma_uniform/bvh/240101/{name}.bvh",
        "move_soma_proportional_path": f"soma_proportional/bvh/240101/{name}.bvh",
        "move_g1_path": f"g1/csv/240101/{name}.csv",
        "move_soma_uniform_shape_path": "soma_shapes/base.npz",
        "move_soma_proportional_shape_path": "soma_shapes/A001.npz",
        "content_natural_desc_1": "A person walks.",
    }


class InventoryTests(unittest.TestCase):
    def test_preserves_every_row_and_groups_mirrors_with_different_frame_counts(self):
        original, mirror = row(), row("walk__A001_M", True)
        mirror["move_duration_frames"] = 120  # Actual v004 has 7,653 differing pairs.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = bones.inventory([mirror, original], output)
            with gzip.open(output / "motions.jsonl.gz", "rt") as stream:
                entries = [json.loads(line) for line in stream]
            self.assertEqual([entry["metadata"] for entry in entries], [original, mirror])
            self.assertEqual({entry["source_group"] for entry in entries}, {"walk__A001"})
            self.assertEqual(summary["metadata_frames"], 241)
            self.assertEqual(summary["source_groups"], 1)
            self.assertEqual(summary["training_admitted_motions"], 0)
            self.assertEqual(summary["local_presence"]["status"], "not_inspected")
            first = (output / "motions.jsonl.gz").read_bytes()
            bones.inventory([original, mirror], output)
            self.assertEqual((output / "motions.jsonl.gz").read_bytes(), first)

    def test_missing_mirror_parent_is_retained_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = bones.inventory([row("walk__A001_M", True)], Path(directory))
            self.assertEqual(summary["motions"], 1)
            self.assertEqual(summary["unresolved_mirror_filenames"], ["walk__A001_M"])
            self.assertEqual(summary["source_groups"], 0)

    def test_duplicate_identity_and_unsafe_path_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate filename"):
                bones.inventory([row(), row()], Path(directory))
            for path in ("/etc/passwd", "../escape.bvh", "soma/../../escape", "soma\\file.bvh"):
                bad = row()
                bad["move_soma_uniform_path"] = path
                with self.assertRaises(ValueError):
                    bones.inventory([bad], Path(directory))

    def test_missing_files_counted_without_claiming_verified_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            source = row()
            present = root / source["move_soma_uniform_path"]
            present.parent.mkdir(parents=True)
            present.write_text("synthetic test placeholder")
            summary = bones.inventory([source], Path(directory) / "inventory", root)
            presence = summary["local_presence"]
            self.assertEqual(presence["status"], "checked_presence_only")
            self.assertEqual(presence["missing_motion_references"]["soma_uniform"], 0)
            self.assertEqual(presence["missing_motion_references"]["g1"], 1)
            self.assertEqual(source, row())


if __name__ == "__main__":
    unittest.main()
