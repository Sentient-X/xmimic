"""Geometric and malformed-input oracles for CPU SOMA ingestion; no licensed data."""

import dataclasses
import hashlib
import unittest

import numpy as np

from xmimic.retarget_source import ParseLimits, SourceError, parse_soma_bvh


HEADER = """HIERARCHY
ROOT Root
{
 OFFSET 1000 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
 JOINT Hips
 {
  OFFSET 0 300 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Hand
  {
   OFFSET 100 0 0
   CHANNELS 3 Yrotation Zrotation Xrotation
   End Site { OFFSET 10 0 0 }
  }
 }
}
MOTION
"""
ROWS = "100 0 0 90 0 0 0 100 0 0 90 0 90 90 0\n100 0 0 0 0 0 0 200 0 0 0 0 0 0 0\n"


def fixture(header=HEADER, rows=ROWS, count=2, dt="0.008333"):
    return (header + f"Frames: {count}\nFrame Time: {dt}\n" + rows).encode()


class SourceTests(unittest.TestCase):
    def parse(self, data=None, **kwargs):
        return parse_soma_bvh(fixture() if data is None else data,
                              world_from_source=kwargs.pop("world_from_source", np.eye(4)), **kwargs)

    def test_moving_virtual_root_and_below_root_translation_have_geometric_oracles(self):
        clip = self.parse()
        self.assertEqual(clip.joint_names, ("Root", "Hips", "Hand"))
        np.testing.assert_array_equal(clip.parents, [-1, 0, 1])
        # Root OFFSET=10 m and Hips OFFSET=3 m are replaced by channels.
        # Root's 90-degree Z rotation moves Hips' local +Y onto world -X.
        np.testing.assert_allclose(clip.world_positions[0], [[1, 0, 0], [0, 0, 0], [0, 0, -1]], atol=1e-14)
        np.testing.assert_allclose(clip.world_positions[1], [[1, 0, 0], [1, 2, 0], [2, 2, 0]], atol=1e-14)
        rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        np.testing.assert_allclose(clip.world_rotations[0, 0], rz, atol=1e-14)
        np.testing.assert_allclose(clip.world_rotations[0, 1], rz @ ry, atol=1e-14)
        # Hand declares YZX, not Root's ZYX: these rotations do not commute.
        np.testing.assert_allclose(clip.world_rotations[0, 2], rz @ ry @ ry @ rz, atol=1e-14)
        np.testing.assert_allclose(clip.world_rotations[1], np.broadcast_to(np.eye(3), (3, 3, 3)), atol=1e-14)

    def test_explicit_world_placement_rotates_positions_and_joint_frames_once(self):
        # Chosen +90 degrees around X maps source +Y to world +Z.
        transform = np.array([[1, 0, 0, 4], [0, 0, -1, 5], [0, 1, 0, 6], [0, 0, 0, 1]], dtype=float)
        clip = self.parse(world_from_source=transform)
        np.testing.assert_allclose(clip.world_positions[1], [[5, 5, 6], [5, 5, 8], [6, 5, 8]], atol=1e-14)
        np.testing.assert_allclose(clip.world_rotations[1, 2], transform[:3, :3], atol=1e-14)
        transform[0, 3] = 99
        self.assertEqual(clip.world_from_source[0, 3], 4)

    def test_timebase_hash_and_array_ownership(self):
        raw = fixture(dt="8.333e-3")
        clip = self.parse(raw)
        self.assertEqual(clip.source_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(clip.frame_time_literal, "8.333e-3")
        self.assertEqual(clip.frame_time_s, 0.008333)
        self.assertAlmostEqual(clip.fps, 120.00480019200768)
        self.assertEqual(clip.frame_count, 2)
        self.assertEqual(clip.joint_index("Hips"), 1)
        self.assertEqual(clip.channels[2], ("Yrotation", "Zrotation", "Xrotation"))
        for array in (clip.parents, clip.offsets_m, clip.world_positions, clip.world_rotations, clip.world_from_source):
            with self.assertRaises(ValueError):
                array.flat[0] = 0
        with self.assertRaises(SourceError):
            clip.joint_index("not-a-joint")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            clip.frame_time_s = 1

    def test_frame_conversion_is_required_and_reflections_scales_nonfinite_refused(self):
        with self.assertRaises(TypeError):
            parse_soma_bvh(fixture())
        for matrix in (np.zeros((3, 3)), np.diag([-1., 1, 1, 1]),
                       np.diag([2., 1, 1, 1]), np.full((4, 4), np.nan)):
            with self.subTest(matrix=matrix), self.assertRaises(SourceError):
                self.parse(world_from_source=matrix)

    def test_channel_axis_lookup_handles_reordered_translation(self):
        header = HEADER.replace("Xposition Yposition Zposition", "Yposition Zposition Xposition")
        # Same channel values now mean a different axis assignment; no assumed xyz columns.
        clip = self.parse(fixture(header=header))
        np.testing.assert_allclose(clip.world_positions[1, 0], [0, 1, 0], atol=1e-14)
        np.testing.assert_allclose(clip.world_positions[1, 1], [0, 1, 2], atol=1e-14)

    def test_bad_counts_channels_hierarchy_and_nonfinite_rows_refused(self):
        malformed = [
            fixture(count=3), fixture(count=1), fixture(rows=ROWS + ROWS),
            fixture(rows=ROWS.replace("100 0 0", "100 0", 1)),
            fixture(rows=ROWS.replace("100", "nan", 1)),
            fixture(rows=ROWS.replace("100", "inf", 1)),
            fixture(header=HEADER.replace("JOINT Hand", "JOINT Hips")),
            fixture(header=HEADER.replace("Yrotation Zrotation Xrotation", "Yrotation Yrotation Xrotation")),
            fixture(header=HEADER.replace("Yrotation Zrotation Xrotation", "Yrotation Scale Xrotation")),
            fixture(header=HEADER.replace("CHANNELS 3", "CHANNELS 4")),
            fixture(header=HEADER.replace("OFFSET 10 0 0", "OFFSET nan 0 0")),
            fixture(header=HEADER.replace("End Site { OFFSET 10 0 0 }", "End Site { OFFSET 10 0 0 } End Site { OFFSET 0 0 0 }")),
            fixture(header=HEADER.replace("}\nMOTION", "MOTION", 1)),
            fixture().replace(b"MOTION", b"NO_MOTION"), b"\xff",
        ]
        for raw in malformed:
            with self.subTest(raw=raw[:40]), self.assertRaises(SourceError):
                self.parse(raw)
        for dt in ("0", "-0.1", "nan", "inf", "1e-999", "1e308"):
            with self.subTest(dt=dt), self.assertRaises(SourceError):
                self.parse(fixture(dt=dt))

    def test_resource_and_numeric_limits_are_explicit_refusals(self):
        for limits in (ParseLimits(max_bytes=10), ParseLimits(max_header_bytes=10),
                       ParseLimits(max_frames=1), ParseLimits(max_joints=2),
                       ParseLimits(max_pose_samples=5), ParseLimits(max_abs_length_m=1),
                       ParseLimits(max_abs_rotation_degrees=45)):
            with self.subTest(limits=limits), self.assertRaises(SourceError):
                self.parse(limits=limits)
        for kwargs in ({"max_frames": True}, {"max_joints": 257}, {"max_bytes": 0},
                       {"max_abs_length_m": np.inf}):
            with self.assertRaises(SourceError):
                ParseLimits(**kwargs)


if __name__ == "__main__":
    unittest.main()
