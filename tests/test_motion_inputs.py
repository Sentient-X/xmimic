"""Motion admission and timebase tests; no GPU or external dataset required."""

import numpy as np
import pytest

from xmimic.kin import Kinematics
from xmimic.motion import load_lafan1_csv, load_motion, load_retargeted_npz


IDENTITY = "example-embodiment-revision"
NAMES = ("left_knee", "right_knee")


def write_clip(path, **changes):
    frames = 13
    t = np.arange(frames) / 120
    data = {
        "root_p": np.column_stack((t, np.zeros(frames), np.ones(frames))),
        "root_quat": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        "q": np.column_stack((t + 2, t + 1)),
        "fps": np.asarray(120.0),
        "joint_names": np.asarray(NAMES[::-1]),
        "embodiment_id": np.asarray(IDENTITY),
    }
    data.update(changes)
    np.savez(path, **data)


def kinematics():
    return Kinematics({
        "kin_parent": [-1, 0, 0],
        "kin_joint_idx": [-1, 0, 1],
        "kin_axes": [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
        "kin_off_p": [[0, 0, 0], [0, 0.2, -1], [0, -0.2, -1]],
        "kin_off_R": np.tile(np.eye(3), (3, 1, 1)).tolist(),
        "kin_tracked_idx": [1, 2],
        "q_lo": [-3, -3],
        "q_hi": [3, 3],
    })


def test_named_clip_reorders_joints_and_uses_its_timebase(tmp_path):
    path = tmp_path / "walk.npz"
    write_clip(path)
    p, quat, q, fps = load_retargeted_npz(path, NAMES, IDENTITY)
    assert fps == 120
    np.testing.assert_allclose(q[0], [1, 2])
    np.testing.assert_allclose(quat[0], [1, 0, 0, 0])
    motion = load_motion(tmp_path, kinematics(), [0, 1], motion_format="retargeted_npz",
                         joint_names=NAMES, embodiment_id=IDENTITY)
    assert motion.num_frames == 6
    assert motion.num_joints == 2
    np.testing.assert_allclose(motion.root_p[:, 0], np.arange(6) * 0.02, atol=1e-7)
    np.testing.assert_allclose(motion.root_v[:, 0], 1, atol=1e-5)
    np.testing.assert_allclose(motion.q[:, 0], 1 + np.arange(6) * 0.02, atol=1e-7)


@pytest.mark.parametrize(("change", "message"), [
    ({"embodiment_id": np.asarray("another-revision")}, "embodiment_id"),
    ({"joint_names": np.asarray(["left_knee", "left_knee"])}, "exactly once"),
    ({"joint_names": np.asarray(["left_knee", "wrong_joint"])}, "exactly once"),
    ({"joint_names": np.asarray([["left_knee", "right_knee"]])}, "string vector"),
    ({"fps": np.asarray([120.0])}, "positive scalar"),
    ({"fps": np.asarray(0.0)}, "positive scalar"),
    ({"fps": np.asarray(np.nan)}, "finite real"),
    ({"fps": np.asarray(True)}, "finite real"),
    ({"root_p": np.zeros((1, 3))}, "T >= 2"),
    ({"root_quat": np.zeros((13, 4))}, "unit quaternions"),
    ({"root_quat": np.tile([2, 0, 0, 0], (13, 1))}, "unit quaternions"),
    ({"q": np.zeros((13, 3))}, "shapes"),
    ({"q": np.full((13, 2), np.inf)}, "finite real"),
    ({"q": np.full((13, 2), 1e100)}, "overflow float32"),
    ({"fps": np.asarray(12000.0)}, "two frames"),
])
def test_invalid_motion_is_refused_before_resampling(tmp_path, change, message):
    path = tmp_path / "invalid.npz"
    write_clip(path, **change)
    with pytest.raises(ValueError, match=message):
        load_retargeted_npz(path, NAMES, IDENTITY)


def test_pickle_arrays_are_refused(tmp_path):
    path = tmp_path / "object.npz"
    write_clip(path, q=np.zeros((13, 2), dtype=object))
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_retargeted_npz(path, NAMES, IDENTITY)


def test_retargeted_input_requires_explicit_target_and_format(tmp_path):
    write_clip(tmp_path / "walk.npz")
    with pytest.raises(ValueError, match="requires joint_names and embodiment_id"):
        load_motion(tmp_path, kinematics(), [0, 1], motion_format="retargeted_npz")
    with pytest.raises(ValueError, match="29-joint G1 baseline"):
        load_motion(tmp_path, kinematics(), [0, 1])
    (tmp_path / "source.csv").write_text("Frame,root_translateX\n")
    with pytest.raises(ValueError, match="mixed motion formats"):
        load_motion(tmp_path, kinematics(), [0, 1], motion_format="retargeted_npz",
                    joint_names=NAMES, embodiment_id=IDENTITY)


def test_contacts_and_future_goals_do_not_cross_clip_boundaries(tmp_path):
    for name, height in (("a", 1.0), ("b", 11.0)):
        write_clip(tmp_path / f"{name}.npz",
                   root_p=np.tile([0.0, 0.0, height], (13, 1)), q=np.zeros((13, 2)))
    motion = load_motion(tmp_path, kinematics(), [0, 1], motion_format="retargeted_npz",
                         joint_names=NAMES, embodiment_id=IDENTITY)
    assert motion.names == ("a", "b")
    np.testing.assert_array_equal(motion.clip_start, [0, 6])
    np.testing.assert_array_equal(motion.clip_end, [5, 11])
    np.testing.assert_array_equal(motion.ref_contact, np.ones((12, 2)))
    np.testing.assert_array_equal(motion.future_frames(motion.clip_end, 3), [[5, 5, 5], [11, 11, 11]])


def test_lafan_baseline_has_exact_width_and_xyzw_conversion(tmp_path):
    path = tmp_path / "lafan.csv"
    raw = np.zeros((3, 36))
    raw[:, 6] = 1
    raw[:, 7:] = np.arange(29)
    np.savetxt(path, raw, delimiter=",")
    _, quat, q, fps = load_lafan1_csv(path)
    assert fps == 30
    np.testing.assert_array_equal(quat, np.tile([1, 0, 0, 0], (3, 1)))
    np.testing.assert_array_equal(q, raw[:, 7:])
    np.savetxt(path, np.column_stack((raw, np.zeros(3))), delimiter=",")
    with pytest.raises(ValueError, match="exactly 36 columns"):
        load_lafan1_csv(path)


def test_bones_header_is_not_guessed_as_lafan(tmp_path):
    path = tmp_path / "bones.csv"
    path.write_text("Frame,root_translateX,root_translateY,root_translateZ\n0,0,0,100\n")
    with pytest.raises(ValueError):
        load_lafan1_csv(path)
