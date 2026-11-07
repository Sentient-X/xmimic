# Motion input to the tracker

The tracker consumes motion already retargeted to its robot. It does not infer a robot from
the joint count, or convert a human skeleton by dropping coordinates.

`motion_format=lafan1_csv` retains the original G1 baseline: headerless rows of root position
in metres, root quaternion in **xyzw**, and exactly 29 joint angles in the baseline G1 order,
at 30 fps. The loader rejects any width other than 36 columns.

`motion_format=retargeted_npz` reads one numeric `.npz` file per clip. Every file requires:

| Array | Shape / type | Meaning |
|---|---|---|
| `root_p` | `(T, 3)`, real numeric | Robot root position, metres, Z-up world |
| `root_quat` | `(T, 4)`, real numeric | Unit **wxyz** quaternion, root orientation in world |
| `q` | `(T, J)`, real numeric | Robot joint angles in radians |
| `fps` | scalar, real numeric | Positive source sample rate; timestamps are `i / fps` |
| `joint_names` | `(J,)`, NumPy Unicode | Native robot joint names, one per `q` column |
| `embodiment_id` | scalar, NumPy Unicode | Exact canonical embodiment revision of the retarget |

The caller supplies the target's joint names and embodiment revision. The loader checks the
exact name set and reorders columns into the target's order. Mismatched revisions, missing or
duplicate joints, object arrays, nonfinite values, non-unit quaternions, invalid shapes, and
clips too short to produce two samples at 50 Hz are refused before device placement.
Quaternion norm error up to 0.001 is treated as serialization rounding and normalized.
All numeric arrays must fit finite float32. Positions and angles interpolate linearly;
quaternions use shortest-arc SLERP. Velocities are derived after resampling at 50 Hz.
The retargeter must supply continuous joint-angle trajectories rather than discontinuous
wrapped angles. Admission does not establish collision, contact, torque, or balance validity.

Example writer after retargeting, using names and identity from the selected embodiment:

```python
np.savez(
    output_path,
    root_p=root_positions_m,
    root_quat=root_quaternions_wxyz,
    q=joint_angles_rad,
    fps=np.asarray(source_fps),
    joint_names=np.asarray(body.joint_names),
    embodiment_id=np.asarray(str(body.id)),
)
```

Use a separate directory for each representation and admitted split. The selected directory's
top-level clips are read in sorted filename order. Mixed CSV/NPZ directories are refused.
Source dataset identity, license, original/mirror grouping, annotations, retargeting provenance,
split membership and rejection accounting remain the governed corpus's responsibility; this
NPZ is the trainer's numeric boundary, not a second catalog or a source archive format.

BONES-SEED's source G1 CSVs have named Euler-angle columns and a different timebase and units
from this LAFAN1 representation. They cannot be placed in the LAFAN1 directory. For another
embodiment, retarget SOMA source motion against that body's actual model before producing these
arrays. A format parser alone does not qualify the retarget.

The current `Motion` library still concatenates **all selected frames on the device**, including
forward-kinematic poses and velocities. Per-clip contact-floor estimation now uses contiguous
clip slices rather than repeatedly scanning all frame IDs, but that does not solve full-corpus
device memory. Bounded-memory loading, sampling and evaluation are required before claiming the
entire BONES-SEED corpus can train in one run.

CPU validation from the repository root:

```bash
JAX_PLATFORMS=cpu python -m pytest tests/test_motion_inputs.py -q
```
