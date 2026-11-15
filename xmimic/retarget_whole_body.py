"""Whole-body, site-calibrated retargeting from a human skeleton to any articulated body.

Every per-body input is derived from the embodiment document's named semantic sites and the
body's own MuJoCo model; nothing here names a robot, a limb or a joint:

* the site tree is the robot's kinematic tree restricted to the sites (each site's parent is
  the nearest site on its ancestor path);
* a segment is *rigid* when the child site stays fixed in the parent site's frame for random
  joint values, otherwise *articulated*;
* scale: articulated segments take the robot's own length along the human segment direction
  (anatomy before endpoints), rigid segments follow the parent's calibrated orientation;
* orientation offsets come from one declared human calibration frame, after a direction-only
  fit of the robot to that frame;
* joints that are not ancestors of any corresponding site are held at the reference state.

The only source-side declaration is ``correspondence``: semantic site name -> source joint name
(written once per source skeleton, e.g. SOMA). The solve is bounded least squares per frame
(position + orientation targets, joint limits as hard bounds, a velocity prior toward the
previous frame), then a whole-body ground pass (lowest body point follows the scaled human's
lowest point, never below the floor) and a zero-phase low-pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from .retarget_source import Clip


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix (numerically safe near 0 and pi)."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    v = np.array((R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]))
    theta = np.arccos(c)
    if theta < 1e-6:
        return 0.5 * v
    if theta > np.pi - 1e-4:
        return Rotation.from_matrix(R).as_rotvec()
    return theta / (2.0 * np.sin(theta)) * v


@dataclass(frozen=True)
class Site:
    name: str
    body: int
    offset: np.ndarray  # position in the body frame
    rotation: np.ndarray  # site frame in the body frame


@dataclass(frozen=True)
class Segment:
    parent: str
    child: str
    rigid: bool
    length: float  # robot length at the calibration pose
    rigid_offset: np.ndarray  # child position in the parent site's frame (rigid segments)
    bone_offset: np.ndarray | None = None  # child minus parent, in the human bone frame (articulated)


@dataclass
class WholeBodyRetargeter:
    model: mujoco.MjModel
    sites: dict[str, Site]
    correspondence: dict[str, str]  # site -> source joint
    root_site: str
    segments: list[Segment]
    solved_dofs: np.ndarray  # dof indices that are optimized (others held)
    reference_qpos: np.ndarray
    rotation_offsets: dict[str, np.ndarray] = field(default_factory=dict)
    root_scale: float = 1.0
    position_weight: float = 1.0
    root_position_weight: float = 3.0
    orientation_weight: float = 0.2
    root_orientation_weight: float = 1.0
    velocity_weight: float = 0.08
    max_joint_speed: float = 25.0  # rad/s, solve-time continuity bound (not an actuator rating)
    posture_weight: float = 0.01
    calibration_posture_weight: float = 0.2  # keep the fit near the body's declared reference state
    calibration_report: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_document(cls, model: mujoco.MjModel, document: Mapping, correspondence: Mapping[str, str],
                      reference_qpos: np.ndarray | None = None) -> "WholeBodyRetargeter":
        if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("the body model must start with a floating root")
        sites: dict[str, Site] = {}
        for entry in document["sites"]:
            if entry["name"] not in correspondence:
                continue
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entry["link"])
            if body < 0:
                raise ValueError(f"site {entry['name']} names an absent link")
            rot = Rotation.from_euler("xyz", entry.get("rpy", (0.0, 0.0, 0.0))).as_matrix()
            sites[entry["name"]] = Site(entry["name"], body, np.asarray(entry["xyz"], float), rot)
        missing = set(correspondence) - set(sites)
        if missing:
            raise ValueError(f"correspondence names sites absent from the body: {sorted(missing)}")
        q0 = model.qpos0.copy() if reference_qpos is None else np.asarray(reference_qpos, float).copy()

        def ancestors(body: int) -> list[int]:
            out = []
            while body > 0:
                out.append(body)
                body = model.body_parentid[body]
            return out

        # site tree: nearest corresponding site on the ancestor path (same body counts as ancestor
        # only for a site declared earlier at a shallower depth -> break ties by ancestry length)
        depth = {n: len(ancestors(s.body)) for n, s in sites.items()}
        parent: dict[str, str | None] = {}
        for name, site in sites.items():
            best, best_depth = None, -1
            chain = ancestors(site.body)
            for other, o in sites.items():
                if other == name or o.body not in chain:
                    continue
                d = depth[other]
                if o.body == site.body:
                    # same link: the site closer to the link origin is the parent
                    if np.linalg.norm(o.offset) >= np.linalg.norm(site.offset):
                        continue
                if d > best_depth or (d == best_depth and best is not None
                                      and np.linalg.norm(o.offset) > np.linalg.norm(sites[best].offset)):
                    best, best_depth = other, d
            parent[name] = best
        roots = [n for n, p in parent.items() if p is None]
        if len(roots) != 1:
            raise ValueError(f"site tree must have one root, found {roots}")

        # solved dofs: ancestors of any corresponding site (excluding the free root, added below)
        solved = set()
        for site in sites.values():
            for b in ancestors(site.body):
                for j in range(model.body_jntnum[b]):
                    jid = model.body_jntadr[b] + j
                    adr = model.jnt_dofadr[jid]
                    width = 6 if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE else 1
                    solved.update(range(adr, adr + width))
        self = cls(model=model, sites=sites, correspondence=dict(correspondence), root_site=roots[0],
                   segments=[], solved_dofs=np.array(sorted(solved)), reference_qpos=q0)
        self.segments = self._segments(parent)
        return self

    def _site_pose(self, data: mujoco.MjData, name: str) -> tuple[np.ndarray, np.ndarray]:
        s = self.sites[name]
        R = data.xmat[s.body].reshape(3, 3)
        return data.xpos[s.body] + R @ s.offset, R @ s.rotation

    def _segments(self, parent: Mapping[str, str | None]) -> list[Segment]:
        m = self.model
        d = mujoco.MjData(m)
        rng = np.random.default_rng(0)
        rel = {n: [] for n in parent if parent[n] is not None}
        for trial in range(9):
            q = self.reference_qpos.copy()
            if trial:
                for j in range(1, m.njnt):
                    if m.jnt_limited[j]:
                        lo, hi = m.jnt_range[j]
                        q[m.jnt_qposadr[j]] = rng.uniform(lo, hi)
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            for child, par in parent.items():
                if par is None:
                    continue
                pp, pr = self._site_pose(d, par)
                cp, _ = self._site_pose(d, child)
                rel[child].append(pr.T @ (cp - pp))
        order = []
        seen = {self_root := next(n for n, p in parent.items() if p is None)}
        while len(order) < len(rel):
            for child, par in parent.items():
                if par in seen and child not in seen:
                    seen.add(child)
                    order.append(child)
        out = []
        for child in order:
            values = np.asarray(rel[child])
            rigid = float(np.max(np.linalg.norm(values - values[0], axis=1))) < 1e-6
            out.append(Segment(parent[child], child, rigid, float(np.linalg.norm(values[0])), values[0]))
        del self_root
        return out

    # ------------------------------------------------------------------ source geometry
    def _source(self, clip: Clip, frame: int, site: str) -> tuple[np.ndarray, np.ndarray]:
        j = clip.joint_index(self.correspondence[site])
        return clip.world_positions[frame, j], clip.world_rotations[frame, j]

    def _body_joints(self, clip: Clip) -> np.ndarray:
        """Source joints in the subtree of the root site's joint (a BVH ground-projection root is excluded)."""
        root = clip.joint_index(self.correspondence[self.root_site])
        keep = []
        for j in range(len(clip.joint_names)):
            p = j
            while p >= 0 and p != root:
                p = int(clip.parents[p])
            if p == root:
                keep.append(j)
        return np.asarray(keep)

    def _bone(self, clip: Clip, site: str) -> int:
        """Source joint whose frame carries the segment ending at ``site`` (its BVH parent)."""
        return int(clip.parents[clip.joint_index(self.correspondence[site])])

    def human_segment_length(self, clip: Clip, frame: int, seg: Segment) -> float:
        a, _ = self._source(clip, frame, seg.parent)
        b, _ = self._source(clip, frame, seg.child)
        return float(np.linalg.norm(b - a))

    def targets(self, clip: Clip, frame: int, root_xy_origin: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Scaled, direction-preserving robot-sized human: site -> (position, orientation)."""
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        hp, hr = self._source(clip, frame, self.root_site)
        p = hp.copy()
        p[:2] = root_xy_origin + (hp[:2] - root_xy_origin) * self.root_scale
        p[2] = hp[2] * self.root_scale
        out[self.root_site] = (p, hr @ self.rotation_offsets[self.root_site])
        for seg in self.segments:
            pp, pr = out[seg.parent]
            a, _ = self._source(clip, frame, seg.parent)
            b, br = self._source(clip, frame, seg.child)
            if seg.rigid:
                cp = pp + pr @ seg.rigid_offset
            else:
                # robot segment vector carried by the human bone that ends at the child joint
                cp = pp + clip.world_rotations[frame, self._bone(clip, seg.child)] @ seg.bone_offset
            out[seg.child] = (cp, br @ self.rotation_offsets[seg.child])
        return out

    # ------------------------------------------------------------------ calibration
    def calibrate(self, clip: Clip, frame: int) -> dict:
        """Fit robot directions to one declared human frame; derive orientation offsets and scale."""
        m = self.model
        d = mujoco.MjData(m)
        # articulated segments (weight 1), rigid leaf pairs and rigid siblings (weight 0.5): a rigid
        # leaf (hand, foot) fixes the twist of its limb; siblings fix the lateral axes of rigid parts.
        # Rigid pairs inside the trunk are left out: their geometry is the body's, not the pose's.
        parents_of = {s.parent for s in self.segments}
        pairs = [(s.parent, s.child, 1.0 if not s.rigid else 0.5) for s in self.segments
                 if not s.rigid or s.child not in parents_of]  # rigid pairs only at leaves (hands, feet)
        children: dict[str, list[str]] = {}
        for s in self.segments:
            if s.rigid:
                children.setdefault(s.parent, []).append(s.child)
        for sib in children.values():
            pairs += [(sib[i], sib[k], 0.5) for i in range(len(sib)) for k in range(i + 1, len(sib))]
        hp = {n: self._source(clip, frame, n)[0] for n in self.sites}

        used: list[tuple[str, str]] = []

        def dirs(q: np.ndarray) -> np.ndarray:
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            rp = {n: self._site_pose(d, n)[0] for n in self.sites}
            res = []
            used.clear()
            for a, b, w in pairs:
                hv, rv = hp[b] - hp[a], rp[b] - rp[a]
                if np.linalg.norm(hv) < 0.01 or np.linalg.norm(rv) < 0.01:
                    continue
                used.append((a, b))
                res.append(w * (rv / np.linalg.norm(rv) - hv / np.linalg.norm(hv)))
            return np.concatenate(res)

        q = self.reference_qpos.copy()
        lo, hi = self._bounds(q)

        def f(x):
            qq = self._qpos(q, x)
            return np.concatenate([dirs(qq), self.calibration_posture_weight * x[6:]])

        idx = self.solved_dofs

        def g(z):
            x = np.zeros(m.nv)
            x[idx] = z
            return f(x)

        sol = least_squares(g, np.zeros(len(idx)), bounds=(lo, hi), method="trf", max_nfev=600, x_scale=1.0)
        x = np.zeros(m.nv)
        x[idx] = sol.x
        qc = self._qpos(q, x)
        d.qpos[:] = qc
        mujoco.mj_kinematics(m, d)
        res = dirs(qc).reshape(-1, 3) / np.asarray([w for a, b, w in pairs if (a, b) in used])[:, None]
        # angle residual per used pair
        ang = 2 * np.degrees(np.arcsin(np.clip(np.linalg.norm(res, axis=1) / 2, 0, 1)))
        self._calibration_pairs = {f"{a}->{b}": round(float(v), 2) for (a, b), v in zip(used, ang)}
        for n in self.sites:
            _, hr = self._source(clip, frame, n)
            _, rr = self._site_pose(d, n)
            self.rotation_offsets[n] = hr.T @ rr
        # segment lengths at the calibration pose (articulated lengths may depend on q)
        segs = []
        for s in self.segments:
            pp, pr = self._site_pose(d, s.parent)
            cp, _ = self._site_pose(d, s.child)
            bone = clip.world_rotations[frame, self._bone(clip, s.child)]
            segs.append(Segment(s.parent, s.child, s.rigid, float(np.linalg.norm(cp - pp)),
                                pr.T @ (cp - pp) if s.rigid else s.rigid_offset, bone.T @ (cp - pp)))
        self.segments = segs
        # root scale: robot vs human length of the articulated chains below the root (support chains)
        ratios = []
        for s in self.segments:
            if not s.rigid:
                h = self.human_segment_length(clip, frame, s)
                if h > 0.05:
                    ratios.append((s.child, s.length / h))
        root_p, _ = self._site_pose(d, self.root_site)
        lowest_robot = self._lowest_point(d)
        human_root = self._source(clip, frame, self.root_site)[0][2]
        human_low = float(np.min(clip.world_positions[frame, self._body_joints(clip), 2]))
        # vertical scale: robot root height above its lowest point / human root above its lowest point
        self.root_scale = float((root_p[2] - lowest_robot) / (human_root - human_low))
        self.calibration_qpos = qc
        self.calibration_report = {
            "frame": frame, "max_direction_deg": float(ang.max()), "mean_direction_deg": float(ang.mean()),
            "root_scale": self.root_scale, "segment_ratios": {k: round(v, 4) for k, v in ratios},
            "segments": [{"parent": s.parent, "child": s.child, "rigid": s.rigid, "length_m": round(s.length, 4)}
                         for s in self.segments],
            "pairs_deg": getattr(self, "_calibration_pairs", {}), "solved_dofs": int(len(idx)), "held_dofs": int(m.nv - len(idx)),
        }
        return self.calibration_report

    # ------------------------------------------------------------------ helpers
    def _qpos(self, base: np.ndarray, x: np.ndarray) -> np.ndarray:
        q = base.copy()
        mujoco.mj_integratePos(self.model, q, x, 1.0)
        return q

    def _bounds(self, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = self.model
        lo = np.full(m.nv, -np.inf)
        hi = np.full(m.nv, np.inf)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE or m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                if m.jnt_limited[j]:
                    adr, qadr = m.jnt_dofadr[j], m.jnt_qposadr[j]
                    lo[adr] = m.jnt_range[j][0] - base[qadr]
                    hi[adr] = m.jnt_range[j][1] - base[qadr]
        lo, hi = lo[self.solved_dofs], hi[self.solved_dofs]
        # tiny margin so a base at the limit stays strictly feasible for TRF
        lo = np.minimum(lo, -1e-9)
        hi = np.maximum(hi, 1e-9)
        return lo, hi

    def _lowest_point(self, data: mujoco.MjData) -> float:
        return float(np.min(self.body_points_z(data)))

    def body_points_z(self, data: mujoco.MjData) -> np.ndarray:
        """Lowest world z per visual mesh geom, from its mesh vertices."""
        m = self.model
        if not hasattr(self, "_mesh_cache"):
            cache = []
            for g in range(m.ngeom):
                if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_bodyid[g] == 0:
                    continue
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                if len(v) > 300:
                    v = v[np.linspace(0, len(v) - 1, 300).astype(int)]
                cache.append((g, np.asarray(v, float)))
            self._mesh_cache = cache
        out = []
        for g, v in self._mesh_cache:
            R = data.geom_xmat[g].reshape(3, 3)
            out.append(np.min(v @ R[2]) + data.geom_xpos[g][2])
        return np.asarray(out)

    # ------------------------------------------------------------------ per-frame solve
    def solve(self, clip: Clip, frames: np.ndarray | None = None, progress=None) -> "Solution":
        m = self.model
        d = mujoco.MjData(m)
        frames = np.arange(clip.frame_count) if frames is None else np.asarray(frames)
        dt = clip.frame_time_s * (int(frames[1] - frames[0]) if len(frames) > 1 else 1)
        idx = self.solved_dofs
        hip0 = self._source(clip, int(frames[0]), self.root_site)[0][:2].copy()
        q = self.calibration_qpos.copy()
        # place the root at the first frame's target before the warm-started sequence
        t0 = self.targets(clip, int(frames[0]), hip0)
        q[:3] = t0[self.root_site][0]
        q[3:7] = Rotation.from_matrix(t0[self.root_site][1] @ self.sites[self.root_site].rotation.T).as_quat(
            scalar_first=True)
        names = list(self.sites)
        # a site reached rigidly that starts an articulated segment sits on a multi-axis proximal joint:
        # its own link carries only part of that joint, so it receives no orientation target
        rigid_child = {s.child for s in self.segments if s.rigid}
        starts = {s.parent for s in self.segments if not s.rigid}
        self.orientation_sites = [n for n in names if not (n in rigid_child and n in starts)]
        wr_site = {n: (self.root_orientation_weight if n == self.root_site else self.orientation_weight)
                   * (n in self.orientation_sites) for n in names}
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        qs, err_p, err_r, nfev = [], [], [], []
        prev = q.copy()
        for k, fr in enumerate(frames):
            tg = self.targets(clip, int(fr), hip0)
            base = self._anchor_root(d, q.copy(), tg[self.root_site])
            lo, hi = self._bounds(base)
            if k:
                # joint speed bound around the previous solution: an IK branch switch becomes a bounded
                # transition instead of a one-frame pop (root dofs stay free)
                step = self.max_joint_speed * dt
                from_prev = np.zeros(m.nv)
                mujoco.mj_differentiatePos(m, from_prev, 1.0, prev, base)  # base - prev
                jl, jh = -step - from_prev[idx], step - from_prev[idx]
                jl[:6], jh[:6] = -np.inf, np.inf
                lo, hi = np.maximum(lo, jl), np.minimum(hi, jh)
                lo, hi = np.minimum(lo, -1e-9), np.maximum(hi, 1e-9)  # the prediction itself stays feasible
            ref_dx = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, ref_dx, 1.0, base, self.calibration_qpos)

            def residual(z):
                x = np.zeros(m.nv)
                x[idx] = z
                d.qpos[:] = self._qpos(base, x)
                mujoco.mj_kinematics(m, d)
                r = []
                for n in names:
                    p, R = self._site_pose(d, n)
                    tp, tR = tg[n]
                    wp = self.root_position_weight if n == self.root_site else self.position_weight
                    wr = wr_site[n]
                    r.append(wp * (p - tp))
                    r.append(wr * so3_log(R @ tR.T))
                # acceleration prior: deviation from the constant-velocity prediction
                r.append(self.velocity_weight * x[idx][6:])
                r.append(self.posture_weight * (x - ref_dx)[idx][6:])
                return np.concatenate(r)

            def jacobian(z):
                x = np.zeros(m.nv)
                x[idx] = z
                d.qpos[:] = self._qpos(base, x)
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                rows = []
                for n in names:
                    s = self.sites[n]
                    p, _ = self._site_pose(d, n)
                    mujoco.mj_jac(m, d, jacp, jacr, p, s.body)
                    wp = self.root_position_weight if n == self.root_site else self.position_weight
                    wr = wr_site[n]
                    rows.append(wp * jacp[:, idx])
                    rows.append(wr * jacr[:, idx])
                eye = np.eye(len(idx))[6:]
                rows.append(self.velocity_weight * eye)
                rows.append(self.posture_weight * eye)
                return np.vstack(rows)

            sol = least_squares(residual, np.zeros(len(idx)), jac=jacobian, bounds=(lo, hi), method="trf",
                                max_nfev=60 if k else 300, xtol=1e-7, ftol=1e-9)
            x = np.zeros(m.nv)
            x[idx] = sol.x
            q = self._qpos(base, x)
            qs.append(q.copy())
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            ep, er = [], []
            for n in names:
                p, R = self._site_pose(d, n)
                ep.append(np.linalg.norm(p - tg[n][0]))
                er.append(np.degrees(np.linalg.norm(so3_log(R @ tg[n][1].T))))
            err_p.append(ep)
            err_r.append(er)
            nfev.append(sol.nfev)
            # constant-velocity prediction is the next frame's base (warm start + prior)
            dq = np.zeros(m.nv)
            if k:  # the first frame starts from rest: no velocity is carried out of the calibration pose
                mujoco.mj_differentiatePos(m, dq, 1.0, prev, q)
            prev = q
            nxt = q.copy()
            mujoco.mj_integratePos(m, nxt, dq, 1.0)
            q = self._clip_limits(nxt)
            if progress and k % 100 == 0:
                progress(k, len(frames))
        return Solution(np.asarray(qs), frames, names, np.asarray(err_p), np.asarray(err_r), np.asarray(nfev))

    def _anchor_root(self, data: mujoco.MjData, q: np.ndarray, target: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Place the floating root so the root site sits exactly on its target pose (joints unchanged).

        The root site's target is the most reliable signal in the frame; starting every solve from it keeps
        a fast rotation (a flip) from accumulating root drift that the limbs then fight."""
        m = self.model
        data.qpos[:] = q
        mujoco.mj_kinematics(m, data)
        p_site, R_site = self._site_pose(data, self.root_site)
        root = m.jnt_bodyid[0]
        p_root, R_root = data.xpos[root].copy(), data.xmat[root].reshape(3, 3).copy()
        tp, tR = target
        R_new = tR @ R_site.T @ R_root
        p_new = tp - R_new @ (R_root.T @ (p_site - p_root))
        q[:3] = p_new
        q[3:7] = Rotation.from_matrix(R_new).as_quat(scalar_first=True)
        return q

    def _clip_limits(self, q: np.ndarray) -> np.ndarray:
        m = self.model
        q = q.copy()
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.jnt_limited[j]:
                a = m.jnt_qposadr[j]
                q[a] = np.clip(q[a], *m.jnt_range[j])
        return q

    # ------------------------------------------------------------------ post-processing
    def ground_and_smooth(self, clip: Clip, sol: "Solution", fps: float, cutoff_hz: float = 12.0) -> np.ndarray:
        """Low-pass joint/root trajectories, then make the lowest body point follow the scaled human's."""
        m = self.model
        q = sol.qpos.copy()
        # unwrap root quaternion sign for filtering
        quat = q[:, 3:7].copy()
        for t in range(1, len(quat)):
            if np.dot(quat[t], quat[t - 1]) < 0:
                quat[t] = -quat[t]
        q[:, 3:7] = quat
        if len(q) > 15 and cutoff_hz < fps / 2:
            b, a = butter(2, cutoff_hz / (fps / 2))
            q = filtfilt(b, a, q, axis=0)
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
        q = np.asarray([self._clip_limits(x) for x in q])
        d = mujoco.MjData(m)
        robot_low = np.empty(len(q))
        for t, x in enumerate(q):
            d.qpos[:] = x
            mujoco.mj_kinematics(m, d)
            robot_low[t] = self._lowest_point(d)
        human_low = np.min(clip.world_positions[sol.frames][:, self._body_joints(clip), 2], axis=1) * self.root_scale
        shift = np.maximum(human_low, 0.0) - robot_low
        if len(shift) > 15:
            b, a = butter(2, min(6.0, fps / 2 - 1) / (fps / 2))
            shift_s = filtfilt(b, a, shift)
        else:
            shift_s = shift
        # never penetrate after smoothing
        shift_s = np.maximum(shift_s, -robot_low)
        q[:, 2] += shift_s
        self.ground_shift = shift_s
        return q


@dataclass
class Solution:
    qpos: np.ndarray
    frames: np.ndarray
    site_names: list[str]
    position_error_m: np.ndarray
    orientation_error_deg: np.ndarray
    nfev: np.ndarray
