"""CPU test harness: XMimic instances without the MuJoCo Warp simulator (it needs a GPU), a synthetic but realistic replay
buffer (states near the reference motion), fitted normalizers and non-trivial networks and optimizer states.

Only the learner parts (world-model update, policy update, critic) are exercised; collection needs the GPU simulator.
"""

from __future__ import annotations

import os
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from xmimic import buffer as rb  # noqa: E402
from xmimic import nets, optim  # noqa: E402
from xmimic.algorithm import ERR_FLOOR_PARTS, Mirror, widths  # noqa: E402
from xmimic.config import Config  # noqa: E402
from xmimic.kin import Kinematics  # noqa: E402
from xmimic.mdp import MDP, flatten_state, inv_dim, obs_dim, state_dim, unflatten_state  # noqa: E402
from xmimic.model import load_meta  # noqa: E402
from xmimic.motion import CONTROL_DT, GOAL_Q_OFFSET, load_motion  # noqa: E402
from xmimic.rot import mm3, so3_exp  # noqa: E402
from xmimic.sim import XiLayout  # noqa: E402

CLIPS = ("dance1_subject1", "walk1_subject1", "fallAndGetUp3_subject1")
MOTION_DIR = os.environ.get("XMIMIC_MOTION_DIR", os.path.join(ROOT, "data", "lafan1_g1"))
_MOTION_CACHE = {}

# small learner sizes for numerics (the program structure does not depend on the widths)
SMALL = dict(num_envs=32, buffer_steps=48, pi_batch=64, wm_batch=64, enc_hidden="64,48,32,16", dec_hidden="64,48,32,16",
             wm_hidden="64,64,64", eprobe_share=0.0, critic_minibatches=4, critic_epochs=2, wm_batches=3)


def motion_for(kin, feet, clips=CLIPS):
    key = tuple(clips)
    if key not in _MOTION_CACHE:
        d = tempfile.mkdtemp(prefix="xm_motion_")
        for c in clips:
            os.symlink(os.path.join(MOTION_DIR, c + ".csv"), os.path.join(d, c + ".csv"))
        _MOTION_CACHE[key] = load_motion(d, kin, feet)
    return _MOTION_CACHE[key]


def build(cls, cfg: Config, clips=CLIPS):
    """cls.__init__ without the simulators."""
    xm = cls.__new__(cls)
    xm.cfg = c = cfg
    xm.mj_model, xm.meta = None, load_meta()
    meta = xm.meta
    xm.J = J = len(meta["joint_names"])
    xm.kin = Kinematics(meta)
    feet = [meta["tracked_bodies"].index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    xm.layout = XiLayout(meta)
    xm.mdp = MDP(xm.kin, meta["default_joint_pos"], feet, xm.layout.dim)
    xm.motion = motion_for(xm.kin, feet, clips)
    xm.goal_dim = xm.motion.goal_dim(c.goal_frames)
    xm.obs_dim = obs_dim(J)
    xm.critic_in = inv_dim(J) + xm.mdp.critic_extra + xm.goal_dim
    xm.wm_in = inv_dim(J) + J + xm.layout.dim
    xm.wm_out = 12 + 2 * J
    xm.ref = (GOAL_Q_OFFSET, np.asarray(meta["default_joint_pos"], np.float32), np.asarray(meta["action_scale"], np.float32))
    xm.mirror = Mirror(meta["joint_names"], xm.layout)
    xm.err_floor = np.concatenate([np.full((n,), v, np.float32) for n, v in ERR_FLOOR_PARTS]
                                  + [np.full((J,), 3e-3, np.float32), np.full((J,), 5e-2, np.float32)])  # fmt: skip
    xm.n_mirror = int(c.wm_batch * c.wm_mirror)
    xm.probes = False
    return xm


def _rand_xi(key, xm, shape):
    L = xm.layout
    kf, ko, kc = jax.random.split(key, 3)
    lo, hi = xm.meta["dr"]["friction_range"]
    fr = jax.random.uniform(kf, shape + (1,), minval=lo, maxval=hi)
    off = (2 * jax.random.uniform(ko, shape + (xm.J,)) - 1) * L.offset_half
    cr = np.asarray(xm.meta["com_range"], np.float32)
    com = jax.random.uniform(kc, shape + (3,), minval=cr[:, 0], maxval=cr[:, 1])
    return jnp.concatenate((fr, off, com), -1)


def _noisy_state(key, S, scale=1.0):
    p, R, v, w, q, qd = S
    k = jax.random.split(key, 6)
    n = lambda kk, x, s: x + s * scale * jax.random.normal(kk, x.shape)  # noqa: E731
    R = mm3(so3_exp(0.1 * scale * jax.random.normal(k[1], R.shape[:-2] + (3,))), R)
    return n(k[0], p, 0.03), R, n(k[2], v, 0.2), n(k[3], w, 0.3), n(k[4], q, 0.05), n(k[5], qd, 0.5)


def make_state(xm, seed=0):
    """(consts, carry) with a full synthetic replay buffer, fitted normalizers and non-trivial networks and optimizer moments."""
    c, J, mo = xm.cfg, xm.J, xm.motion
    key = jax.random.PRNGKey(seed)
    ks = iter(jax.random.split(key, 64))
    T, N = c.buffer_steps, c.num_envs
    f = jax.random.randint(next(ks), (T, N), 0, mo.num_frames - 1)
    f = jnp.minimum(f, mo.clip_end[mo.clip_id[f]] - 1)
    k0, k1 = next(ks), next(ks)
    S0, S1 = jax.jit(lambda f, mo: (flatten_state(_noisy_state(k0, mo.ref_state(f))), flatten_state(_noisy_state(k1, mo.ref_state(f + 1)))))(f, mo)
    a = 0.5 * jax.random.normal(next(ks), (T, N, J))
    a_prev = 0.5 * jax.random.normal(next(ks), (T, N, J))
    xi = _rand_xi(next(ks), xm, (T, N))
    buf = rb.init(T, N, state_dim(J), J, xm.layout.dim)
    cut = jax.random.uniform(next(ks), (T, N)) < 0.05
    ep_step = jax.random.randint(next(ks), (T, N), 0, 40)
    for k_, v in (("s", S0), ("s1", S1), ("a", a), ("a_prev", a_prev), ("xi", xi), ("f", f), ("ep_step", ep_step), ("cut", cut)):
        buf[k_] = v.astype(buf[k_].dtype)
    buf["ptr"], buf["size"] = jnp.asarray(0, jnp.int32), jnp.asarray(T, jnp.int32)

    pi, pi_b = nets.pi_init(next(ks), xm.obs_dim, xm.goal_dim, J, c.hist, enc_hidden=widths(c.enc_hidden), dec_hidden=widths(c.dec_hidden))
    wm, wm_b = nets.wm_init(next(ks), xm.wm_in, xm.wm_out, widths(c.wm_hidden))
    critic, critic_b = nets.critic_init(next(ks), xm.critic_in)
    # a trained-looking world model and policy: non-zero output layers and LayerNorm affine parameters
    kk = next(ks)
    wm["net"][-1]["w"] = 0.02 * jax.random.normal(kk, wm["net"][-1]["w"].shape)
    wm["net"][-1]["b"] = 0.01 * jax.random.normal(next(ks), wm["net"][-1]["b"].shape)
    for layer in wm["net"][:-1]:
        layer["ln_g"] = 1 + 0.1 * jax.random.normal(next(ks), layer["ln_g"].shape)
        layer["ln_b"] = 0.1 * jax.random.normal(next(ks), layer["ln_b"].shape)
    pi["dec"][-1]["w"] = pi["dec"][-1]["w"] * 30.0

    @jax.jit
    def fit_norms(buf, pi_b, wm_b, critic_b, mo):  # normalizers from the buffer
        Sf = unflatten_state(buf["s"].reshape(T * N, -1))
        S1f = unflatten_state(buf["s1"].reshape(T * N, -1))
        ff, xif = buf["f"].reshape(-1), buf["xi"].reshape(T * N, -1)
        o = xm.mdp.obs(mo, Sf, buf["a_prev"].reshape(T * N, -1), xif, ff)
        pi_b = {"obs_norm": nets.rn_update(pi_b["obs_norm"], o), "goal_norm": nets.rn_update(pi_b["goal_norm"], mo.goal(ff, c.goal_frames))}
        wm_b = {"in_norm": nets.rn_update(wm_b["in_norm"], nets.wm_inputs(Sf, buf["a"].reshape(T * N, -1), xm.layout.normalize(xif))),
                "out_norm": nets.rn_update(wm_b["out_norm"], nets.wm_targets(Sf, S1f, CONTROL_DT)),
                "inc_norm": nets.rn_update(wm_b["inc_norm"], nets.increments(Sf, S1f))}  # fmt: skip
        X = jnp.concatenate((xm.mdp.critic_features(mo, Sf, ff, xm.layout.normalize(xif)), mo.goal(ff, c.goal_frames)), -1)
        return pi_b, wm_b, {"norm": nets.rn_update(critic_b["norm"], X)}

    pi_b, wm_b, critic_b = fit_norms(buf, pi_b, wm_b, critic_b, mo)
    pi_mask = {"enc": nets.muon_mask(pi["enc"]), "dec": nets.muon_mask(pi["dec"])}
    wm_mask = {"net": nets.muon_mask(wm["net"])}
    critic_mask = jax.tree.map(lambda _: False, critic)
    xm.pi_mask, xm.wm_mask, xm.critic_mask = pi_mask, wm_mask, critic_mask

    def opt_with_moments(params, mask, k):
        st = optim.init(params, mask)
        leaves, tdef = jax.tree.flatten(st["leaves"])
        kl = jax.random.split(k, len(leaves))
        leaves = [0.01 * jnp.abs(jax.random.normal(kq, x.shape)) * 1e-2 if i % 2 else 1e-3 * jax.random.normal(kq, x.shape)
                  for i, (kq, x) in enumerate(zip(kl, leaves))]  # fmt: skip
        return {"leaves": jax.tree.unflatten(tdef, leaves), "t": jnp.asarray(7, jnp.int32)}

    def fix_v(st):  # Adam second moments must be >= 0
        return jax.tree_util.tree_map_with_path(lambda p, x: jnp.abs(x) if getattr(p[-1], "key", None) == "v" else x, st)

    avg = jax.tree.map(lambda x: x + 0.001 * jnp.ones_like(x), (pi, pi_b))
    carry = {"buf": buf, "pi": pi, "pi_b": pi_b, "avg": avg, "avg_n": jnp.asarray(3, jnp.int32), "wm": wm, "wm_b": wm_b,
             "critic": critic, "critic_b": critic_b,
             "opt_pi": fix_v(opt_with_moments(pi, pi_mask, next(ks))), "opt_wm": fix_v(opt_with_moments(wm, wm_mask, next(ks))),
             "opt_v": fix_v(opt_with_moments(critic, critic_mask, next(ks))),
             "nonfinite": jnp.zeros((), jnp.int32)}  # fmt: skip
    return {"motion": mo}, carry


def max_rel(a, b, floor=None):
    """max |a-b| / (max |a| + tiny) per leaf: the worst leaf value and its path."""
    worst, where = 0.0, None
    for (pa, x), y in zip(jax.tree_util.tree_flatten_with_path(a)[0], jax.tree.leaves(b)):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if x.size == 0:
            continue
        scale = max(np.abs(x).max(), floor if floor is not None else 1e-30)
        r = np.abs(x - y).max() / scale
        if not np.isfinite(r):
            r = np.inf if not (np.array_equal(np.isfinite(x), np.isfinite(y))) else 0.0
        if r > worst:
            worst, where = r, jax.tree_util.keystr(pa)
    return worst, where
