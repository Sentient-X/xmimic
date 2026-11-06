"""CPU equivalence tests of the fast learner (xmimic/fast.py) against the reference (xmimic/algorithm.py).

    python tests/test_fast_equiv.py [groups ...]     groups: noise perm sample muon rot ln imag policy wm (default: all)

Needs the motion data (tools/download_data.sh) and runs on the CPU in a few minutes (JAX_PLATFORMS=cpu).

The fast learner rounds differently in bf16 than the reference, and any fp32-level reordering flips individual bf16
roundings, so exact agreement is not expected everywhere. Each piece is checked at its own level:
  exact  bitwise (batched random draws, start sampling);
  fp32   with every bf16 cast of the networks replaced by float32 (F32Matmuls), fast and reference compute the same math
         and agree to fp32 rounding;
  bf16   as trained, the deviation is compared with calibrations measured in the same run: the reference's own bf16 error
         (reference vs reference with float32 matmuls), the error of a torch-autocast emulation (BlinkMimic's precision),
         and the chaos floor of the bf16 Muon iteration (the reference against itself from weights perturbed by 2 ulp).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from xmimic import fast as fast_mod  # noqa: E402
from xmimic import fast_nets as fn  # noqa: E402
from xmimic import nets, optim, rot  # noqa: E402
from xmimic.algorithm import XMimic  # noqa: E402
from xmimic.config import Config  # noqa: E402
from xmimic.fast import FastXMimic, keyed_map  # noqa: E402

TOL = {"exact": 1e-6, "fp32": 2e-5, "bf16": 3e-2}
RESULTS = []


def report(name, cls, err, where, tol=None, extra=""):
    tol = TOL[cls] if tol is None else tol
    ok = err <= tol
    RESULTS.append((name, cls, err, tol, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name:46s} [{cls:5s}] max rel err {err:9.2e} (tol {tol:.2g})  {where or ''} {extra}", flush=True)
    return ok


def rel_l2(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    num = sum(float(np.sum((np.asarray(x, np.float64) - np.asarray(y, np.float64)) ** 2)) for x, y in zip(la, lb))
    den = sum(float(np.sum(np.asarray(x, np.float64) ** 2)) for x in la)
    return np.sqrt(num / max(den, 1e-300))


def cmp_tree(name, cls, a, b, tol=None, extra=""):
    err, where = H.max_rel(a, b)
    return report(name, cls, err, where, tol, extra)


def perturb_ulp(tree, seed=0, ulps=2.0):
    """Every float leaf times (1 +- ulps * 2^-23): a last-bits perturbation of the starting point."""
    leaves, tdef = jax.tree.flatten(tree)
    ks = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    out = [x * (1 + ulps * 2.0**-23 * jnp.sign(jax.random.normal(k, x.shape))) if jnp.issubdtype(x.dtype, jnp.floating) else x
           for k, x in zip(ks, leaves)]  # fmt: skip
    return jax.tree.unflatten(tdef, out)


class F32Matmuls:
    """Every bf16 cast of the networks (policy, world model, critic) becomes float32: without bf16 roundings, the fast
    learner must match the reference to fp32 rounding (a check of the math, independent of rounding flips)."""

    def __enter__(self):
        self.mods = [nets, fn, fast_mod]
        self.old = [m.BF16 for m in self.mods]
        for m in self.mods:
            m.BF16 = jnp.float32
        return self

    def __exit__(self, *a):
        for m, o in zip(self.mods, self.old):
            m.BF16 = o


def ln_ref(h, g, b, eps=1e-5):
    """nets.mlp_apply's LayerNorm, as a plain function."""
    mu = h.mean(-1, keepdims=True)
    var = ((h - mu) ** 2).mean(-1, keepdims=True)
    return (h - mu) * jax.lax.rsqrt(var + eps) * g + b


def torch_autocast_mlp(layers, x, act, *, defer=False, taps=None, out_dtype=jnp.float32):
    """torch.autocast(bfloat16) semantics for an MLP (BlinkMimic --amp bf16), as a precision reference: Linear = bf16 GEMM
    with fp32 accumulation, bias added in bf16, bf16 output; ELU on bf16 tensors; LayerNorm in fp32 on the upcast input.
    JAX's autodiff then runs the backward pass in these dtypes, as torch does."""
    assert not defer and taps is None
    h, ins = x, []
    for i, layer in enumerate(layers):
        xb = h.astype(jnp.bfloat16)
        ins.append(xb)
        y = jnp.dot(xb, layer["w"].astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16) + layer["b"].astype(jnp.bfloat16)
        if i == len(layers) - 1:
            return y.astype(out_dtype), ins
        h = act(ln_ref(y.astype(jnp.float32), layer["ln_g"], layer["ln_b"])) if "ln_g" in layer else act(y)


class Env:
    """Reference and fast learner sharing one synthetic state."""

    def __init__(self, horizon):
        self.cfg = Config(**{**H.SMALL, "horizon": horizon})
        self.ref = H.build(XMimic, self.cfg)
        self.fast = H.build(FastXMimic, self.cfg)
        self.consts, self.carry = H.make_state(self.ref)
        self.fast.pi_mask, self.fast.wm_mask, self.fast.critic_mask = self.ref.pi_mask, self.ref.wm_mask, self.ref.critic_mask


# ---------------------------------------------------------------- exact pieces
def test_noise_vmap(env):
    keys = jax.random.split(jax.random.PRNGKey(3), 8)
    a = jax.jit(lambda ks: keyed_map(lambda k: jax.random.uniform(k, (16, 99)), ks))(keys)
    b = jnp.stack([jax.random.uniform(k, (16, 99)) for k in keys])
    report("batched draws (keyed_map) == per-key draws", "exact", 0.0 if bool(jnp.array_equal(a, b)) else 1.0, "", extra="(bitwise)")


def test_permutation_vmap(env):
    ks = jax.random.split(jax.random.PRNGKey(5), 5)
    n = 4096
    a = jax.jit(lambda ks: keyed_map(lambda k: jax.random.permutation(k, n), ks))(ks)
    b = jax.jit(lambda ks: jnp.stack([jax.random.permutation(k, n) for k in ks]))(ks)
    report("critic permutations, batched == loop", "exact", 0.0 if bool(jnp.array_equal(a, b)) else 1.0, "", extra="(bitwise)")


def test_fast_sample(env):
    key = jax.random.PRNGKey(11)
    for buf_size in (env.carry["buf"]["size"], jnp.asarray(10, jnp.int32)):
        buf = {**env.carry["buf"], "size": buf_size, "ptr": jnp.asarray(7, jnp.int32)}
        a = jax.jit(lambda b, k: env.ref.sample_valid_starts(env.consts["motion"], b, k, env.cfg.pi_batch))(buf, key)
        b = jax.jit(lambda b, k: env.fast.sample_valid_starts(env.consts["motion"], b, k, env.cfg.pi_batch))(buf, key)
        eq = all(bool(jnp.array_equal(x, y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
        report(f"start sampling (buffer fill {int(buf_size)})", "exact", 0.0 if eq else 1.0, "", extra="(bitwise)")


# ---------------------------------------------------------------- building blocks
def test_muon_stacked(env):
    pi, opt = env.carry["pi"], env.carry["opt_pi"]
    g = jax.tree.map(lambda x: 0.01 * jax.random.normal(jax.random.PRNGKey(x.size % 97), x.shape), pi)
    a = jax.jit(lambda p, g, s: optim.step(p, g, s, env.ref.pi_mask, 1e-3, betas=(0.9, 0.95), ok=jnp.bool_(True)))(pi, g, opt)
    b = jax.jit(lambda p, g, s: fn.optim_step(p, g, s, env.ref.pi_mask, 1e-3, betas=(0.9, 0.95), ok=jnp.bool_(True)))(pi, g, opt)
    cmp_tree("stacked Muon: optimizer step (policy)", "bf16", a, b, tol=1e-3)
    wm, ow = env.carry["wm"], env.carry["opt_wm"]
    g = jax.tree.map(lambda x: 0.01 * jax.random.normal(jax.random.PRNGKey(x.size % 89), x.shape), wm)
    a = jax.jit(lambda p, g, s: optim.step(p, g, s, env.ref.wm_mask, 1e-3))(wm, g, ow)
    b = jax.jit(lambda p, g, s: fn.optim_step(p, g, s, env.ref.wm_mask, 1e-3))(wm, g, ow)
    cmp_tree("stacked Muon: optimizer step (world model)", "bf16", a, b, tol=1e-3)


def test_rot(env):
    k = jax.random.split(jax.random.PRNGKey(7), 4)
    v = jax.random.normal(k[0], (4096, 3)) * jnp.asarray([0.3, 1.0, 3.1])[jax.random.randint(k[1], (4096, 1), 0, 3)]
    R = rot.so3_exp(v)
    # near-pi rotations and exact axis rotations exercise every branch and the ties
    R = jnp.concatenate((R, rot.so3_exp(jnp.asarray([[np.pi - 1e-4, 0, 0], [0, np.pi, 0], [0, 0, 3.14159], [0, 0, 0.0], [1e-7, 0, 0]]))))
    report("elementwise rotations: mat_to_quat", "fp32", float(jnp.abs(rot.mat_to_quat(R) - fn.mat_to_quat(R)).max()), "", tol=1e-6)
    report("elementwise rotations: so3_log", "fp32", float(jnp.abs(rot.so3_log(R) - fn.so3_log(R)).max() / jnp.abs(rot.so3_log(R)).max()), "")
    ga = jax.grad(lambda R: (rot.so3_log(R) ** 2 * jnp.arange(1, 4)).sum())(R)
    gb = jax.grad(lambda R: (fn.so3_log(R) ** 2 * jnp.arange(1, 4)).sum())(R)
    same_nan = bool(jnp.array_equal(jnp.isnan(ga), jnp.isnan(gb)))  # both NaN at the exact identity (norm of a zero vector)
    fin = jnp.isfinite(ga)
    err = float(jnp.abs(jnp.where(fin, ga - gb, 0)).max() / jnp.abs(jnp.where(fin, ga, 0)).max())
    report("elementwise rotations: so3_log gradient", "fp32", err if same_nan else 1.0, "", extra=f"(NaN at identity in both: {same_nan})")


def test_ln(env):
    k = jax.random.split(jax.random.PRNGKey(9), 4)
    h = 3.0 * jax.random.normal(k[0], (256, 512)) + 0.5
    g = 1 + 0.1 * jax.random.normal(k[1], (512,))
    b = 0.1 * jax.random.normal(k[2], (512,))
    w = jax.random.normal(k[3], (256, 512))
    fa = lambda h, g, b: (jax.nn.silu(ln_ref(h, g, b)) * w).sum()  # noqa: E731
    fb = lambda h, g, b: (jax.nn.silu(fn.layer_norm(h, g, b)) * w).sum()  # noqa: E731
    _, ga = jax.value_and_grad(fa, argnums=(0, 1, 2))(h, g, b)
    _, gb = jax.value_and_grad(fb, argnums=(0, 1, 2))(h, g, b)
    report("LayerNorm: forward", "exact", float(jnp.abs(ln_ref(h, g, b) - fn.layer_norm(h, g, b)).max()), "")
    cmp_tree("LayerNorm: gradients", "fp32", ga, gb)


# ---------------------------------------------------------------- policy gradient and policy update
def batch_and_keys(env, key):
    k_s, _, k_i, _ = jax.random.split(key, 4)
    start, oh0 = jax.jit(lambda b, k: env.ref.sample_policy_batch(env.consts, b, k))(env.carry["buf"], k_s)
    return start, oh0, jax.random.split(k_i, env.cfg.horizon)


def ref_grads(env, start, oh0, keys):
    c, mo = env.carry, env.consts["motion"]
    f = jax.jit(lambda c, s, o, k: jax.value_and_grad(env.ref.imagine, has_aux=True)(c["pi"], c["pi_b"], c["wm"], c["wm_b"], c["critic"],
                                                                                    c["critic_b"], mo, s, o, k))  # fmt: skip
    (loss, aux), g = f(c, start, oh0, keys)
    return loss, aux, g


def fast_grads(env, start, oh0, keys):
    mo = env.consts["motion"]
    return jax.jit(lambda c, s, o, k: env.fast.policy_grads(c, mo, s, o, k))(env.carry, start, oh0, keys)


def test_imagine_grads(env):
    """The policy gradient through the imagination. With float32 matmuls: the same math to fp32 rounding. As trained: its
    error against the float32 truth must stay within the reference's own bf16 error or that of torch autocast."""
    start, oh0, keys = batch_and_keys(env, jax.random.PRNGKey(21))
    with F32Matmuls():
        l_t, aux_t, g_t = ref_grads(env, start, oh0, keys)  # truth: the same math without bf16 roundings
        l_f, aux_f, g_f32 = fast_grads(env, start, oh0, keys)
    report("imagination: loss (f32 matmuls)", "fp32", abs(float(l_t) - float(l_f)) / abs(float(l_t)), "")
    cmp_tree("imagination: rewards, dones, states (f32 matmuls)", "fp32", (aux_t["r"], aux_t["d"], aux_t["S_all"]),
             (aux_f["r"], aux_f["d"], aux_f["S_all"]))  # fmt: skip
    cmp_tree("imagination: gradient (f32 matmuls)", "fp32", g_t, g_f32)
    _, _, g_r = ref_grads(env, start, oh0, keys)
    _, _, g_f = fast_grads(env, start, oh0, keys)
    old = fn.mlp_apply
    fn.mlp_apply = torch_autocast_mlp
    try:  # the fast learner's structure with torch-autocast layers (no deferred weight gradients)
        mo = env.consts["motion"]
        g_torch = jax.jit(lambda c, s, o, k: jax.value_and_grad(env.fast.imagine, has_aux=True)(
            c["pi"], c["pi_b"], c["wm"], c["wm_b"], c["critic"], c["critic_b"], mo, s, o, k)[1])(env.carry, start, oh0, keys)
    finally:
        fn.mlp_apply = old
    e_ref, e_torch, e_fast = rel_l2(g_t, g_r), rel_l2(g_t, g_torch), rel_l2(g_t, g_f)
    report("imagination: gradient error vs f32 truth (bf16)", "bf16", e_fast, "", tol=max(e_ref, e_torch),
           extra=f"(reference {e_ref:.1e}, torch-autocast emulation {e_torch:.1e})")


def run_policy_update(xm, consts, carry, key):
    return jax.jit(lambda c, k: xm.policy_update(consts, c, k, 4e-4, 1.6e-3))(carry, key)


def _adam_leaves(tree, mask):
    return [x for x, m in zip(jax.tree.leaves(tree), jax.tree.leaves(mask)) if not m]


def test_policy_update(env):
    """One full policy update (BPTT, clipping, Muon and Adam, EMA, critic) against the reference's own variability: the
    policy delta within its bf16 error scale or twice the Muon chaos floor; the critic within its own bf16 error scale; the
    Adam-updated leaves (first and last layers, biases) to fp32 rounding with float32 matmuls."""
    key = jax.random.PRNGKey(31)
    c0, s0 = run_policy_update(env.ref, env.consts, env.carry, key)
    with F32Matmuls():
        c0_32, _ = run_policy_update(env.ref, env.consts, env.carry, key)
        c1_32, _ = run_policy_update(env.fast, env.consts, env.carry, key)
    pc = {**env.carry, "pi": perturb_ulp(env.carry["pi"])}
    d = lambda out, c=env.carry: jax.tree.map(lambda a, b: a - b, out["pi"], c["pi"])  # noqa: E731
    scale = rel_l2(d(c0_32), d(c0))
    chaos = rel_l2(d(c0), d(run_policy_update(env.ref, env.consts, pc, key)[0], pc))
    print(f"      reference policy delta: bf16 vs f32 matmuls rel L2 {scale:.2e} (its bf16 error scale); vs itself from "
          f"2-ulp-perturbed weights {chaos:.2e} (Muon chaos floor)", flush=True)
    c1, s1 = run_policy_update(env.fast, env.consts, env.carry, key)
    dl = rel_l2(d(c0), d(c1))
    report("policy update: policy delta vs reference (bf16)", "bf16", dl, "", tol=max(scale, 2 * chaos),
           extra=f"= {dl / chaos:.2f} x the chaos floor, {dl / scale:.2f} x the bf16 error scale")
    kv = ("critic", "critic_b", "opt_v")
    sv = H.max_rel({k: c0[k] for k in kv}, {k: c0_32[k] for k in kv})[0]
    err_v, where_v = H.max_rel({k: c0[k] for k in kv}, {k: c1[k] for k in kv})
    report("policy update: critic vs reference (bf16)", "bf16", err_v, where_v, tol=max(sv, 1e-5), extra=f"(critic bf16 error scale {sv:.1e})")
    for k in ("pi", "v", "ir", "surv", "gn"):  # bf16 GEMM outputs: forward values differ at the bf16 level (2^-9 = 2e-3)
        report(f"policy update: stat {k}", "bf16", abs(float(s0[k]) - float(s1[k])) / max(abs(float(s0[k])), 1e-12), "", tol=2e-2)
    mask = env.ref.pi_mask
    cmp_tree("policy update: Adam-leaf deltas (f32 matmuls)", "fp32", _adam_leaves(d(c0_32), mask), _adam_leaves(d(c1_32), mask), tol=1e-4)


# ---------------------------------------------------------------- world model
def wm_ref_updates(env, carry, k_w):
    ref = env.ref
    return jax.jit(lambda c, k: jax.lax.scan(lambda cr, kk: ref.wm_update(cr, kk, 4e-4), c, jax.random.split(k, env.cfg.wm_batches)))(carry, k_w)


def test_wm(env):
    """The iteration's world-model updates (presampled windows, deferred weight gradients, stacked Muon): the weight delta
    within the reference's bf16 error scale or twice the Muon chaos floor; with float32 matmuls, the loss to fp32 rounding."""
    k_w = jax.random.PRNGKey(41)
    c0, s0 = wm_ref_updates(env, env.carry, k_w)
    with F32Matmuls():
        c0_32, s0_32 = wm_ref_updates(env, env.carry, k_w)
        c1_32, s1_32 = jax.jit(lambda c, k: env.fast.wm_updates(c, k, 4e-4))(env.carry, k_w)
    report("world model: first-update loss (f32 matmuls)", "fp32", abs(float(s0_32["wm"][0]) - float(s1_32["wm"][0])) / abs(float(s0_32["wm"][0])), "")
    d = lambda out, c=env.carry: jax.tree.map(lambda a, b: a - b, out["wm"], c["wm"])  # noqa: E731
    scale = rel_l2(d(c0_32), d(c0))
    pc = {**env.carry, "wm": perturb_ulp(env.carry["wm"])}
    chaos = rel_l2(d(c0), d(wm_ref_updates(env, pc, k_w)[0], pc))
    print(f"      reference world-model delta: bf16 vs f32 matmuls rel L2 {scale:.2e} (its bf16 error scale); vs itself from "
          f"2-ulp-perturbed weights {chaos:.2e} (Muon chaos floor)", flush=True)
    c1, s1 = jax.jit(lambda c, k: env.fast.wm_updates(c, k, 4e-4))(env.carry, k_w)
    dl = rel_l2(d(c0), d(c1))
    report(f"world model: delta after {env.cfg.wm_batches} updates (bf16)", "bf16", dl, "", tol=max(scale, 2 * chaos),
           extra=f"= {dl / chaos:.2f} x the chaos floor, {dl / scale:.2f} x the bf16 error scale")
    for k in ("wm", "wm8"):
        report(f"world model: stat {k}", "bf16", float(jnp.abs(s0[k] - s1[k]).max() / jnp.abs(s0[k]).max()), "", tol=2e-2)


GROUPS = {"noise": test_noise_vmap, "perm": test_permutation_vmap, "sample": test_fast_sample, "muon": test_muon_stacked,
          "rot": test_rot, "ln": test_ln, "imag": test_imagine_grads, "policy": test_policy_update, "wm": test_wm}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="*", help=f"any of {list(GROUPS)} (default: all)")
    ap.add_argument("--horizon", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    env = Env(args.horizon)
    print(f"state ready ({time.time() - t0:.0f} s): horizon {args.horizon}, pi_batch {env.cfg.pi_batch}, enc {env.cfg.enc_hidden}, "
          f"dec {env.cfg.dec_hidden}, wm {env.cfg.wm_hidden}", flush=True)
    for name, test in GROUPS.items():
        if not args.groups or name in args.groups:
            test(env)
    n_fail = sum(not r[-1] for r in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed in {time.time() - t0:.0f} s")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
