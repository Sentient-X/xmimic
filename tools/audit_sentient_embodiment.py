"""Inspect the installed Sentient embodiment sources without running training or hardware.

Run with Python containing sx-embodiments and MuJoCo:
    python tools/audit_sentient_embodiment.py

The JSON is a source audit, not a training-readiness or physical-qualification receipt.
Assets resolve through sx-embodiments' verified closure. No robot geometry is copied here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import mujoco

from sx_embodiments import Embodiment, development_embodiments, embodiments, materialized
from sx_embodiments.assets import AssetFormat
from sx_embodiments.layout import Bounds


@dataclass(frozen=True)
class BodyAudit:
    name: str
    identity: str
    joint_names: tuple[str, ...]
    arms: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class SimulationAudit:
    asset_uri: str
    asset_sha256: str
    nq: int
    nv: int
    actuators: int
    missing_coordinates: tuple[str, ...]
    extra_coordinates: tuple[str, ...]
    limit_mismatches: tuple[str, ...]


def body_audit(body: Embodiment) -> BodyAudit:
    return BodyAudit(
        name=str(body.name),
        identity=str(body.id),
        joint_names=body.joint_names,
        arms=tuple(
            (side, tuple(name for name in body.joint_names if name.startswith(f"{side}_")
                         and name in {c.joint_name for c in body.state.coordinates
                                      if str(c.instance) == f"{side}_arm"}))
            for side in ("left", "right")
        ),
    )


def simulation_audits(body: Embodiment) -> tuple[SimulationAudit, ...]:
    reports = []
    canonical = {c.joint_name: c.axis for c in body.state.coordinates}
    for source in body.assets:
        asset = source.asset
        if asset.format is not AssetFormat.MJCF:
            continue
        model = mujoco.MjModel.from_xml_path(str(materialized(body, asset)))
        joints = {
            model.joint(i).name: i
            for i in range(model.njnt)
            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        }
        mismatches = []
        for name in canonical.keys() & joints.keys():
            bounds = canonical[name].bounds
            index = joints[name]
            if isinstance(bounds, Bounds) and (
                not model.jnt_limited[index]
                or abs(float(model.jnt_range[index, 0]) - bounds.lower) > 1e-6
                or abs(float(model.jnt_range[index, 1]) - bounds.upper) > 1e-6
            ):
                mismatches.append(name)
        reports.append(SimulationAudit(
            asset_uri=asset.uri,
            asset_sha256=str(asset.content.sha256),
            nq=model.nq,
            nv=model.nv,
            actuators=model.nu,
            missing_coordinates=tuple(sorted(canonical.keys() - joints.keys())),
            extra_coordinates=tuple(sorted(joints.keys() - canonical.keys())),
            limit_mismatches=tuple(sorted(mismatches)),
        ))
    return tuple(reports)


def main() -> None:
    full = embodiments["sentient-humanoid"]
    upper = development_embodiments["sentient-static-upper-body"]
    full_axes = {c.joint_name: c.axis for c in full.state.coordinates}
    upper_axes = {c.joint_name: c.axis for c in upper.state.coordinates}
    print(json.dumps({
        "scope": "source audit only; does not establish training or hardware readiness",
        "full": asdict(body_audit(full)),
        "corrected_upper": asdict(body_audit(upper)),
        "shared_names_with_different_axis_facts": sorted(
            name for name in full_axes.keys() & upper_axes.keys()
            if full_axes[name] != upper_axes[name]
        ),
        "full_simulation_assets": [asdict(report) for report in simulation_audits(full)],
    }, indent=2))


if __name__ == "__main__":
    main()
