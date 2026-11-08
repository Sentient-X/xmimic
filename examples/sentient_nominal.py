"""Build an explicit development-only Sentient simulation and SOMA recipe.

Run from the checkout with sx-embodiments, sx-geometry and MuJoCo installed:
  python examples/sentient_nominal.py --source /private/source.bvh --output /private/derived
No download, paid job, training admission, or hardware operation occurs here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np
from sx_contracts.identity import canonical_json
from sx_embodiments import Embodiment
from sx_embodiments.embodiment import embodiment_from_definition
from sx_embodiments.known.humanoid import SENTIENT_HUMANOID_SPEC
from sx_embodiments.kinematics import Kinematics
from sx_embodiments.nominal_mujoco import CollisionMode, NominalJoint, NominalProfile, SupportContact, export_nominal_mujoco
from sx_embodiments.parts import ChainSpec, GripperSpec, Limb

from xmimic.embodiment_bundle import TrainingProfile, adapt_nominal_bundle
from xmimic.retarget_source import ParseLimits, parse_soma_bvh
from xmimic.retarget_target import Direction, IKConfig, Landmark, TargetKind, TargetProfile, prepare_target, solve_sequence
from xmimic.retarget_workflow import RetargetRecipe
from xmimic.retarget_quality import NominalQuality

# Same approved geometry; eight owner-defined pivot sites add semantic identity.
EXPECTED_BODY = 'eb249c00dcff12dc212db1e6122df0ea4f13d57f17fc90d4c4a8855051e036f5'
# Nominal anatomical correspondence: source +Z forward, +X left, +Y up.
WORLD_FROM_SOURCE = ((0.,0.,1.,0.),(1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,0.,1.))
TARGETS = (('Chest','chest'),('LeftHand','left_wrist'),('RightHand','right_wrist'),
           ('LeftFoot','left_ankle'),('RightFoot','right_ankle'))
SUPPORTS = ('left_foot2_link','right_foot2_link')


def _control(body):
    """Experimental servo choices by owning component role, never hardware drive facts."""
    components = {c.component_id:c for c in body.components}
    # kp [Nm/rad], kd [Nm s/rad], effort [Nm], armature [kg m²], action scale [rad].
    assumptions = {
        Limb.LEG:(120.,12.,80.,.05,.25), Limb.TORSO:(100.,10.,60.,.05,.20),
        Limb.ARM:(60.,6.,30.,.02,.30), Limb.HEAD:(8.,.8,3.,.005,.20),
    }
    settings = []
    for name, coordinate in zip(body.joint_names, body.state.coordinates, strict=True):
        component = components[coordinate.instance]
        if isinstance(component.part, GripperSpec):
            values = (10.,1.,5.,.005,.10)
        elif isinstance(component.part, ChainSpec):
            values = assumptions[component.limb]
        else:
            raise ValueError('This explicit experiment has no control assumption for a component')
        kp,kd,effort,armature,scale = values
        # The nominal initial angle is nearest bounded zero, not a calibrated Ready pose.
        seed = 0.
        default = max(coordinate.axis.bounds.lower, min(coordinate.axis.bounds.upper, seed))
        settings.append(NominalJoint(name,kp,kd,effort,armature,default,scale))
    return tuple(settings)


def _sole_height(model, data, link):
    """Lowest collision vertex of the exact designated foot at the nominal pose."""
    body_id = model.body(link).id
    lows = []
    for geom in np.flatnonzero(model.geom_bodyid == body_id):
        if not model.geom_contype[geom] and not model.geom_conaffinity[geom]:
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError('This Sentient support derivation expects its canonical mesh collision')
        mesh = model.geom_dataid[geom]
        start,count = model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start+count]
        world = vertices @ data.geom_xmat[geom].reshape(3,3).T + data.geom_xpos[geom]
        lows.append(float(world[:,2].min()))
    if not lows:
        raise ValueError('Designated support has no source collision geometry')
    low = min(lows)
    return float(data.xpos[body_id,2]-low), low


def build_example(source: bytes, output: Path, *, body: Embodiment | None = None):
    body = embodiment_from_definition(SENTIENT_HUMANOID_SPEC) if body is None else body
    if str(body.id) != EXPECTED_BODY:
        raise ValueError('Development body changed; review the explicit experimental profile')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, mode=0o700)
    joints = _control(body)
    kin = Kinematics.of(body.urdf_bytes)
    root = next(iter(kin.links-kin.parent_joint.keys()))
    tracked = tuple(dict.fromkeys([root]+[kin.joints[n].child for n in body.joint_names]))
    assumptions = (
        'Development nominal simulation only; no hardware drive, physical qualification or standing claim.',
        'All joint PD gains, effort limits, armatures and action scales are experimental component-class choices.',
        'Initial simulation pose is nearest bounded zero; IK seed bends knees +0.15 and hip pitch -0.075 rad, not calibrated Ready postures.',
        'waist_rod_joint, neck_yaw_joint and head_joint are fixed at zero as explicit nominal mechanism/head projections; no neck actuator facts are invented.',
        'foot2 links are designated support links; sole offsets use their lowest source collision vertex at nominal pose.',
        'Source-derived URDF inertials are preserved, including their existing estimates; they are not measured dynamics.',
        'Self collisions are disabled explicitly in this nominal ground-only simulation; no self-collision qualification.',
        'Ground is a rigid plane with nominal friction (1,0.005,0.0001); no contact calibration is claimed.',
    )
    preliminary = NominalProfile(str(body.id),joints,(('waist_rod_joint',0.),('neck_yaw_joint',0.),('head_joint',0.)),tracked,
                  tuple(SupportContact(link,0.) for link in SUPPORTS),(0.,0.,0.),(1.,0.,0.,0.),
                  .005,(0.,0.,-9.81),(1.,.005,.0001),assumptions, CollisionMode.GROUND_ONLY)
    with tempfile.TemporaryDirectory(prefix='geometry-',dir=output) as scratch:
        bundle = Path(scratch)/'bundle'
        export_nominal_mujoco(body,preliminary,bundle)
        model = mujoco.MjModel.from_xml_path(str(bundle/'scene.xml'))
        data = mujoco.MjData(model)
        for j in joints:
            data.qpos[model.jnt_qposadr[model.joint(j.name).id]] = j.default_position
        mujoco.mj_forward(model,data)
        support = [_sole_height(model,data,link) for link in SUPPORTS]
        contacts = tuple(SupportContact(link,max(0.,height)) for link,(height,_) in zip(SUPPORTS,support,strict=True))
        root_height = -min(low for _,low in support)+.01  # explicit 1 cm initial clearance
    nominal = replace(preliminary,contacts=contacts,root_position=(0.,0.,root_height),
                      assumptions=(*assumptions,'Initial pelvis height gives the lowest designated sole 0.01 m ground clearance.'))
    receipt = export_nominal_mujoco(body,nominal,output/'robot')
    training = TrainingProfile(root,((-.005,.005),(-.005,.005),(-.005,.005)),(.9,1.1),.002,
                    (.005,.005,.005),(.005,.005,.005),(.01,.01,.01),(.01,.01,.01),.005,.01,256,128)
    adapt_nominal_bundle(output/'robot',training)
    clip = parse_soma_bvh(source,world_from_source=WORLD_FROM_SOURCE)
    source_root = clip.joint_index('Hips')
    # SOMA's XYZ channels replace OFFSET values; the current first-frame pose
    # is the explicitly selected calibration posture, not a fabricated rest pose.
    positions = {j.name:j.default_position for j in joints}
    passive = (('waist_rod_joint',0.),('neck_yaw_joint',0.),('head_joint',0.))
    positions.update(passive)
    anatomy = (
        (('LeftArm','LeftForeArm','LeftHand'),('left_shoulder','left_elbow','left_wrist')),
        (('RightArm','RightForeArm','RightHand'),('right_shoulder','right_elbow','right_wrist')),
        (('LeftLeg','LeftShin','LeftFoot'),('left_hip','left_knee','left_ankle')),
        (('RightLeg','RightShin','RightFoot'),('right_hip','right_knee','right_ankle')),
    )
    directions = []
    leg_ratios = []
    for source_nodes, robot_links in anatomy:
        directions.extend(Direction(a,b,c,d,kind=TargetKind.SITE) for a,b,c,d in zip(
            source_nodes[:-1],source_nodes[1:],robot_links[:-1],robot_links[1:],strict=True))
        if source_nodes[-1].endswith('Foot'):
            # The actual hip pitch pivot, not a downstream yaw link, defines
            # the robot leg chain used solely for root-displacement scaling.
            robot_points = [np.asarray(body.site(link).transform(kin,root,positions)).reshape(4,4)[:3,3] for link in robot_links]
            lengths = [np.linalg.norm(b-a) for a,b in zip(robot_points[:-1],robot_points[1:],strict=True)]
            human_lengths = [np.linalg.norm(clip.world_positions[0,clip.joint_index(b)]-clip.world_positions[0,clip.joint_index(a)])
                             for a,b in zip(source_nodes[:-1],source_nodes[1:],strict=True)]
            if min(human_lengths) <= 1e-9:
                raise ValueError('First-frame source leg has a degenerate segment')
            leg_ratios.append(float(sum(lengths)/sum(human_lengths)))
    root_scale = float(sum(leg_ratios)/len(leg_ratios))
    human_R = clip.world_rotations[0,source_root]
    root_offset_world = np.array([0.,0.,root_height-clip.world_positions[0,source_root,2]*root_scale])
    root_offset = human_R.T @ root_offset_world
    held = tuple((j.name,j.default_position) for j in joints if 'finger_joint' in j.name)
    initial = np.asarray([.15 if 'knee_joint' in j.name else (-.075 if 'hip_pitch_joint' in j.name else j.default_position) for j in joints])
    solver = IKConfig(max_iterations=60)
    # Fit limb directions before assigning distal frame orientations. Binding
    # a nonneutral human foot to a neutral robot foot conflicts with the observed
    # leg pose. Only terminal rolls are held for this static fit, since they do
    # not change these endpoint positions. Two identical frames are calibration
    # only, never synthesized training evidence.
    calibration_targets = (('Chest','chest'),)
    calibration_profile = TargetProfile(
        tuple(Landmark(name,link,TargetKind.SITE,orientation=True,position=False) for name,link in calibration_targets),
        held+tuple((j.name,j.default_position) for j in joints if 'wrist_roll_joint' in j.name or 'foot2_joint' in j.name),passive,tuple(directions))
    calibration_target = prepare_target(body,calibration_profile,expected_embodiment_id=EXPECTED_BODY)
    direction_values = {}
    for mapping in directions:
        vector = clip.world_positions[0,clip.joint_index(mapping.source_child)]-clip.world_positions[0,clip.joint_index(mapping.source_parent)]
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            raise ValueError('First-frame source has a degenerate direction')
        direction_values[mapping.key] = np.repeat((vector/length)[None],2,axis=0)
    calibration_orientations = {name:np.repeat(np.asarray(body.site(link).transform(kin,root,positions)).reshape(4,4)[None,:3,:3],2,axis=0)
                                for name,link in calibration_targets}
    calibration_p = clip.world_positions[0,source_root]*root_scale+human_R@root_offset
    calibration_initial = initial.copy()
    for name,value in calibration_profile.holds:
        calibration_initial[body.joint_names.index(name)] = value
    fitted = solve_sequence(calibration_target,{},root_p=np.repeat(calibration_p[None],2,axis=0),
        root_quat=np.array([[1.,0.,0.,0.]]*2),fps=50.,initial_q=calibration_initial,
        orientations=calibration_orientations,directions=direction_values,config=replace(solver,max_iterations=120))
    fit_report = fitted.provenance()
    fit_report['source_sha256'] = hashlib.sha256(source).hexdigest()
    fit_report['calibration_scope'] = 'First source frame repeated twice only for a static directional fit; no source frames removed or synthesized for training.'
    (output/'first-pose-fit.json').write_text(canonical_json(fit_report))
    np.savez(output/'first-pose-fit.npz',q=fitted.q,root_p=fitted.root_p,root_quat=fitted.root_quat,
             joint_names=np.asarray(body.joint_names),embodiment_id=np.asarray(str(body.id)),
             scope=np.asarray('static hand calibration only; not a training motion'))
    if not fitted.feasible:
        raise ValueError('First-pose directional calibration failed; diagnostics retained in first-pose-fit.json')
    fitted_positions = dict(zip(body.joint_names,fitted.q[0],strict=True))
    fitted_positions.update(passive)
    orientation_offsets = []
    for name,link in TARGETS:
        # Bind observed orientation changes to the fitted first pose, never an
        # unrelated neutral posture. These are nominal frame correspondences,
        # not measured tool calibration or a flat-sole/contact constraint.
        q = fitted_positions
        rotation = clip.world_rotations[0,clip.joint_index(name)].T @ np.asarray(body.site(link).transform(kin,root,q)).reshape(4,4)[:3,:3]
        orientation_offsets.append((name,tuple(tuple(row) for row in rotation)))
    final_initial = fitted.q[0].copy()
    target = TargetProfile(tuple(Landmark(name,link,TargetKind.SITE,orientation=True,position=False) for name,link in TARGETS),
                           held,passive,tuple(directions))
    recipe = RetargetRecipe(target,WORLD_FROM_SOURCE,'Hips',
                           tuple(tuple(row) for row in human_R.T),
                           tuple(float(v) for v in root_offset),root_scale,(),tuple(orientation_offsets),
                           tuple((name,float(q)) for name,q in zip(body.joint_names,final_initial,strict=True)),solver,(),
                           NominalQuality(.03,.01,SUPPORTS,.002,.05,4.,6.,.001))
    (output/'recipe.json').write_text(canonical_json(asdict(recipe)))
    (output/'training-profile.json').write_text(canonical_json(asdict(training)))
    (output/'nominal-profile.json').write_text(canonical_json(asdict(nominal)))
    calibration = {'source_sha256':hashlib.sha256(source).hexdigest(),'body_id':str(body.id),
        'urdf_sha256':receipt.urdf_sha256,'nominal_profile_sha256':nominal.sha256,
        'source_frame_time_literal':clip.frame_time_literal,'source_frames':clip.frame_count,
        'source_axis_assumption':'SOMA +Z forward, +X left, +Y up mapped to robot FLU',
        'source_pose_calibration_frame':0,'root_orientation_assumption':'first observed Hips orientation corresponds to upright forward robot root','root_height_m':root_height,
        'support_heights_m':{c.link:c.height_m for c in contacts},
        'anatomical_scaling':'Eight source unit bone directions constrained against owner-defined physical sites through shared URR MotionBinding; no fixed root anchors or endpoint targets. Root translation scale uses hip-pitch-to-knee-to-foot lengths.',
        'hand_orientation_calibration':'First-frame eight-direction IK with chest nominal orientation and terminal wrist/ankle rolls held only for calibration; source hand and foot rotations offset to fitted sites. No measured palm/tool or foot-contact claim.',
        'first_pose_fit_sha256':hashlib.sha256((output/'first-pose-fit.json').read_bytes()).hexdigest(),
        'first_pose_direction_max_rad':float(fitted.direction_residual_rad.max()),
        'retarget_scope':'Eight limb directions plus chest/hand/foot orientations; torso/waist free, fingers held, passive neck/head fixed. Diagnostic only until whole-pose and visual review; no collision/contact/balance qualification',
        'training_admitted':False}
    (output/'calibration.json').write_text(canonical_json(calibration))
    return body, recipe, calibration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--embodiment-document',type=Path,help='Exact portable development body; avoids registry-version assumptions')
    args = parser.parse_args()
    if args.source.stat().st_size > ParseLimits().max_bytes:
        raise ValueError('Source exceeds the bounded BVH parser limit')
    body = None
    if args.embodiment_document is not None:
        if args.embodiment_document.stat().st_size > 8*1024**2:
            raise ValueError('Embodiment document exceeds its bound')
        body = Embodiment.from_json(args.embodiment_document.read_text())
    _,_,report = build_example(args.source.read_bytes(),args.output,body=body)
    print(json.dumps(report,sort_keys=True))
