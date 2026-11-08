"""Replay retained URR evaluation evidence; never solve or simulate a new trajectory.

Example: python tools/render_urr_benchmark.py --prepared PATH --task DLS=TASK.json
    --task Combined=TASK.json --output NEW_DIRECTORY
Target frames are the URR request, not an invented target joint configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sx_contracts.identity import canonical_json
from sx_geometry import quat_to_mat


# Generated from sx-ui categorical.ts, chart.ts light rules; ranked order preserved.
COLORS = ('#E76443', '#289FF2', '#3BE3D3', '#EBC834', '#D66AC1', '#5AEA99')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_task(path, prepared, spec):
    request = json.loads((path.parent / 'request.json').read_bytes())
    task = json.loads(path.read_bytes())
    if task['request_sha256'] != hashlib.sha256(canonical_json(request).encode()).hexdigest():
        raise ValueError('Task request binding changed')
    binding = request['environment']
    for key, member in [('scene_sha256', 'robot/scene.xml'),
                        ('body_document_sha256', 'embodiment.json'),
                        ('preparation_sha256', 'benchmark.json')]:
        if binding[key] != digest(prepared / member):
            raise ValueError(f'Prepared {member} does not match the evaluation')
    if binding['body_id'] != spec['embodiment_id']:
        raise ValueError('Evaluation body mismatch')
    q = np.asarray(task['native_q'], dtype=float)
    count = len(task['measurements'])
    if q.shape != (count, len(spec['joint_names'])) or not np.isfinite(q).all():
        raise ValueError('Invalid retained native coordinates')
    if count != binding['task']['episode_steps']:
        raise ValueError('Incomplete evaluation episode')
    for member in ('targets', 'achieved'):
        if task[member]['frame'] != 'world' or len(task[member]['frames']) != count:
            raise ValueError('Incomplete world-frame URR evidence')
    return task, request


def font(size):
    try:
        return ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', size)
    except OSError:
        return ImageFont.truetype('DejaVuSans.ttf', size)


def target_frames(scene, frame):
    """Draw actual requested coordinate frames. RGB means local XYZ, not task score."""
    for pose in frame['poses'].values():
        p = np.asarray(pose['position'])
        rotation = np.asarray(quat_to_mat(pose['quaternion_wxyz']))
        for axis, color in enumerate(((1., .2, .15, 1.), (.2, 1., .3, 1.), (.2, .5, 1., 1.))):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                               np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(color))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .005,
                                 p, p + .075 * rotation[:, axis])
            scene.ngeom += 1


def error_plot(tasks, output):
    """F2 Hairline Line: sample ticks, thin paths and genuine peak labels; no smoothing."""
    width, height = 1200, 720
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
              'viewBox="0 0 1200 720"><rect width="1200" height="720" fill="#F0F0EE"/>']
    chunks.append('<style>text{font-family:Arial,sans-serif;fill:#1F1E1C;font-size:12px}</style>')
    def text(x, y, value, size=12):
        chunks.append(f'<text x="{x}" y="{y}" style="font-size:{size}px">{html.escape(value)}</text>')
    text(35, 28, 'Recorded errors throughout one held-out task', 22)
    keys = [('position_mm', 'Hand position error · mm'), ('orientation_deg', 'Hand orientation error · degrees'),
            ('aperture_mm', 'Physical aperture error · mm')]
    for panel, (key, label) in enumerate(keys):
        base = 210 + panel * 215
        series = [(f'{name} / {side}', np.asarray([m[key][j] for m in task['measurements']]),
                   np.asarray([m['time_s'] for m in task['measurements']]))
                  for name, task in tasks for j, side in enumerate(('left', 'right'))]
        maximum = max(1e-9, max(float(v.max()) for _, v, _ in series)) * 1.12
        duration = max(float(t[-1]) for _, _, t in series)
        text(35, base - 137, label, 16)
        for fraction in (0., .5, 1.):
            y = base - fraction * 110
            chunks.append(f'<path d="M65 {y}H1160" stroke="#C0BFB7" stroke-width=".7"/>')
            text(10, y + 4, f'{fraction*maximum:.2g}')
        for t in series[0][2]:
            x = 65 + t / duration * 1095
            chunks.append(f'<path d="M{x} {base}v-4" stroke="#C0BFB7" stroke-width=".5"/>')
        for index, (name, values, times) in enumerate(series):
            color = COLORS[index % len(COLORS)]
            points = ' L '.join(f'{65+t/duration*1095:.3f} {base-v/maximum*110:.3f}' for t, v in zip(times, values, strict=True))
            chunks.append(f'<path d="M{points}" fill="none" stroke="{color}" stroke-width="1.4"/>')
            chunks.append(f'<rect x="{65+index*260}" y="{base+25}" width="14" height="3" fill="{color}"/>')
            text(85+index*260, base+31, name)
        text(65, base+17, '0 s')
        text(1120, base+17, f'{duration:g} s')
    chunks.append('</svg>')
    (output/'errors.svg').write_text(''.join(chunks))
    (output/'errors.html').write_text('<!doctype html><meta charset="utf-8"><title>URR recorded errors</title>'
        '<body style="background:#F0F0EE;margin:20px">'+''.join(chunks)+
        '<p>Every point is a retained post-step measurement. Nominal simulation; no physical qualification. '
        'F2 Hairline Line grammar, lieflat-charts; ranked sx-ui categorical colors.</p></body>')


def render(prepared, labeled_paths, output, fps=25):
    if output.exists() or 'raw' in output.resolve().parts or fps <= 0:
        raise ValueError('Use a new derived output directory and positive FPS')
    spec = json.loads((prepared/'benchmark.json').read_bytes())
    receipt_path = prepared/'robot/receipt.json'
    if digest(receipt_path) != spec['owner_receipt_sha256']:
        raise ValueError('Owner export receipt changed')
    for relative, expected in json.loads(receipt_path.read_bytes())['files']:
        member = (prepared/'robot'/relative).resolve()
        if not member.is_relative_to((prepared/'robot').resolve()) or digest(member) != expected:
            raise ValueError('Owner scene or mesh bytes changed')
    loaded = [(name, *load_task(path, prepared, spec)) for name, path in labeled_paths]
    reference = loaded[0][1]
    if any(task['seed'] != reference['seed'] or task['targets'] != reference['targets']
           for _, task, _ in loaded):
        raise ValueError('Side-by-side replays must share the exact task and target sequence')
    if not 1 <= len(loaded) <= 2:
        raise ValueError('Render one or two policies at a time for readable evidence')
    output.mkdir(parents=True)
    model = mujoco.MjModel.from_xml_path(str(prepared/'robot/scene.xml'))
    model.vis.global_.offwidth = 600
    model.vis.global_.offheight = 540
    model.vis.headlight.ambient[:] = (.4, .4, .4)
    model.vis.headlight.diffuse[:] = (.7, .7, .7)
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in spec['joint_names']]
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[1] = 1  # owner visual meshes; collision proxies remain hidden
    option.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (.12, 0, .25)
    camera.distance, camera.azimuth, camera.elevation = 1.55, 140, -24
    frames = reference['targets']['frames']
    times = np.asarray([frame['time_ns']/1e9 for frame in frames])
    if np.any(np.diff(times) <= 0):
        raise ValueError('Non-monotonic recorded times')
    indices = np.searchsorted(times, np.arange(times[0], times[-1]+.5/fps, 1/fps), side='right')-1
    indices = np.clip(indices, 0, len(times)-1)
    indices[-1] = len(times)-1  # Include the retained terminal state exactly.
    snapshots = []
    with mujoco.Renderer(model, 540, 600) as renderer, imageio.get_writer(output/'replay.mp4', fps=fps, codec='libx264', macro_block_size=1) as video:
        for display_i, i in enumerate(indices):
            canvas = Image.new('RGB', (600*(len(loaded)+1), 660), '#1C1C1A')
            draw = ImageDraw.Draw(canvas)
            for column in range(len(loaded)+1):
                if column:
                    name, task, _ = loaded[column-1]
                    q = np.asarray(task['native_q'][i])
                else:
                    name, task, q = 'URR target frames', reference, np.asarray(spec['nominal_q'])
                data.qpos[qadr] = q
                positions = dict(zip(spec['joint_names'], q, strict=True))
                for follower in spec['dependent_joints']:
                    data.qpos[model.jnt_qposadr[model.joint(follower['joint']).id]] = positions[follower['leader']]*follower['multiplier']+follower['offset']
                mujoco.mj_forward(model, data)
                option.geomgroup[1] = int(column > 0)
                renderer.update_scene(data, camera=camera, scene_option=option)
                target_frames(renderer.scene, frames[i])
                canvas.paste(Image.fromarray(renderer.render()), (600*column, 45))
                x = 600*column+15
                draw.text((x, 10), name, fill='#F0EFEB', font=font(24))
                apertures = task['achieved' if column else 'targets']['frames'][i]['apertures']
                draw.text((x, 590), 'Aperture '+', '.join(f'{k}: {v*1000:.2f} mm' for k,v in apertures.items()), fill='#F0EFEB', font=font(14))
                if column:
                    m = task['measurements'][i]
                    draw.text((x, 615), f"Position {max(m['position_mm']):.2f} mm | angle {max(m['orientation_deg']):.2f}°", fill='#F0EFEB', font=font(16))
            draw.text((15, 640), f'{times[i]:.2f} s · recorded native coordinates · nominal simulator only · RGB arrows = requested local XYZ', fill='#C0BFB7', font=font(13))
            video.append_data(np.asarray(canvas))
            if display_i in set(np.linspace(0, len(indices)-1, 3).astype(int)):
                snapshots.append(canvas.copy())
    plate = Image.new('RGB', (snapshots[0].width, 660*len(snapshots)))
    for i, frame in enumerate(snapshots):
        plate.paste(frame, (0,660*i))
    plate.save(output/'contact-sheet.jpg', quality=92)
    error_plot([(name, task) for name, task, _ in loaded], output)
    provenance = {'schema':'xmimic.urr-replay/v1', 'body_id':spec['embodiment_id'],
        'scene_sha256':digest(prepared/'robot/scene.xml'),
        'tasks':[{ 'label':name,'sha256':digest(path)} for name,path in labeled_paths],
        'renderer_sha256':digest(__file__), 'fps':fps, 'display_frames':len(indices),
        'scope':'Recorded native coordinates, no new physics or IK. Unrecorded mimic coordinates reconstructed from owner-declared coupling. Requested poses are axes only; no target robot posture invented.'}
    (output/'replay.json').write_text(canonical_json(provenance))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--task', action='append', required=True, help='LABEL=task-N.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=25)
    args = parser.parse_args()
    render(args.prepared, [(s.split('=',1)[0], Path(s.split('=',1)[1])) for s in args.task], args.output, args.fps)
