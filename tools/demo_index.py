"""One gallery over several render_demo outputs (one per policy), with the cost ledger.

    python tools/demo_index.py <demo_dir> <gallery_subdir>... [--note "..."]

Reads <demo_dir>/<sub>/metrics.json for each policy and <demo_dir>/costs.json, writes <demo_dir>/index.html.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo_dir", type=Path)
    ap.add_argument("subdirs", nargs="+")
    ap.add_argument("--note", action="append", default=[])
    a = ap.parse_args()
    costs = json.loads((a.demo_dir / "costs.json").read_text()) if (a.demo_dir / "costs.json").exists() else None
    sections, table, n_ok, n_all = [], [], 0, 0
    for sub in a.subdirs:
        m = json.loads((a.demo_dir / sub / "metrics.json").read_text())
        cells = []
        for r in m["clips"]:
            n_all += 1
            n_ok += not r["fell"]
            badge = '<span class="bad">fell</span>' if r["fell"] else '<span class="ok">tracked</span>'
            peak = r.get("peak_speed_over_rated")
            speed = "" if peak is None else f" · peak joint speed {peak:.1f}× rated ({html.escape(r['peak_speed_joint'])})"
            cells.append(f'<figure><video src="{sub}/{html.escape(r["clip"])}.mp4" controls loop muted preload="metadata">'
                         f'</video><figcaption><b>{html.escape(r["clip"])}</b> {badge}<br>body error '
                         f'{100 * r["body_pos_err_m"]:.1f} cm · survived {100 * r["survived_frac"]:.0f}%{speed}</figcaption></figure>')
            table.append(f'<tr><td>{html.escape(r["clip"])}</td><td>{html.escape(m["run"])} @ {m["iteration"]}</td>'
                         f'<td>{"fell" if r["fell"] else "no"}</td><td>{100 * r["survived_frac"]:.0f}%</td>'
                         f'<td>{100 * r["body_pos_err_m"]:.1f}</td><td>{100 * r["body_pos_err_rootrel_m"]:.1f}</td>'
                         f'<td>{"" if peak is None else f"{peak:.2f}"}</td></tr>')
        sections.append(f'<h2>{html.escape(m["run"])} · iteration {m["iteration"]} · {m["tracked"]}/{m["total"]} tracked</h2>'
                        f'<div class="sub">{html.escape(m["label"])}</div><div class="grid">{"".join(cells)}</div>')
    spend = ""
    if costs:
        rows = "".join(f'<tr><td>{html.escape(c["name"])}</td><td>{c["gpu"]}</td><td>{c["seconds"]:.0f}</td>'
                       f'<td>{c["usd_per_h"]:.2f}</td><td>{c["usd"]:.2f}</td><td>{c["status"]}</td></tr>' for c in costs["calls"])
        spend = (f'<h2>Modal spend: ${costs["total_usd"]:.2f}</h2><table><tr><th>run</th><th>GPU</th><th>s</th><th>$/h</th>'
                 f'<th>$</th><th>status</th></tr>{rows}</table>')
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in a.note)
    (a.demo_dir / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Sentient humanoid · XMimic demo</title>
<style>body{{background:#0b0d12;color:#e8e8ea;font:15px -apple-system,Helvetica,sans-serif;margin:32px;max-width:1800px}}
h1{{font-size:30px;margin:0 0 8px}} h2{{font-size:20px;margin:34px 0 4px}} .sub{{color:#9fb3d6;margin-bottom:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:22px}}
video{{width:100%;border-radius:8px;background:#000}} figure{{margin:0}} figcaption{{margin-top:6px;color:#c8c8cc}}
.ok{{background:#1f7a4d;padding:1px 8px;border-radius:9px}} .bad{{background:#a3343a;padding:1px 8px;border-radius:9px}}
table{{border-collapse:collapse;margin:14px 0}} td,th{{padding:5px 14px;border-bottom:1px solid #262a33;text-align:left}}
li{{margin:4px 0;color:#c8c8cc}}</style>
<h1>Sentient humanoid · BONES-SEED motions tracked in simulation</h1>
<div class="sub">{n_ok} of {n_all} clips tracked to the end without falling. Left in every video: the simulated robot driven by
an XMimic policy trained on Modal; right: the kinematic reference it tracks. Simulation only (MuJoCo); no physical robot moved.</div>
<ul>{notes}</ul>
{''.join(sections)}
<h2>Per-clip metrics</h2><table><tr><th>clip</th><th>policy</th><th>fell</th><th>survived</th><th>body err (cm)</th>
<th>root-rel err (cm)</th><th>peak speed / rated</th></tr>{''.join(table)}</table>
{spend}""")
    print(f"{n_ok}/{n_all} -> {a.demo_dir / 'index.html'}")


if __name__ == "__main__":
    main()
