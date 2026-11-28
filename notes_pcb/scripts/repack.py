"""Repack a board's existing notes onto a (new) card size, scaling every note's traced outlines as vectors
(no re-rasterising), at the largest common line pitch that fits.  Card size and keep-out areas: board_cfg.py.

  python3 notes_pcb/scripts/repack.py PERSON OLD_BW
OLD_BW = board width the current notes_B_Silkscreen.svg was made for (its outlines are mirrored about it)."""
import sys, os, json, random, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board_cfg as C

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
person, BW0 = sys.argv[1], float(sys.argv[2])
base = os.path.join(PROJ, C.DIRS[person]); SVG = f"{base}/notes_B_Silkscreen.svg"; LAY = f"{base}/scripts/layout.json"
L = json.load(open(LAY)); P0 = L["place"]; Pold = L["P"]
polys = [(int(l), np.array([tuple(map(float, q.split(","))) for q in p.split()]))
         for l, p in re.findall(r'<polygon data-level="(\d+)" points="([^"]+)"', open(SVG).read())]
polys = [(l, np.column_stack([BW0 - a[:, 0], a[:, 1]])) for l, a in polys]          # back view
owner = {n: [] for n in P0}; lost = 0
for lev, a in polys:
    cx, cy = a[:, 0].mean(), a[:, 1].mean()
    hit = [n for n, (x, y, w, h) in P0.items() if x - 0.5 <= cx <= x + w + 0.5 and y - 0.5 <= cy <= y + h + 0.5]
    if hit: owner[hit[0]].append((lev, a))
    else: lost += 1
assert lost == 0, f"{lost} outlines outside every note"
unit = {n: (w / Pold, h / Pold) for n, (x, y, w, h) in P0.items()}

OBS = C.obstacles(person); BW, BH, M, GAP = C.BW, C.BH, C.M, C.GAP
def cut(free, o):
    ox0, oy0, ox1, oy1 = o; out = []
    for (x0, y0, x1, y1) in free:
        if ox0 >= x1 or ox1 <= x0 or oy0 >= y1 or oy1 <= y0: out.append((x0, y0, x1, y1)); continue
        if ox0 > x0: out.append((x0, y0, ox0, y1))
        if ox1 < x1: out.append((ox1, y0, x1, y1))
        if oy0 > y0: out.append((x0, y0, x1, oy0))
        if oy1 < y1: out.append((x0, oy1, x1, y1))
    out = [r for r in out if r[2] - r[0] > 0.5 and r[3] - r[1] > 0.5]
    return [r for r in out if not any(q != r and q[0] <= r[0] and q[1] <= r[1] and q[2] >= r[2] and q[3] >= r[3] for q in out)]
def pack(P, order):
    g = GAP / 2; free = [(M - g, M - g, BW - M + g, BH - M + g)]
    for o in OBS: free = cut(free, (o[0] - g, o[1] - g, o[2] + g, o[3] + g))
    place = {}
    for n in order:
        w, h = unit[n][0] * P + GAP, unit[n][1] * P + GAP; best = None
        for (x0, y0, x1, y1) in free:
            if x1 - x0 >= w and y1 - y0 >= h and (best is None or (y0, x0) < best[0]): best = ((y0, x0), x0, y0)
        if best is None: return None
        _, x, y = best; place[n] = (x + g, y + g, w - GAP, h - GAP); free = cut(free, (x, y, x + w, y + h))
    return place
random.seed(1); names = list(unit)
orders = [sorted(names, key=lambda n: -unit[n][0] * unit[n][1]), sorted(names, key=lambda n: -unit[n][1]),
          sorted(names, key=lambda n: -unit[n][0]), L["order"]]
for _ in range(4000): o = names[:]; random.shuffle(o); orders.append(o)
bestP, best = 0, None
for o in orders:
    lo, hi = max(bestP, 1.2), 8.0
    if not pack(lo + 1e-3, o): continue
    for _ in range(20):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if pack(m, o) else (lo, m)
    if lo > bestP: bestP, best = lo, (o, pack(lo, o))
P = bestP; order, place = best
print(f"{person}: line pitch {Pold:.2f} -> {P:.2f} mm (letters {P / Pold:.2f}x bigger)")

out = []
for n, (x, y, w, h) in place.items():
    ox, oy, ow, oh = P0[n]; k = w / ow
    for lev, a in owner[n]:
        b = np.column_stack([x + (a[:, 0] - ox) * k, y + (a[:, 1] - oy) * k])
        out.append((lev, np.column_stack([BW - b[:, 0], b[:, 1]])))                 # back to board coords (mirrored)
out.sort(key=lambda t: t[0])
with open(SVG, "w") as f:
    f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{BW}mm" height="{BH}mm" viewBox="0 0 {BW} {BH}">\n')
    f.write('<!-- handwritten notes, traced. Board coordinates as seen from the FRONT (bottom-side art is mirrored). data-level: even=feature, odd=hole -->\n')
    f.write('<path fill="black" fill-rule="evenodd" d="' + " ".join("M" + " ".join(f"{px:.3f},{py:.3f}" for px, py in a) + "Z" for _, a in out) + '"/>\n')
    for lev, a in out: f.write(f'<polygon data-level="{lev}" points="{" ".join(f"{px:.3f},{py:.3f}" for px, py in a)}" fill="none"/>\n')
    f.write("</svg>\n")
json.dump(dict(P=P, place=place, order=order), open(LAY, "w"), indent=1)
print(f"wrote {len(out)} outlines; notes now at", {n: [round(v, 1) for v in r] for n, r in place.items()})
