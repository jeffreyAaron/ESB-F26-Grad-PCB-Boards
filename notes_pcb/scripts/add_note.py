"""Add one more handwritten note to a board's back and repack every note at the largest common line pitch.

The existing notes are cut back out of the board's current notes_B_Silkscreen.svg (using layout.json), so the
original crops aren't needed.  The new note is a crop of a scan, with its line pitch in scan pixels.

  python3 notes_pcb/scripts/add_note.py PERSON NAME SCAN X0 Y0 X1 Y1 PITCH_PX [EXTRA_OBSTACLE x0,y0,x1,y1]
PERSON = reagan | carolyn | doruk.  Rewrites the person's notes_B_Silkscreen.svg and scripts/layout.json."""
import sys, os, json, random, re
import cv2, numpy as np
from skimage.morphology import skeletonize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracelib import write_svg

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
person, name, scan = sys.argv[1:4]
X0, Y0, X1, Y1 = map(int, sys.argv[4:8]); PITCH_PX = float(sys.argv[8])
D = {"reagan": ("notes_pcb", (145.0, 96.4, 179.5, 152.4)),
     "carolyn": ("people/carolyn", None),
     "doruk": ("people/doruk", (144.0, 96.4, 189.0, 152.4))}[person]
base = os.path.join(PROJ, D[0]); SVG = f"{base}/notes_B_Silkscreen.svg"; LAY = f"{base}/scripts/layout.json"
BW, BH, RES, MINW = 203.2, 152.4, 0.02, 0.16
W, H = int(round(BW / RES)), int(round(BH / RES))

# ---- current notes, back view, at RES ----
canvas = np.zeros((H, W), np.uint8)
polys = re.findall(r'<polygon data-level="(\d+)" points="([^"]+)"', open(SVG).read())
for lev, pts in sorted(polys, key=lambda t: int(t[0])):
    xy = np.array([tuple(map(float, q.split(","))) for q in pts.split()])
    px = np.column_stack([(BW - xy[:, 0]) / RES - 0.5, xy[:, 1] / RES - 0.5]) * 16
    cv2.fillPoly(canvas, [np.round(px).astype(np.int64)], 1 if int(lev) % 2 == 0 else 0, cv2.LINE_8, shift=4)
L = json.load(open(LAY)); P0 = L["P"]
blocks, unit = {}, {}                                  # bitmap, and (w, h) in mm per mm of line pitch
for n, (x, y, w, h) in L["place"].items():
    a, b = int(round(x / RES)), int(round(y / RES))
    blocks[n] = canvas[b:b + int(round(h / RES)), a:a + int(round(w / RES))].copy()
    unit[n] = (w / P0, h / P0)
# ---- new note from the scan ----
g = cv2.imread(scan, 0)[Y0:Y1, X0:X1]
ink = (g < 140).astype(np.uint8)
nl, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
ink[np.isin(lab, [i for i in range(1, nl) if st[i, 4] < 6])] = 0          # scan dust
ys, xs = np.nonzero(ink); ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
blocks[name] = ink; unit[name] = (ink.shape[1] / PITCH_PX, ink.shape[0] / PITCH_PX)

# ---- pack (same rules as pack.py: corners + group photo + back portrait reserved, 2.5 mm margin, 3 mm gaps) ----
M, GAP = 2.5, 3.0
OBS = [(0, 0, 10.5, 10.5), (BW - 10.5, 0, BW, 10.5), (0, BH - 10.5, 10.5, BH), (BW - 10.5, BH - 10.5, BW, BH),
       (59.6, 96.4, 143.6, BH)] + ([D[1]] if D[1] else [])
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
random.seed(1); names = list(blocks)
orders = [sorted(names, key=lambda n: -unit[n][0] * unit[n][1]), sorted(names, key=lambda n: -unit[n][1]),
          sorted(names, key=lambda n: -unit[n][0]), L["order"] + [name]]
for _ in range(4000): o = names[:]; random.shuffle(o); orders.append(o)
bestP, best = 0, None
for o in orders:
    lo, hi = max(bestP, 1.2), 4.0
    if not pack(lo + 1e-3, o): continue
    for _ in range(18):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if pack(m, o) else (lo, m)
    if lo > bestP: bestP, best = lo, (o, pack(lo, o))
P = bestP; order, place = best
print(f"{person}: line pitch {P0:.2f} -> {P:.2f} mm ({P / P0 * 100:.0f}% of before)")

# ---- render back view, min stroke, trace ----
out = np.zeros((H, W), np.uint8)
for n, (x, y, w, h) in place.items():
    b = blocks[n]; pw, ph = int(round(w / RES)), int(round(h / RES))
    r = cv2.resize(b.astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA if pw < b.shape[1] else cv2.INTER_CUBIC) > 0.45
    a, c = int(round(x / RES)), int(round(y / RES)); out[c:c + ph, a:a + pw] |= r.astype(np.uint8)
sk = skeletonize(out > 0).astype(np.uint8); rr = int(round(MINW / RES / 2))
out = np.maximum(out, cv2.dilate(sk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))))
nl, lab, st, _ = cv2.connectedComponentsWithStats(out, 8)
out[np.isin(lab, [i for i in range(1, nl) if st[i, 4] < (0.18 / RES) ** 2])] = 0
k = write_svg(out, RES, 0, 0, SVG, "handwritten notes, traced")
json.dump(dict(P=P, place=place, order=order), open(LAY, "w"), indent=1)
print(f"wrote {k} outlines to {SVG}; {name} at", [round(v, 1) for v in place[name]])
