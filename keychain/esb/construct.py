"""ESB cube logo built from exact geometry (no tracing): a true isometric hexagon, three rhombus faces, block
letters E (top), S (left), B (right) drawn on a grid and mapped onto each face so they sit exactly in the middle.

Everything is in real mm, measured perpendicular to the edges:
  LINE    purple line width: the cube outline (as seen after the 0.3 mm copper inset) and the three inner edges
  STROKE  letter stroke as a fraction of the letter's side; GAPF the slots between strokes (3 strokes + 2 slots = 1)
  margin  gold band between the lines and the letter = the letter's own slot width, the same on all four sides
  python3 keychain/esb/construct.py WIDTH_MM [NAME]   -> keychain/esb/NAME.json (default esb_small), preview_construct.png"""
import os, sys, json, math, numpy as np, cv2
from shapely.geometry import Polygon, LineString, mapping, box
from shapely.ops import unary_union
from shapely import affinity
D = os.path.dirname(os.path.abspath(__file__))
WIDTH = float(sys.argv[1]); NAME = sys.argv[2] if len(sys.argv) > 2 else "esb_small"
LINE = float(os.environ.get("LINE", "0.6"))
STROKE = float(os.environ.get("STROKE", "0.18")); GAPF = (1 - 3 * STROKE) / 2
ROUND = float(os.environ.get("ROUND", "0.25"))      # letter corner radius (mm); cube corners stay sharp
INSET = 0.3                                          # keychain copper is inset 0.3 mm from the cut edge (keychain/row.py)
c30, s30, s60 = math.cos(math.radians(30)), 0.5, math.sin(math.radians(60))
s = WIDTH / (2 * c30); h = s * c30                   # cube edge, half width
T, UR, LR, B, LL, UL, C = (0, 0), (h, s / 2), (h, 1.5 * s), (0, 2 * s), (-h, 1.5 * s), (-h, s / 2), (0, s)
sil = Polygon([T, UR, LR, B, LL, UL])
# faces: (corners in order origin, u-corner, opposite, v-corner), letter
faces = {"E": (C, UR, T, UL), "S": (LL, B, C, UL), "B": (B, LR, UR, C)}
# purple: outer border (LINE visible after the copper inset) + the three inner edges (LINE wide, centred)
border = sil.difference(sil.buffer(-(LINE + INSET), join_style=2))
edges = unary_union([LineString([C, p]).buffer(LINE / 2, cap_style=2, join_style=2) for p in (UR, B, UL)])
lines_frame = unary_union([border, edges])
def letter(ch, t=STROKE, g=GAPF):
    """block letter on the unit square, x right, y up; 3 horizontal strokes, 2 slots"""
    y0, y1, y2 = 0, t + g, 2 * (t + g)                # bottom of the three bars
    bars = [box(0, y, 1, y + t) for y in (y0, y1, y2)]
    if ch == "E": parts = bars + [box(0, 0, t, 1)]
    elif ch == "S": parts = bars + [box(0, y1, t, y2 + t), box(1 - t, y0, 1, y1 + t)]
    elif ch == "B": parts = bars + [box(0, 0, t, 1), box(1 - t, 0, 1, 1)]
    return unary_union(parts)
purple_letters, gold_faces = [], []
for ch, (O, U, X, V) in faces.items():
    face = Polygon([O, U, X, V])
    gold = face.difference(lines_frame).intersection(sil.buffer(-(LINE + INSET), join_style=2))
    gold = max(getattr(gold, "geoms", [gold]), key=lambda q: q.area)          # visible gold rhombus of this face
    # the gold rhombus's corners matching O, U, V of the face (it is a smaller rhombus with the same directions)
    gc = list(gold.exterior.coords)[:-1]
    near = lambda P: min(gc, key=lambda q: math.hypot(q[0] - P[0], q[1] - P[1]))
    side = math.hypot(near(U)[0] - near(O)[0], near(U)[1] - near(O)[1])      # gold rhombus side (both sides equal)
    # letter rhombus: inset the gold rhombus by the slot width on all sides; slot width depends on the letter size,
    # so solve: a = side - 2 * margin / sin60, margin = GAPF * a * sin60  ->  a = side / (1 + 2 * GAPF)
    a = side / (1 + 2 * GAPF); margin = GAPF * a * s60
    L = gold.buffer(-margin, join_style=2)
    lc = list(L.exterior.coords)[:-1]
    nearL = lambda P: min(lc, key=lambda q: math.hypot(q[0] - P[0], q[1] - P[1]))
    o, u, v = np.array(nearL(O)), np.array(nearL(U)), np.array(nearL(V))
    M = [u[0] - o[0], v[0] - o[0], u[1] - o[1], v[1] - o[1], o[0], o[1]]       # unit square -> letter rhombus
    Lr = affinity.affine_transform(letter(ch), M)
    if ROUND > 0:                                    # soften every letter corner by ROUND mm (outer and inner corners)
        Lr = Lr.buffer(-ROUND, join_style=1).buffer(ROUND, join_style=1)
        Lr = Lr.buffer(ROUND, join_style=1).buffer(-ROUND, join_style=1)
    purple_letters.append(Lr)
    gold_faces.append(gold)
    print(f"{ch}: gold face side {side:.2f} mm, letter side {a:.2f} mm, margin = slot {margin:.2f} mm, stroke {STROKE * a * s60:.2f} mm")
lines = unary_union([lines_frame] + purple_letters)
gold_area = sil.difference(lines).area; purp_area = lines.intersection(sil.buffer(-INSET, join_style=2)).area
print(f"cube {WIDTH:.2f} x {2 * s:.2f} mm, line {LINE} mm | visible purple {purp_area / (purp_area + gold_area) * 100:.0f}% / gold {gold_area / (purp_area + gold_area) * 100:.0f}%")
json.dump(dict(sil=mapping(sil), lines=mapping(lines), bounds=sil.bounds), open(f"{D}/{NAME}.json", "w"))
K = 40; b = sil.bounds; img = np.full((int((b[3] - b[1] + 2) * K), int((b[2] - b[0] + 2) * K), 3), 40, np.uint8)
def fill(g, col, hole):
    for p in sorted([p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"], key=lambda q: -q.area):
        cv2.fillPoly(img, [np.array([((x - b[0] + 1) * K, (y - b[1] + 1) * K) for x, y in p.exterior.coords], np.int32)], col, cv2.LINE_AA)
        for hh in p.interiors: cv2.fillPoly(img, [np.array([((x - b[0] + 1) * K, (y - b[1] + 1) * K) for x, y in hh.coords], np.int32)], hole, cv2.LINE_AA)
cu = sil.buffer(-INSET, join_style=2)
fill(cu, (104, 186, 226), (40, 40, 40)); fill(lines.intersection(cu), (138, 48, 96), (104, 186, 226))
cv2.imwrite(f"{D}/preview_construct.png", img)
