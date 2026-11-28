"""ESB cube logo as clean vector geometry (instead of tracing the thin lines):
 - silhouette = the logo's outer contour simplified to straight edges (the hexagon)
 - gold = the white areas inside it (the three faces; with FILL=0 also the insides of the E / S / B), straightened, pulled back so the
   purple lines between them come out LINE mm wide; the outer border is LINE mm too
 - purple lines = silhouette minus gold, so every corner and junction is closed by construction
  python3 -I keychain/esb/vector.py HEIGHT_MM LINE_MM [NAME]   -> keychain/esb/NAME.json (default esb), preview_trace.png"""
import os, sys, json, numpy as np, cv2
from PIL import Image
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
D = os.path.dirname(os.path.abspath(__file__))
H_MM, LINE = float(sys.argv[1]), float(sys.argv[2]); NAME = sys.argv[3] if len(sys.argv) > 3 else "esb"
a = np.asarray(Image.open(f"{D}/source.png").convert("RGBA")).astype(int)
ink = ((a[..., 3] > 128) & (a[..., :3].sum(-1) < 3 * 225)).astype(np.uint8)
ink = np.pad(ink, 20)
n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
ink = np.isin(lab, [i for i in range(1, n) if st[i, 4] >= 30]).astype(np.uint8)
ys, xs = np.nonzero(ink); S = H_MM / (ys.max() - ys.min() + 1); X0 = (xs.min() + xs.max()) / 2; Y0 = ys.min()
mm = lambda c: [((x + 0.5 - X0) * S, (y - Y0) * S) for x, y in c]
# silhouette: outer contour -> straight-edged hexagon
c = max(cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
hull = cv2.convexHull(c)                                                  # the logo's corner tick marks would add nubs
eps = 2.0
while True:
    hexa = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2)
    if len(hexa) <= 6: break
    eps += 1.0
sil = Polygon(mm(hexa)).buffer(0)
# white areas inside the logo (faces minus letters, letter insides), each straightened
white = ((1 - ink) * 255).astype(np.uint8)
ff = white.copy(); cv2.floodFill(ff, None, (0, 0), 0)                    # drop the background
cs, h = cv2.findContours((ff > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
regions = []
for i, cc in enumerate(cs):
    if h[0][i][3] != -1 or cv2.contourArea(cc) < 50: continue
    holes = [Polygon(mm(cs[j].reshape(-1, 2))) for j in range(len(cs)) if h[0][j][3] == i and cv2.contourArea(cs[j]) >= 50]
    p = Polygon(mm(cc.reshape(-1, 2)))
    for q in holes: p = p.difference(q)
    regions.append(p.buffer(0).simplify(0.04))
FILL = os.environ.get("FILL", "1") == "1"                                  # 1: solid letters
src_line = 10 * S                                                          # drawn line width (~10 px)
grow = (LINE - src_line) / 2
if FILL:
    # nesting: faces (depth 0) gold, letter bodies (1) purple, openings inside letters (2) gold.
    # Only the cube edges are widened to LINE; letters and their openings keep their drawn shape.
    outer = [Polygon(r.exterior) if r.geom_type == "Polygon" else unary_union([Polygon(q.exterior) for q in r.geoms]) for r in regions]
    depth = [sum(1 for j, o in enumerate(outer) if j != i and o.contains(r.representative_point())) for i, r in enumerate(regions)]
    gold = []
    for r, o, d in zip(regions, outer, depth):
        if d == 0: gold.append(o.buffer(-grow, join_style=2, mitre_limit=3).difference(o.difference(r)))
        elif d % 2 == 0: gold.append(r)
    gold = unary_union(gold)
    print("depths", sorted(depth))
else:
    gold = unary_union([r.buffer(-grow, join_style=2, mitre_limit=3) for r in regions])
gold = gold.intersection(sil.buffer(-(LINE + 0.3), join_style=2))       # +0.3: the keychain copper is inset 0.3 mm from the edge
lines = sil.difference(gold).buffer(0)
json.dump(dict(sil=mapping(sil), lines=mapping(lines), bounds=sil.bounds), open(f"{D}/{NAME}.json", "w"))
K = 25; b = sil.bounds; img = np.full((int((b[3] - b[1] + 2) * K), int((b[2] - b[0] + 2) * K), 3), 40, np.uint8)
def fill(g, col, hole):
    for p in sorted([p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"], key=lambda q: -q.area):
        cv2.fillPoly(img, [np.array([((x - b[0] + 1) * K, (y - b[1] + 1) * K) for x, y in p.exterior.coords], np.int32)], col)
        for hh in p.interiors: cv2.fillPoly(img, [np.array([((x - b[0] + 1) * K, (y - b[1] + 1) * K) for x, y in hh.coords], np.int32)], hole)
fill(sil, (138, 48, 96), (40, 40, 40)); fill(gold, (104, 186, 226), (138, 48, 96))
cv2.imwrite(f"{D}/preview_trace.png", img)
print(f"hexagon {len(hexa)} corners, {len(regions)} white areas, {b[2] - b[0]:.1f} x {b[3] - b[1]:.1f} mm, drawn line {src_line:.2f} mm -> {LINE} mm")
