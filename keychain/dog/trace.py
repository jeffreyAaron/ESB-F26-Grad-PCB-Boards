"""Trace keychain/dog/sketch.png (your own drawing) into keychain/dog/dog.json, cartoon-cleaned:
 - every pen stroke becomes one smooth spline (fits out the pen jitter, keeps the lumps and the off-centre smile),
   drawn at an even STROKE width with round ends; stray tick marks shorter than SPUR mm are dropped
 - eyes / nose keep their own hand-drawn shapes, just with smoothed edges
 - silhouette = smoothed outer shape; the outer outline is a STROKE-wide ring inside it
 Usage: trace.py HEIGHT_MM STROKE_MM [SMOOTH_MM] [SPUR_MM]"""
import cv2, numpy as np, json, sys, os
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon, LineString, mapping
from shapely.ops import unary_union
D = os.path.dirname(os.path.abspath(__file__))
H_MM, STROKE = float(sys.argv[1]), float(sys.argv[2])
SMOOTH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.12     # mm of jitter to fit out (small = keeps lumps)
SPUR = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
g = cv2.imread(f"{D}/sketch.png", 0); ink = (g < 128).astype(np.uint8)
ys, xs = np.nonzero(ink); ink = ink[ys.min()-20:ys.max()+21, xs.min()-20:xs.max()+21]
RES = H_MM / (ys.max() - ys.min())
H, W = ink.shape
to_mm = lambda p: ((p[0] - W/2) * RES, (p[1] - 20) * RES)       # x centred, head top at y = 0
def smooth_closed(pts, s_mm):
    pts = np.asarray(pts, float); pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]]
    if len(pts) < 8: return pts
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * s_mm**2, per=1)
    u = np.linspace(0, 1, max(40, int(LineString(pts).length / 0.08)))
    return np.column_stack(splev(u, tck))
def smooth_open(pts, s_mm):
    pts = np.asarray(pts, float)
    if len(pts) < 5: return pts
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * s_mm**2, k=3 if len(pts) > 6 else 1)
    u = np.linspace(0, 1, max(10, int(LineString(pts).length / 0.08)))
    return np.column_stack(splev(u, tck))
# ---- solid blobs (pupils, nose) ----
blobs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
n, lab, st, _ = cv2.connectedComponentsWithStats(blobs)
blob_ids = [i for i in range(1, n) if st[i, 4] > 2500]          # pupils + nose only (thick double-stroked rim spots are smaller)
blob_polys = []
for i in blob_ids:
    m = cv2.dilate((lab == i).astype(np.uint8), np.ones((7, 7), np.uint8)) & ink
    c = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea).reshape(-1, 2)
    blob_polys.append(Polygon(smooth_closed([to_mm(p) for p in c], SMOOTH)).buffer(0))
blob_px = np.zeros_like(ink)
for i in blob_ids: blob_px |= (lab == i)
# ---- strokes: skeleton graph -> paths ----
CLOSE = int(os.environ.get("CLOSE", "15"))       # px: merges places where the pen went over a line twice
closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE, CLOSE)))
closed[cv2.dilate(blob_px, np.ones((9, 9), np.uint8)) > 0] = 0            # blobs are handled separately
sk = skeletonize(closed > 0).astype(np.uint8)
P = set(map(tuple, np.argwhere(sk)))                                      # (row, col)
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
nbrs = lambda p: [(p[0]+a, p[1]+b) for a, b in NB if (p[0]+a, p[1]+b) in P]
deg = {p: len(nbrs(p)) for p in P}
nodes = {p for p in P if deg[p] != 2}
seen, paths = set(), []
def walk(a, b):
    path = [a, b]; seen.add(frozenset((a, b)))
    while path[-1] not in nodes:
        cur, prev = path[-1], path[-2]
        nxt = [q for q in nbrs(cur) if q != prev and frozenset((cur, q)) not in seen]
        if not nxt: break
        q = nxt[0]; seen.add(frozenset((cur, q))); path.append(q)
        if q == path[0]: break
    return path
for a in nodes:
    for b in nbrs(a):
        if frozenset((a, b)) not in seen: paths.append(walk(a, b))
for a in P:                                                               # pure loops (no nodes)
    for b in nbrs(a):
        if frozenset((a, b)) not in seen: paths.append(walk(a, b))
strokes = []
for p in paths:
    pts = [to_mm((c, r)) for r, c in p]; L = LineString(pts).length if len(pts) > 1 else 0
    free = deg[p[0]] == 1 or deg[p[-1]] == 1
    if free and L < SPUR: continue                                        # stray ticks (connectors between junctions are kept)
    if len(pts) < 2: continue
    closed_loop = p[0] == p[-1] and len(p) > 8
    sm = smooth_closed(pts[:-1], SMOOTH) if closed_loop else (smooth_open(pts, SMOOTH) if L > 0.4 else np.asarray(pts))
    if closed_loop: sm = np.vstack([sm, sm[:1]])
    strokes.append(LineString(sm).buffer(STROKE/2, cap_style=1, join_style=1, quad_segs=8))
lines = unary_union(strokes + blob_polys)
# ---- silhouette: fill from outside, smooth the outer contour ----
outer = cv2.morphologyEx(cv2.dilate(ink, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
ff = np.pad(1 - outer, 1, constant_values=1).astype(np.uint8); fm = np.zeros((ff.shape[0]+2, ff.shape[1]+2), np.uint8)
cv2.floodFill(ff, fm, (0, 0), 2); sil_px = (ff[1:-1, 1:-1] != 2).astype(np.uint8)
c = max(cv2.findContours(sil_px, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea).reshape(-1, 2)
sil = Polygon(smooth_closed([to_mm(p) for p in c], SMOOTH)).buffer(0)
sil = unary_union([sil, lines]).buffer(0.05).buffer(-0.05)
sil = Polygon(max(getattr(sil, "geoms", [sil]), key=lambda q: q.area).exterior)
ring = sil.difference(sil.buffer(-STROKE))
lines = unary_union([lines.intersection(sil), ring]).buffer(0)
# ---- vector export of the exact traced geometry (same outlines that go into the gerbers) ----
def path_d(gm):
    out = []
    for p in ([gm] if gm.geom_type == "Polygon" else [q for q in gm.geoms if q.geom_type == "Polygon"]):
        for r in [p.exterior, *p.interiors]:
            c = list(r.coords)[:-1]
            out.append("M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in c) + "Z")
    return " ".join(out)
bx = sil.bounds; pad = 1
with open(f"{D}/dog.svg", "w") as f:
    f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{bx[2]-bx[0]+2*pad:.2f}mm" height="{bx[3]-bx[1]+2*pad:.2f}mm" '
            f'viewBox="{bx[0]-pad:.3f} {bx[1]-pad:.3f} {bx[2]-bx[0]+2*pad:.3f} {bx[3]-bx[1]+2*pad:.3f}">\n')
    f.write(f'<path d="{path_d(sil)}" fill="#E2BA68"/>\n<path d="{path_d(lines)}" fill="#60308A" fill-rule="evenodd"/>\n</svg>\n')
json.dump(dict(sil=mapping(sil), lines=mapping(lines), bounds=sil.bounds), open(f"{D}/dog.json", "w"))
K = 30; b = sil.bounds; img = np.full((int((b[3]-b[1])*K)+20, int((b[2]-b[0])*K)+20, 3), 40, np.uint8)
def fill(gm, col, hole):
    for p in sorted([gm] if gm.geom_type == "Polygon" else [q for q in gm.geoms if q.geom_type == "Polygon"], key=lambda q: -q.area):
        cv2.fillPoly(img, [np.array([[(x-b[0])*K+10, (y-b[1])*K+10] for x, y in p.exterior.coords], np.int32)], col, cv2.LINE_AA)
        for h in p.interiors: cv2.fillPoly(img, [np.array([[(x-b[0])*K+10, (y-b[1])*K+10] for x, y in h.coords], np.int32)], hole, cv2.LINE_AA)
fill(sil, (104, 186, 226), (104, 186, 226)); fill(lines, (138, 48, 96), (104, 186, 226))
cv2.imwrite(f"{D}/preview_trace.png", img)
print("dog %.1f x %.1f mm, stroke %.2f, %d strokes, %d blobs" % (b[2]-b[0], b[3]-b[1], STROKE, len(strokes), len(blob_polys)))
