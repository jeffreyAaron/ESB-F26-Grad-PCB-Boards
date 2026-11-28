"""Trace clean digital line art (keychain/sus/sketch.png) into NAME.json for the keychain row: the lines are kept
as drawn (smoothed edges, thickened to at least STROKE mm), the silhouette is everything inside the outer line.
  python3 keychain/sus/trace_clean.py HEIGHT_MM STROKE_MM"""
import cv2, numpy as np, json, sys, os
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
D = os.environ.get("TRACE_DIR", os.path.dirname(os.path.abspath(__file__))); NAME = os.environ.get("NAME", "sus")
H_MM, STROKE = float(sys.argv[1]), float(sys.argv[2])
g = cv2.imread(f"{D}/sketch.png", 0); ink = (g < 128).astype(np.uint8)
ys, xs = np.nonzero(ink); ink = ink[ys.min() - 40:ys.max() + 41, xs.min() - 40:xs.max() + 41]
RES = H_MM / (ys.max() - ys.min()); H, W = ink.shape
# thicken thin lines to STROKE and smooth the jaggies
from skimage.morphology import skeletonize
SMOOTH = float(os.environ.get("SMOOTH_MM", "0.1"))                 # edge smoothing (mm)
ink = (cv2.GaussianBlur(ink.astype(np.float32), (0, 0), max(1.0, SMOOTH / RES)) > 0.5).astype(np.uint8)
sk = skeletonize(ink > 0)
dt = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
thin = (sk & (2 * dt < STROKE / RES)).astype(np.uint8)            # only strokes thinner than STROKE get thickened
k = int(round(STROKE / RES / 2))
lines = np.maximum(ink, cv2.dilate(thin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))))
lines = (cv2.GaussianBlur(lines.astype(np.float32), (0, 0), 2.0) > 0.5).astype(np.uint8)
# silhouette: flood the outside
outer = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
ff = np.pad(1 - outer, 1, constant_values=1).astype(np.uint8); fm = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
cv2.floodFill(ff, fm, (0, 0), 2); sil_px = (ff[1:-1, 1:-1] != 2).astype(np.uint8)
to_mm = lambda c: [((x - W / 2) * RES, (y - 40) * RES) for x, y in c]
def polys(m):
    cs, h = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE); out = []
    if h is None: return out
    for i, c in enumerate(cs):
        if h[0][i][3] != -1 or len(c) < 8: continue
        holes = [to_mm(cv2.approxPolyDP(cs[j], 0.7, True).reshape(-1, 2)) for j in range(len(cs)) if h[0][j][3] == i and len(cs[j]) >= 8]
        out.append(Polygon(to_mm(cv2.approxPolyDP(c, 0.7, True).reshape(-1, 2)), [q for q in holes if len(q) >= 3]).buffer(0))
    return out
sil = max(polys(sil_px), key=lambda p: p.area)
sil = Polygon(sil.exterior)
L = unary_union(polys(lines)).intersection(sil)
L = unary_union([L, sil.difference(sil.buffer(-STROKE))]).buffer(0)
json.dump(dict(sil=mapping(sil), lines=mapping(L), bounds=sil.bounds), open(f"{D}/{NAME}.json", "w"))
K = 25; b = sil.bounds; img = np.full((int((b[3] - b[1] + 2) * K), int((b[2] - b[0] + 2) * K), 3), 40, np.uint8)
def fill(gm, col, hole):
    for p in sorted([p for p in getattr(gm, "geoms", [gm]) if p.geom_type == "Polygon"], key=lambda q: -q.area):
        cv2.fillPoly(img, [np.array([((x - b[0] + 1) * K, (y - b[1] + 1) * K) for x, y in p.exterior.coords], np.int32)], col)
        for hh in p.interiors: cv2.fillPoly(img, [np.array([((x - b[0] + 1) * K, (y - b[1] + 1) * K) for x, y in hh.coords], np.int32)], hole)
fill(sil, (104, 186, 226), (40, 40, 40)); fill(L, (138, 48, 96), (104, 186, 226))
cv2.imwrite(f"{D}/preview_trace.png", img)
print("%s %.1f x %.1f mm" % (NAME, b[2] - b[0], b[3] - b[1]))
