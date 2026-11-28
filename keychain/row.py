"""Row of break-off keychains along the bottom edge of the card, mixing any traced shapes.
Each keychain hangs from the card by a keyring loop above its topmost point (all loop tops level); neighbours are
tied side to side where both reach out; mouse bites at both ends of every tab. Both sides: gold body, purple
(mask-over-copper) lines, gold loop with a plated hole. Lettered shapes (esb*) get mirrored back art so the
letters read correctly from behind; the others show their mirror image on the back, like a cut-out.

  python3 keychain/row.py OUTDIR "esb/esb_small.json*14"
  python3 keychain/row.py OUTDIR "esb/esb_small.json*5,dog/dog.json*3,esb/esb_small.json*6"
Shapes are relative to keychain/. Writes keychain/OUTDIR/{geometry.json, keychain_Cu.svg, keychain_Mask.svg,
preview_keychain.png}; apply_to_board.py and make_gerbers.py read that folder."""
import json, sys, os
import numpy as np
from shapely.geometry import shape, Point, box, LineString, Polygon
from shapely.ops import unary_union
from shapely import affinity
KC = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(KC, sys.argv[1]); os.makedirs(OUT, exist_ok=True)
SPEC = [(s.split("*")[0], int(s.split("*")[1]) if "*" in s else 1) for s in sys.argv[2].split(",")]
BW, BH, BR = 304.8, 228.6, 5.0                     # 12 x 9 in card
GAP, BITE_D, BITE_P, TIE_H = 2.0, 0.5, 0.8, 2.4
KEY_DRILL, KEY_PAD, LOOP_R = 3.5, 4.2, 4.2         # pad kept clear of the outline under the ring (its mask opening would cover it)
Y_TOP = BH + GAP                                    # loop tops sit GAP below the card edge

class Item:
    def __init__(self, path):
        D = json.load(open(os.path.join(KC, path)))
        self.name = os.path.basename(path).split(".")[0]
        self.sil, self.lines = shape(D["sil"]), shape(D["lines"])
        ys = self.sil.bounds[1]; top = [c for c in self.sil.exterior.coords if c[1] < ys + 0.4]
        self.key = (float(np.mean([c[0] for c in top])), ys - 2.3)
        self.loop = Point(*self.key).buffer(LOOP_R, 64)
        self.xl, self.xr = self.sil.bounds[0], self.sil.bounds[2]
        self.top = self.key[1] - LOOP_R
        self.mirror_back = self.name.startswith("esb")   # has lettering: back art mirrored so it reads right from behind
    def placed(self, dx):
        dy = Y_TOP - self.top; T = lambda g: affinity.translate(g, dx, dy)
        sil, lines = T(self.sil), T(self.lines)
        if self.mirror_back:                            # back layers are drawn as seen through the board: mirror the art
            cx = (sil.bounds[0] + sil.bounds[2]) / 2     # about the shape's centre line, keep the real outline's border
            back = affinity.scale(lines, xfact=-1, yfact=1, origin=(cx, 0)).intersection(sil)
            back = unary_union([back, sil.difference(sil.buffer(-0.9))])
        else: back = lines                               # no lettering: the back is simply the front seen from behind
        return dict(sil=sil, body=T(unary_union([self.sil, self.loop])), lines=lines, back=back, loop=T(self.loop),
                    key=(self.key[0] + dx, self.key[1] + dy), xl=self.xl + dx, xr=self.xr + dx)
cache = {}
items = [cache.setdefault(p, Item(p)) for p, n in SPEC for _ in range(n)]
total = sum(it.xr - it.xl for it in items) + GAP * (len(items) - 1)
assert total <= BW - 2 * 9.0 + 1e-6, f"row is {total:.1f} mm, only {BW - 18:.1f} mm available"
x = BW / 2 - total / 2; P = []
for it in items:
    P.append(it.placed(x - it.xl)); x += (it.xr - it.xl) + GAP

def pieces(sil, y):
    seg = sil.intersection(LineString([(-1e3, y), (1e3, y)]))
    return sorted([g for g in getattr(seg, "geoms", [seg]) if g.length > 0], key=lambda g: g.bounds[0])
def tie(A, B):
    """level where A's right-most piece and B's left-most piece are both thick and near the outer extremes"""
    y0 = max(A["sil"].bounds[1], B["sil"].bounds[1]) + 2; y1 = min(A["sil"].bounds[3], B["sil"].bounds[3]) - 2
    for tol in (0.8, 1.5, 2.5, 4.0):
        best = None
        for y in np.arange(y0, y1, 0.1):
            a, b = pieces(A["sil"], y), pieces(B["sil"], y)
            if not a or not b: continue
            ar, bl = a[-1], b[0]
            if ar.bounds[2] < A["xr"] - tol or bl.bounds[0] > B["xl"] + tol: continue
            th = min(ar.length, bl.length)
            if th >= 1.6 and (best is None or th > best[0]): best = (th, y, ar.bounds[2], bl.bounds[0])
        if best: return best
    raise SystemExit("no tie level found")
tabs, bites, ties = [], [], []
for i, p in enumerate(P):
    kx = p["key"][0]; lt = p["loop"].bounds[1]
    tabs.append(box(kx - 2.0, BH - 0.6, kx + 2.0, lt + 0.6))
    for k in range(5):
        xx = kx + (k - 2) * BITE_P; bites += [(xx, BH + 0.1), (xx, lt - 0.1)]
    if i < len(P) - 1:
        th, y, xr, xl = tie(p, P[i + 1]); ties.append(round(y - Y_TOP, 1))
        tabs.append(box(xr - 0.6, y - TIE_H / 2, xl + 0.6, y + TIE_H / 2))
        for k in range(3):
            yy = y + (k - 1) * BITE_P; bites += [(xr + 0.1, yy), (xl - 0.1, yy)]
board = box(BR, 0, BW - BR, BH).union(box(0, BR, BW, BH - BR))
for c in ((BR, BR), (BW - BR, BR), (BR, BH - BR), (BW - BR, BH - BR)): board = board.union(Point(*c).buffer(BR, 64))
outline = unary_union([board] + [p["body"] for p in P] + tabs).simplify(0.005)
assert outline.geom_type == "Polygon", outline.geom_type
keys = [p["key"] for p in P]
cu = unary_union([p["body"].buffer(-0.3, join_style=2) for p in P])   # mitre: sharp corners stay sharp
cu = cu.difference(unary_union([Point(xx, yy).buffer(BITE_D / 2 + 0.45, 24) for xx, yy in bites]))
det = unary_union([p["lines"] for p in P]).intersection(cu)
pads = unary_union([Point(*k).buffer(KEY_PAD / 2 + 0.05, 48) for k in keys])
mask_open = cu.union(pads).difference(det)
mask_open_b = cu.union(pads).difference(unary_union([p["back"] for p in P]).intersection(cu))
def polys(g): return [g] if g.geom_type == "Polygon" else [q for q in g.geoms if q.geom_type == "Polygon"]
def write_svg(g, path, note):
    its = []; PP = polys(g)
    for p in PP:
        d = sum(1 for q in PP if q is not p for h in q.interiors if Polygon(h).contains(p.representative_point()))
        its.append((2 * d, list(p.exterior.coords)))
        for h in p.interiors: its.append((2 * d + 1, list(h.coords)))
    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{BW}mm" height="{outline.bounds[3]:.1f}mm" viewBox="0 0 {BW} {outline.bounds[3]:.1f}">\n')
        f.write(f"<!-- {note}. Board coordinates seen from the FRONT; identical on both sides. data-level: even=feature, odd=hole -->\n")
        for lev, pts in sorted(its, key=lambda t: t[0]):
            f.write(f'<polygon data-level="{lev}" points="{" ".join(f"{a:.3f},{b:.3f}" for a, b in pts)}" fill="none"/>\n')
        f.write("</svg>\n")
write_svg(cu, f"{OUT}/keychain_Cu.svg", "keychain copper (F.Cu and B.Cu)")
write_svg(mask_open, f"{OUT}/keychain_Mask.svg", "keychain solder-mask openings = gold (F.Mask)")
write_svg(mask_open_b, f"{OUT}/keychain_Mask_B.svg", "keychain solder-mask openings = gold (B.Mask; lettered shapes mirrored to read right from the back)")
json.dump(dict(outline=list(outline.exterior.coords), slots=[list(h.coords) for h in outline.interiors], bites=bites,
               bite_d=BITE_D, keys=keys, key_drill=KEY_DRILL, key_pad=KEY_PAD, bounds=outline.bounds, count=len(P),
               design=sys.argv[2]), open(f"{OUT}/geometry.json", "w"))
import cv2
K = 8; x0, y0, x1, y1 = -2, BH - 8, BW + 2, outline.bounds[3] + 2
img = np.full((int((y1 - y0) * K), int((x1 - x0) * K), 3), 40, np.uint8)
def fill(g, col, hole):
    for p in sorted(polys(g), key=lambda q: -q.area):
        cv2.fillPoly(img, [np.array([[(a - x0) * K, (b - y0) * K] for a, b in p.exterior.coords], np.int32)], col)
        for h in p.interiors: cv2.fillPoly(img, [np.array([[(a - x0) * K, (b - y0) * K] for a, b in h.coords], np.int32)], hole)
fill(outline, (86, 26, 58), (40, 40, 40)); fill(cu, (138, 48, 96), (86, 26, 58)); fill(mask_open.intersection(cu), (104, 186, 226), (138, 48, 96))
for (a, b), d in [(k, KEY_DRILL) for k in keys] + [(bb, BITE_D) for bb in bites]:
    cv2.circle(img, (int((a - x0) * K), int((b - y0) * K)), max(1, int(d / 2 * K)), (40, 40, 40), -1)
cv2.imwrite(f"{OUT}/preview_keychain.png", img)
from collections import Counter
print(f"{sys.argv[1]}: {len(P)} keychains {dict(Counter(it.name for it in items))}, row {total:.1f} of {BW - 18:.1f} mm, "
      f"ties at {sorted(set(ties))} mm below the loop tops | panel {outline.bounds[2] - outline.bounds[0]:.1f} x {outline.bounds[3] - outline.bounds[1]:.1f} mm | bites {len(bites)}")
