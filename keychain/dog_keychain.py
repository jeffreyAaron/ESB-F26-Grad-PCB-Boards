"""Row of break-off dog keychains (traced from keychain/dog/sketch.png by keychain/dog/trace.py) along the bottom edge.
Each dog hangs from the card by a keyring loop above its head; neighbours are tied ear to ear. Mouse bites at both
ends of every tab. Writes the same files as keychain.py (geometry.json, keychain_Cu.svg, keychain_Mask.svg, preview),
so whichever script ran last is what make_gerbers / apply_to_board use.
Both sides: gold body, purple (mask-over-copper) bold outlines + solid eyes/nose, gold loop with a plated hole."""
import json, sys, os
import numpy as np
from shapely.geometry import shape, Point, box, LineString, Polygon
from shapely.ops import unary_union
from shapely import affinity
OUT = sys.argv[1]
D = json.load(open(os.path.join(OUT, "dog", "dog.json")))
SIL, LINES = shape(D["sil"]), shape(D["lines"])
BW, BH, BR = 203.2, 152.4, 5.0
GAP, BITE_D, BITE_P = 2.0, 0.5, 0.8
KEY_DRILL, KEY_PAD, LOOP_R = 3.5, 6.0, 4.2
KEY = (0.0, -2.6)                                  # loop centre, local coords (head top at y = 0)
loop = Point(*KEY).buffer(LOOP_R, 64)
body = unary_union([SIL, loop])
top = KEY[1] - LOOP_R; bot = SIL.bounds[3]; hw = max(-SIL.bounds[0], SIL.bounds[2])
# ear-to-ear tie: the level where the outer edge piece is thickest while the dog is near its widest
def edge_piece(y):
    seg = SIL.intersection(LineString([(-50, y), (50, y)]))
    parts = sorted([g for g in getattr(seg, "geoms", [seg]) if g.length > 0], key=lambda g: g.bounds[0])
    return (parts[0].bounds[0], parts[-1].bounds[2], parts[-1].length) if parts else None
cands = []
for y in np.arange(4, 14, 0.1):
    e = edge_piece(y)
    if e and e[1] > hw - 0.8 and e[2] >= 1.6: cands.append((e[2], y, e))
TIE_LEN, TIE_Y, (xl_local, xr_local, _) = max(cands)
TIE_H = 2.4
PITCH = 2*hw + GAP
N = int((BW - 2*9.0 - 2*hw) // PITCH) + 1
x_first = BW/2 - (N-1)*PITCH/2
Y_TOP = BH + GAP                                   # loop top sits GAP below the card edge; dog hangs upright
DY = Y_TOP - top
def place(g, cx): return affinity.translate(g, cx, DY)
def place_pt(p, cx): return (cx + p[0], DY + p[1])
dogs, lines, keys, tabs, bites = [], [], [], [], []
for i in range(N):
    cx = x_first + i*PITCH
    dogs.append(place(body, cx)); lines.append(place(LINES, cx)); keys.append(place_pt(KEY, cx))
    lt = place(loop, cx).bounds[1]                 # loop top in board coords
    tabs.append(box(cx-2.0, BH-0.6, cx+2.0, lt+0.6))
    for k in range(5):
        x = cx + (k-2)*BITE_P; bites += [(x, BH+0.1), (x, lt-0.1)]
    if i < N-1:
        y_b = DY + TIE_Y                           # tie level in board coords
        xr = cx + xr_local; xl = cx + PITCH + xl_local   # right ear edge of this dog, left ear edge of the next
        tabs.append(box(xr-0.6, y_b-TIE_H/2, xl+0.6, y_b+TIE_H/2))
        for k in range(3):
            y = y_b + (k-1)*BITE_P; bites += [(xr+0.1, y), (xl-0.1, y)]
board = box(BR, 0, BW-BR, BH).union(box(0, BR, BW, BH-BR))
for c in ((BR, BR), (BW-BR, BR), (BR, BH-BR), (BW-BR, BH-BR)): board = board.union(Point(*c).buffer(BR, 64))
outline = unary_union([board] + dogs + tabs).simplify(0.005)
assert outline.geom_type == "Polygon", outline.geom_type
cu = unary_union([d.buffer(-0.3) for d in dogs])
cu = cu.difference(unary_union([Point(x, y).buffer(BITE_D/2 + 0.45, 24) for x, y in bites]))
det = unary_union(lines).intersection(cu)
mask_open = cu.union(unary_union([Point(*k).buffer(KEY_PAD/2 + 0.05, 48) for k in keys])).difference(det)   # head outline stays purple under the ring pad
def polys(g): return [g] if g.geom_type == "Polygon" else [p for p in g.geoms if p.geom_type == "Polygon"]
def write_svg(g, path, note):
    items = []; P = polys(g)
    for p in P:
        d = sum(1 for q in P if q is not p for h in q.interiors if Polygon(h).contains(p.representative_point()))
        items.append((2*d, list(p.exterior.coords)))
        for h in p.interiors: items.append((2*d+1, list(h.coords)))
    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{BW}mm" height="{outline.bounds[3]:.1f}mm" viewBox="0 0 {BW} {outline.bounds[3]:.1f}">\n')
        f.write(f"<!-- {note}. Board coordinates seen from the FRONT; identical on both sides. data-level: even=feature, odd=hole -->\n")
        for lev, pts in sorted(items, key=lambda t: t[0]):
            f.write(f'<polygon data-level="{lev}" points="{" ".join(f"{x:.3f},{y:.3f}" for x, y in pts)}" fill="none"/>\n')
        f.write("</svg>\n")
write_svg(cu, f"{OUT}/keychain_Cu.svg", "dog keychain copper (F.Cu and B.Cu)")
write_svg(mask_open, f"{OUT}/keychain_Mask.svg", "dog keychain solder-mask openings = gold (F.Mask and B.Mask)")
json.dump(dict(outline=list(outline.exterior.coords), slots=[list(h.coords) for h in outline.interiors], bites=bites,
               bite_d=BITE_D, keys=keys, key_drill=KEY_DRILL, key_pad=KEY_PAD, bounds=outline.bounds, count=N, design="dog"),
          open(f"{OUT}/geometry.json", "w"))
import cv2
K = 8; x0, y0, x1, y1 = -2, BH - 8, BW + 2, outline.bounds[3] + 2
img = np.full((int((y1-y0)*K), int((x1-x0)*K), 3), 40, np.uint8)
def fill(g, col, hole):
    for p in sorted(polys(g), key=lambda q: -q.area):
        cv2.fillPoly(img, [np.array([[(x-x0)*K, (y-y0)*K] for x, y in p.exterior.coords], np.int32)], col)
        for h in p.interiors: cv2.fillPoly(img, [np.array([[(x-x0)*K, (y-y0)*K] for x, y in h.coords], np.int32)], hole)
fill(outline, (86, 26, 58), (40, 40, 40)); fill(cu, (138, 48, 96), (86, 26, 58)); fill(mask_open.intersection(cu), (104, 186, 226), (138, 48, 96))
for (x, y), d in [(k, KEY_DRILL) for k in keys] + [(b, BITE_D) for b in bites]:
    cv2.circle(img, (int((x-x0)*K), int((y-y0)*K)), max(1, int(d/2*K)), (40, 40, 40), -1)
cv2.imwrite(f"{OUT}/preview_keychain.png", img)
print(N, "dogs, pitch %.1f, dog %.1f x %.1f mm incl. loop, tie at local y %.1f (edge piece %.1f mm)" % (PITCH, 2*hw, bot-top, TIE_Y, TIE_LEN),
      "| board %.1f x %.1f mm | slots %d | bites %d" % (outline.bounds[2]-outline.bounds[0], outline.bounds[3]-outline.bounds[1], len(outline.interiors), len(bites)))
