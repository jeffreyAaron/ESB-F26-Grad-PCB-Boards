"""A row of break-off Campanile keychains along the bottom edge of the 203.2 x 152.4 board.
Each tower hangs base-first from the card on a tab, and neighbours are tied together at the shaft, all with
mouse-bite perforations at both ends of every tab so nothing keeps a stub.
Writes keychain/geometry.json (board outline incl. tabs + enclosed slots, mouse bites, keyring holes) and
keychain/keychain_Cu.svg / keychain_Mask.svg (board mm, data-level even=feature odd=hole) for make_gerbers.
Both sides: gold everywhere (copper + mask opening), windows/clock/lines stay purple, 0.4 mm purple rim."""
import json, sys
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import unary_union
from shapely import affinity
OUT = sys.argv[1]
BW, BH, BR = 203.2, 152.4, 5.0
TH, PITCH, GAP = 60.0, 17.0, 2.0                 # tower height, centre spacing, card-to-base gap
BITE_D, BITE_P = 0.5, 0.8
KEY_DRILL, KEY_PAD = 3.5, 6.0
# ---------------- one tower, local coords: x about the centre line, y down from the lantern top ----------------
def band(y0, y1, hw): return box(-hw, y0, hw, y1)
spire = Polygon([(-1.5, 3.0), (1.5, 3.0), (5.6, 17.6), (-5.6, 17.6)])
lantern = band(1.2, 3.2, 1.1).union(Point(0, 1.2).buffer(1.1, 32))
T_SIL = unary_union([lantern, spire, band(17.5, 19.6, 7.3), band(19.5, 30.2, 6.5), band(30.1, 31.3, 7.0),
                     band(31.2, 37.1, 6.5), band(37.0, 37.9, 6.8), band(37.8, 54.1, 6.0), band(54.0, 56.6, 6.8),
                     band(56.5, 60.0, 7.5)]).buffer(0.15, join_style=2).buffer(-0.15, join_style=2)
T_KEY = (0.0, 24.6)
def arch(xc, ytop, ybot, w): return box(xc-w/2, ytop+w/2, xc+w/2, ybot).union(Point(xc, ytop+w/2).buffer(w/2, 24))
cc = (0.0, 34.15)
T_DET = unary_union([arch(-4.55, 21.3, 28.4, 1.5), arch(4.55, 21.3, 28.4, 1.5),                   # belfry windows
                     Point(*cc).buffer(2.35, 48).difference(Point(*cc).buffer(1.95, 48)),          # clock face
                     LineString([cc, (0, cc[1]-1.45)]).buffer(0.2), LineString([cc, (1.05, cc[1]+0.35)]).buffer(0.2),
                     arch(-3.0, 39.6, 52.6, 0.9), arch(0.0, 39.6, 52.6, 0.9), arch(3.0, 39.6, 52.6, 0.9),   # shaft slits
                     box(-6, 53.85, 6, 54.25), box(-6, 19.35, 6, 19.75), box(-6, 29.95, 6, 30.35)])      # ledges
# ---------------- place the row: base towards the card, spire pointing away ----------------
Y_BASE = BH + GAP
N = int((BW - 2*9.0 - 15.0) // PITCH) + 1                       # keep clear of the rounded corners / corner holes
x_first = BW/2 - (N-1)*PITCH/2
def place(g, cx):                                               # rotate 180 deg, base at the top, then move
    return affinity.translate(affinity.rotate(g, 180, origin=(0, TH/2)), cx, Y_BASE - 0.0)
def place_pt(p, cx): return (cx - p[0], Y_BASE + TH - p[1])
towers, dets, keys, tabs, bites = [], [], [], [], []
for i in range(N):
    cx = x_first + i*PITCH
    towers.append(place(T_SIL, cx)); dets.append(place(T_DET, cx)); keys.append(place_pt(T_KEY, cx))
    # base tab to the card
    tabs.append(box(cx-2.0, BH-0.6, cx+2.0, Y_BASE+0.6))
    for k in range(5):
        x = cx + (k-2)*BITE_P; bites += [(x, BH+0.1), (x, Y_BASE-0.1)]
    # shaft tab to the next tower (shaft is local y 37.8-54.1 -> placed y = Y_BASE + 60 - y)
    if i < N-1:
        ya, yb = Y_BASE + 12.0, Y_BASE + 14.5; xl, xr = cx + 6.0, cx + PITCH - 6.0
        tabs.append(box(xl-0.6, ya, xr+0.6, yb))
        for k in range(3):
            y = (ya+yb)/2 + (k-1)*BITE_P; bites += [(xl+0.1, y), (xr-0.1, y)]
tower_all = unary_union(towers)
board = box(BR, 0, BW-BR, BH).union(box(0, BR, BW, BH-BR))
for c in ((BR, BR), (BW-BR, BR), (BR, BH-BR), (BW-BR, BH-BR)): board = board.union(Point(*c).buffer(BR, 64))
outline = unary_union([board, tower_all] + tabs).simplify(0.005)
assert outline.geom_type == "Polygon"          # interiors = routed slots enclosed by the tabs
cu = unary_union([t.buffer(-0.4, join_style=2) for t in towers])
cu = cu.difference(unary_union([Point(x, y).buffer(BITE_D/2 + 0.45, 24) for x, y in bites]))
det = unary_union(dets).intersection(cu.buffer(-0.3))
mask_open = cu.difference(det).union(unary_union([Point(*k).buffer(KEY_PAD/2 + 0.05, 48) for k in keys]))
def polys(g): return [g] if g.geom_type == "Polygon" else list(g.geoms)
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
write_svg(cu, f"{OUT}/keychain_Cu.svg", "keychain copper (F.Cu and B.Cu)")
write_svg(mask_open, f"{OUT}/keychain_Mask.svg", "keychain solder-mask openings = gold (F.Mask and B.Mask)")
json.dump(dict(outline=list(outline.exterior.coords), slots=[list(h.coords) for h in outline.interiors], bites=bites,
               bite_d=BITE_D, keys=keys, key_drill=KEY_DRILL, key_pad=KEY_PAD, bounds=outline.bounds, count=N),
          open(f"{OUT}/geometry.json", "w"))
# close-up preview of the row (both sides look the same)
import numpy as np, cv2
K = 8; x0, y0, x1, y1 = -2, BH - 8, BW + 2, outline.bounds[3] + 2
img = np.full((int((y1-y0)*K), int((x1-x0)*K), 3), 40, np.uint8)
def fill(g, col, hole=(40, 40, 40)):
    for p in polys(g):
        cv2.fillPoly(img, [np.array([[(x-x0)*K, (y-y0)*K] for x, y in p.exterior.coords], np.int32)], col)
        for h in p.interiors: cv2.fillPoly(img, [np.array([[(x-x0)*K, (y-y0)*K] for x, y in h.coords], np.int32)], hole)
fill(outline, (86, 26, 58)); fill(cu, (138, 48, 96)); fill(mask_open.intersection(cu), (104, 186, 226), (138, 48, 96))
for (x, y), d in [(k, KEY_DRILL) for k in keys] + [(b, BITE_D) for b in bites]:
    cv2.circle(img, (int((x-x0)*K), int((y-y0)*K)), max(1, int(d/2*K)), (40, 40, 40), -1)
cv2.imwrite(f"{OUT}/preview_keychain.png", img)
print(N, "keychains, pitch", PITCH, "board now %.1f x %.1f mm" % (outline.bounds[2]-outline.bounds[0], outline.bounds[3]-outline.bounds[1]),
      "| slots", len(outline.interiors), "| bites", len(bites))
