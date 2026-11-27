"""Break-off Campanile keychain on the right edge of the 203.2 x 152.4 board.
Writes keychain/geometry.json (board outline incl. tabs, mouse-bite holes, keyring hole) and
keychain/keychain_Cu.svg / keychain_Mask.svg (polygons in board mm, data-level even=feature odd=hole) for make_gerbers.
Both sides: gold everywhere (copper + mask opening), windows/clock/lines stay purple (mask over copper), 0.4 mm purple rim."""
import json, math, sys
from shapely.geometry import Polygon, Point, box, LineString, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity
OUT = sys.argv[1]
BW, BH, BR = 203.2, 152.4, 5.0
CX, Y0 = 212.7, 46.0                       # tower centre line, top of tower (board mm)
def band(y0, y1, hw): return box(CX-hw, Y0+y0, CX+hw, Y0+y1)
# ---- silhouette (y measured down from the top of the lantern) ----
spire = Polygon([(CX-1.5, Y0+3.0), (CX+1.5, Y0+3.0), (CX+5.6, Y0+17.6), (CX-5.6, Y0+17.6)])
lantern = band(1.2, 3.2, 1.1).union(Point(CX, Y0+1.2).buffer(1.1, 32))
tower = unary_union([lantern, spire,
    band(17.5, 19.6, 7.3),      # balustrade / cornice
    band(19.5, 30.2, 6.5),      # belfry
    band(30.1, 31.3, 7.0),      # cornice under the belfry
    band(31.2, 37.1, 6.5),      # clock stage
    band(37.0, 37.9, 6.8),      # string course
    band(37.8, 54.1, 6.0),      # shaft
    band(54.0, 56.6, 6.8),      # plinth step
    band(56.5, 60.0, 7.5)])     # base
tower = tower.buffer(0.15, join_style=2).buffer(-0.15, join_style=2)   # merge sliver joins
KEY = (CX, Y0 + 24.6); KEY_DRILL, KEY_PAD = 3.5, 6.0
# ---- tabs + mouse bites (both ends, so neither the card nor the keychain keeps a stub) ----
TABS = [(Y0+40.5, Y0+43.5), (Y0+48.0, Y0+51.0)]
left_shaft = CX - 6.0
tabs = [box(BW-0.6, a, left_shaft+0.6, b) for a, b in TABS]
BITE_D, BITE_P = 0.5, 0.8
bites = []
for a, b in TABS:
    n = int(round((b - a) / BITE_P)) ; ys = [a + (b-a)/2 + (k-(n-1)/2)*BITE_P for k in range(n)]
    for y in ys:
        bites.append((BW + 0.1, y)); bites.append((left_shaft - 0.1, y))
# ---- board outline (rounded rect) + tabs + tower ----
board = box(BR, 0, BW-BR, BH).union(box(0, BR, BW, BH-BR))
for cx, cy in ((BR, BR), (BW-BR, BR), (BR, BH-BR), (BW-BR, BH-BR)): board = board.union(Point(cx, cy).buffer(BR, 64))
outline = unary_union([board] + tabs + [tower]).simplify(0.005)
assert outline.geom_type == "Polygon"          # interiors = the routed slot between the two tabs
# ---- gold areas ----
cu = tower.buffer(-0.4, join_style=2)
for x, y in bites: cu = cu.difference(Point(x, y).buffer(BITE_D/2 + 0.45, 24))
details = []
def arch(xc, ytop, ybot, w):                 # arched window
    return box(xc-w/2, ytop+w/2, xc+w/2, ybot).union(Point(xc, ytop+w/2).buffer(w/2, 24))
for s in (-1, 1): details.append(arch(CX + s*4.55, Y0+21.3, Y0+28.4, 1.5))     # belfry windows either side of the keyring
cc = (CX, Y0+34.15)
details.append(Point(*cc).buffer(2.35, 48).difference(Point(*cc).buffer(1.95, 48)))   # clock face ring
details.append(LineString([cc, (cc[0], cc[1]-1.45)]).buffer(0.2))                      # minute hand
details.append(LineString([cc, (cc[0]+1.05, cc[1]+0.35)]).buffer(0.2))                 # hour hand
for x in (-3.0, 0.0, 3.0): details.append(arch(CX + x, Y0+39.6, Y0+52.6, 0.9))         # shaft slits
details.append(box(CX-6.0, Y0+54.05-0.2, CX+6.0, Y0+54.05+0.2))                        # plinth line
details.append(box(CX-6.0, Y0+19.55-0.2, CX+6.0, Y0+19.55+0.2))                        # balustrade line
details.append(box(CX-6.0, Y0+30.15-0.2, CX+6.0, Y0+30.15+0.2))                        # belfry cornice line
det = unary_union(details).intersection(cu.buffer(-0.3))
mask_open = cu.difference(det).union(Point(*KEY).buffer(KEY_PAD/2 + 0.05, 48))
def polys(g): return [g] if g.geom_type == "Polygon" else list(g.geoms)
def write_svg(g, path, note):
    items = []
    P = polys(g)
    for p in P:
        # nesting depth: how many holes of other polygons contain this one
        d = sum(1 for q in P if q is not p for h in q.interiors if Polygon(h).contains(p.representative_point()))
        items.append((2*d, list(p.exterior.coords)))
        for h in p.interiors: items.append((2*d+1, list(h.coords)))
    with open(path, "w") as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" width="222mm" height="152.4mm" viewBox="0 0 222 152.4">\n')
        f.write(f"<!-- {note}. Board coordinates seen from the FRONT; identical on both sides. data-level: even=feature, odd=hole -->\n")
        for lev, pts in sorted(items, key=lambda t: t[0]):
            f.write(f'<polygon data-level="{lev}" points="{" ".join(f"{x:.3f},{y:.3f}" for x, y in pts)}" fill="none"/>\n')
        f.write("</svg>\n")
write_svg(cu, f"{OUT}/keychain_Cu.svg", "keychain copper (F.Cu and B.Cu)")
write_svg(mask_open, f"{OUT}/keychain_Mask.svg", "keychain solder-mask openings = gold (F.Mask and B.Mask)")
json.dump(dict(outline=list(outline.exterior.coords), slots=[list(h.coords) for h in outline.interiors], bites=bites, bite_d=BITE_D, key=KEY, key_drill=KEY_DRILL, key_pad=KEY_PAD,
               tower_bounds=tower.bounds), open(f"{OUT}/geometry.json", "w"))
# preview (both sides look the same)
import numpy as np, cv2
K = 20; x0, y0, x1, y1 = tower.bounds; x0 -= 9; x1 += 2; y0 -= 2; y1 += 2
img = np.full((int((y1-y0)*K), int((x1-x0)*K), 3), 245, np.uint8)
def fill(g, col, hole=(245, 245, 245)):
    for p in polys(g):
        cv2.fillPoly(img, [np.array([[(x-x0)*K, (y-y0)*K] for x, y in p.exterior.coords], np.int32)], col)
        for h in p.interiors: cv2.fillPoly(img, [np.array([[(x-x0)*K, (y-y0)*K] for x, y in h.coords], np.int32)], hole)
fill(outline.intersection(box(x0, y0, x1, y1)), (86, 26, 58))
fill(cu, (138, 48, 96)); fill(mask_open.intersection(cu.union(Point(*KEY).buffer(KEY_PAD/2))), (104, 186, 226), (138, 48, 96))
cv2.circle(img, (int((KEY[0]-x0)*K), int((KEY[1]-y0)*K)), int(KEY_DRILL/2*K), (245, 245, 245), -1)
for x, y in bites: cv2.circle(img, (int((x-x0)*K), int((y-y0)*K)), int(BITE_D/2*K), (245, 245, 245), -1)
cv2.imwrite(f"{OUT}/preview_keychain.png", img)
print("tower bounds", [round(v, 2) for v in tower.bounds], "outline pts", len(outline.exterior.coords), "bites", len(bites), "slots", len(outline.interiors))
