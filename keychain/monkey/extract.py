"""keychain/monkey/source.png -> sketch.png: the monkey alone (stray marks around it dropped), rotated upright
(laptop edges level), black on white, upscaled for trace_clean.py."""
import cv2, numpy as np, os
D = os.path.dirname(os.path.abspath(__file__))
g = cv2.imread(f"{D}/source.png", 0); ink = (g < 128).astype(np.uint8)
n, lab, st, cen = cv2.connectedComponentsWithStats(ink, 8)
main = 1 + int(np.argmax(st[1:, 4]))
body = (lab == main).astype(np.uint8)
ff = np.pad(1 - body, 1, constant_values=1); fm = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
cv2.floodFill(ff, fm, (0, 0), 2); inside = ff[1:-1, 1:-1] != 2
keep = [i for i in range(1, n) if i == main or inside[int(cen[i][1]), int(cen[i][0])]]
ink = np.isin(lab, keep).astype(np.uint8)
# straighten: the laptop (largest white area enclosed by the body, after the face) has its edges level
w = ((1 - ink) & inside).astype(np.uint8)
n2, l2, s2, c2 = cv2.connectedComponentsWithStats(w, 4)
lap = sorted(range(1, n2), key=lambda i: -s2[i, 4])[1]          # [0] = face, [1] = laptop
(cx, cy), (rw, rh), ang = cv2.minAreaRect(np.argwhere(l2 == lap)[:, ::-1].astype(np.float32))
a = ang if rw >= rh else ang - 90                                  # angle of the laptop's long edge
a = (a + 45) % 90 - 45
# of the four level orientations, take the one with the face (largest enclosed white area) on top
face = sorted(range(1, n2), key=lambda i: -s2[i, 4])[0]
fy, fx = np.argwhere(l2 == face).mean(0); by, bx = np.argwhere(ink > 0).mean(0)
def up(ang):                                                       # face height above body after rotating ang deg CCW
    r = np.radians(ang); dx, dy = fx - bx, fy - by
    return -(-np.sin(r) * dx + np.cos(r) * dy)
a = max((a + k * 90 for k in range(4)), key=up)
a = (a + 180) % 360 - 180
ANGLE = float(os.environ.get("ANGLE", a))
pad = 200; big = np.pad(ink, pad)
M = cv2.getRotationMatrix2D((big.shape[1] / 2, big.shape[0] / 2), ANGLE, 1.0)
rot = cv2.warpAffine(big.astype(np.float32), M, big.shape[::-1], flags=cv2.INTER_CUBIC) > 0.5
ys, xs = np.nonzero(rot); rot = rot[ys.min() - 20:ys.max() + 21, xs.min() - 20:xs.max() + 21]
sk = np.where(rot, 0, 255).astype(np.uint8)
sk = cv2.resize(sk, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(f"{D}/sketch.png", np.where(sk > 128, 255, 0).astype(np.uint8))
print(f"kept {len(keep)} of {n - 1} ink pieces; rotated {ANGLE:.1f} deg; sketch {sk.shape[1]} x {sk.shape[0]} px")
