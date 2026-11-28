"""Front Campanile art (4-tone dither at 0.2 mm dots), rebuilt at any card scale.

The original generator was lost, so the look is learned from the original 8 x 6 in art
(board_203x152/art_*.png): the photo crop is found in source.jpg, the tone curve is fitted by histogram
matching, and the left-edge fade into the field purple is fitted column by column. The photo is then
re-dithered at SCALE x the original size, so the detail comes from the photo, not from enlarging dots.

  python3 campanile_pcb/campanile_art.py SCALE OUTDIR HOLE_X,HOLE_Y [HOLE_X,HOLE_Y ...]
HOLE = mounting-hole centres in art-local mm (copper cleared around them like the original)."""
import sys, os, numpy as np, cv2
from PIL import Image
D = os.path.dirname(os.path.abspath(__file__))
SCALE, OUT = float(sys.argv[1]), sys.argv[2]
HOLES = [tuple(map(float, h.split(","))) for h in sys.argv[3:]]
PITCH = 0.2
CLIP = float(os.environ.get('CLIP', '2.0'))
CROP = (175, 389, 2443, 3110)                       # source px; found by template matching against the original art
DARK, MID, GOLD, SILK = (58, 26, 86), (96, 48, 138), (226, 186, 104), (238, 236, 228)
os.makedirs(OUT, exist_ok=True)

def load(n): return np.asarray(Image.open(f"{D}/board_203x152/art_{n}.png").convert("L")) < 128
old = load("F_Cu").astype(np.float32) + load("F_Mask_openings") + 2 * load("F_Silkscreen")
oh, ow = old.shape
src = cv2.imread(f"{D}/source.jpg", 0)[CROP[1]:CROP[3], CROP[0]:CROP[2]]

def prep(n_w, n_h):
    g = cv2.resize(src, (n_w, n_h), interpolation=cv2.INTER_AREA)
    tile = max(2, int(round(6 * n_w / 635)))           # CLAHE tiles keep the same size in mm at any scale
    g = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(tile, tile)).apply(g).astype(np.float32)
    return np.clip(g + 0.6 * (g - cv2.GaussianBlur(g, (0, 0), 1.2)), 0, 255)

# ---- learn tone curve + fade from the original art ----
g0 = prep(ow, oh)
tb = cv2.GaussianBlur(old, (0, 0), 2.0); gb = cv2.GaussianBlur(g0, (0, 0), 2.0)
R = slice(int(ow * 0.55), ow)                       # right part: no fade there
qs = np.linspace(0, 1, 201)
gq, tq = np.quantile(gb[:, R], qs), np.quantile(tb[:, R], qs)
gq = np.maximum.accumulate(gq + np.arange(len(gq)) * 1e-6)
f = lambda g: np.interp(g, gq, tq)
fb = f(gb)
A = np.ones(ow)
for x in range(ow):                                 # old - 1 = A(x) * (f(src) - 1)
    a, b = fb[:, x] - 1, tb[:, x] - 1
    A[x] = np.clip((a * b).sum() / max((a * a).sum(), 1e-6), 0, 1)
A = cv2.GaussianBlur(A.reshape(1, -1).astype(np.float32), (0, 0), 12, borderType=cv2.BORDER_REPLICATE).ravel()
A = np.clip(A / np.median(A[int(ow * 0.6):]), 0, 1)
A = np.maximum.accumulate(A)                        # the fade only ever increases to the right

# ---- new art ----
nw, nh = int(round(ow * SCALE)), int(round(oh * SCALE))
g = prep(nw, nh)
An = np.interp((np.arange(nw) + 0.5) / nw, (np.arange(ow) + 0.5) / ow, A)
T = np.clip(1 + An[None, :] * (f(g) - 1), 0, 3)
def diffuse(t):
    f_ = t.copy(); o = np.zeros(t.shape, np.uint8); H, W = t.shape
    for y in range(H):
        d = 1 if y % 2 == 0 else -1; row = f_[y]; nxt = f_[y + 1] if y + 1 < H else None
        for x in (range(W) if d == 1 else range(W - 1, -1, -1)):
            q = int(min(3, max(0, round(row[x])))); o[y, x] = q; e = (row[x] - q) / 16
            if 0 <= x + d < W: row[x + d] += 7 * e
            if nxt is not None:
                if 0 <= x - d < W: nxt[x - d] += 3 * e
                nxt[x] += 5 * e
                if 0 <= x + d < W: nxt[x + d] += e
    return o
a4 = diffuse(T)
yy, xx = np.mgrid[0:nh, 0:nw]
for hx, hy in HOLES:                                # same treatment as the original: clear 3.8 mm, solid ring to 4.4 mm
    d = np.hypot((xx + 0.5) * PITCH - hx, (yy + 0.5) * PITCH - hy)
    a4[(d >= 3.8) & (d < 4.4)] = 1; a4[d < 3.8] = 0
def save(mask, name):
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).convert("1").save(f"{OUT}/{name}.png", dpi=(25.4 / PITCH,) * 2)
save(a4 >= 1, "art_F_Cu"); save(a4 == 2, "art_F_Mask_openings"); save(a4 == 3, "art_F_Silkscreen")
rgb = np.zeros((nh, nw, 3), np.uint8)
for i, c in enumerate([DARK, MID, GOLD, SILK]): rgb[a4 == i] = c
Image.fromarray(rgb).resize((nw * 2, nh * 2), Image.NEAREST).save(f"{OUT}/art_reference.png")
print(f"{nw} x {nh} dots = {nw * PITCH:.1f} x {nh * PITCH:.1f} mm; levels", np.bincount(a4.ravel(), minlength=4),
      "| original", np.bincount(old.astype(int).ravel(), minlength=4) * SCALE ** 2)
