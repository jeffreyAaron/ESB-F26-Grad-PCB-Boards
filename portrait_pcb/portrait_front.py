"""Front portrait box: 4-tone dither of the cut-out portrait on the field purple, inside a rounded box
with a thin bare-board (dark) outline. Writes front_F_Cu / front_F_Mask_openings / front_F_Silkscreen
(0.2 mm dots, black = feature) plus front_preview.png.  Box placement on the board: BOX below."""
import sys, numpy as np, cv2
from PIL import Image
OUT = sys.argv[1]; DST = sys.argv[2] if len(sys.argv) > 2 else OUT
PITCH = 0.2
BOX = (12.0, 63.0, 109.5, 160.5)    # board mm (x0, y0, x1, y1) - rounded box, outer edge
R, RING = 6.0, 0.0                 # no border: the portrait sits straight on the field purple
DARK, MID, GOLD, SILK = (58, 26, 86), (96, 48, 138), (226, 186, 104), (238, 236, 228)
W = int(round((BOX[2]-BOX[0])/PITCH)); H = int(round((BOX[3]-BOX[1])/PITCH))
inner = int(round(RING/PITCH)); n = W - 2*inner                      # portrait square inside the ring
im = cv2.imread(f"{OUT}/source.jpg"); m = cv2.imread(f"{OUT}/cutout_mask.png", 0)
Lfull = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
yy = np.arange(800)[:, None]
m = np.maximum(m, np.where((yy > 600) & (Lfull < 110), 255, 0).astype(np.uint8))   # keep suit shoulders
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
L = cv2.resize(Lfull, (n, n), interpolation=cv2.INTER_AREA)
import os
# CLAHE clip, midtone gamma, sharpen (original look; "1.5,0.75,0.9" gives lighter, more open eyes)
CLIP, GAMMA, SHARP = (float(v) for v in os.environ.get("PF", "2.0,1.0,0.6").split(","))
L = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(4, 4)).apply(L).astype(np.float32) / 255
L = np.clip(L + SHARP * (L - cv2.GaussianBlur(L, (0, 0), 1.2)), 0, 1)
M = cv2.resize(m, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 > 0.5
lo, hi = np.percentile(L[M], [2, 99]); L = np.clip((L - lo) / (hi - lo), 0, 1) ** GAMMA
def diffuse(t, nlev, kernel, div):
    f = t.copy(); o = np.full(t.shape, 1, np.uint8); Hh, Ww = t.shape
    for y in range(Hh):
        d = 1 if y % 2 == 0 else -1
        for x in (range(Ww) if d == 1 else range(Ww-1, -1, -1)):
            if not M[y, x]: continue                                   # background stays field purple
            q = int(np.clip(round(f[y, x]), 0, nlev)); o[y, x] = q; e = (f[y, x] - q) / div
            for dx, dy, k in kernel:
                X, Y = x + dx*d, y + dy
                if 0 <= X < Ww and Y < Hh and M[Y, X]: f[Y, X] += e * k
    return o
lv = diffuse(L * 3, 3, [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], 16)
full = np.full((H, W), 1, np.uint8)
full[inner:inner+n, inner:inner+n] = lv[:H-2*inner, :]
# rounded-box shape: outside -> 255 (not ours), ring -> 0 (dark)
def rr(w, h, r):
    k = np.zeros((h, w), np.uint8); r = int(round(r/PITCH))
    cv2.rectangle(k, (r, 0), (w-1-r, h-1), 1, -1); cv2.rectangle(k, (0, r), (w-1, h-1-r), 1, -1)
    for cx, cy in ((r, r), (w-1-r, r), (r, h-1-r), (w-1-r, h-1-r)): cv2.circle(k, (cx, cy), r, 1, -1)
    return k > 0
outer = rr(W, H, R)
inn = np.zeros_like(outer); inn[inner:H-inner, inner:W-inner] = rr(W-2*inner, H-2*inner, R-RING)
full[~outer] = 255
def save(mask, name):
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).convert("1").save(f"{DST}/{name}.png", dpi=(25.4/PITCH,)*2)
save(full == 0, "front_F_Cu_clear")   # dark dots: cut out of the field copper (clear polarity)
save(full == 2, "front_F_Mask_openings"); save(full == 3, "front_F_Silkscreen")
rgb = np.zeros((H, W, 3), np.uint8); rgb[:] = MID
for i, c in enumerate([DARK, MID, GOLD, SILK]): rgb[full == i] = c
Image.fromarray(rgb).resize((W*4, H*4), Image.NEAREST).save(f"{DST}/front_preview.png")
print(W, H, n)
