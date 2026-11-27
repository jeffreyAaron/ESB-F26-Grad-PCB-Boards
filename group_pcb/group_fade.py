# plain 4-tone dither of the group photo, faded to the bare-board purple at the edges and near the NFC coil
import sys, numpy as np, cv2
from PIL import Image
OUT = sys.argv[1]
W_MM, PITCH = 80.0, 0.2
OX, OY = 61.6, 98.9          # back-view top-left (mm): centred on the 203.2 mm board
FADE = 10.0                   # mm of fade on each edge
COIL = (133.2, 48.0, 173.2, 88.0)    # NFC coil outer edge, back-view mm (board x 30-70, y 48-88, mirrored)
CROP = (0, 690, 1500, 1640)
DARK, MID, GOLD, SILK = (58, 26, 86), (96, 48, 138), (226, 186, 104), (238, 236, 228)
im = cv2.imread(f"{OUT}/source.jpg"); x0, y0, x1, y1 = CROP; im = im[y0:y1, x0:x1]
Lfull = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
nw = int(W_MM / PITCH); nh = int(round(nw * (y1 - y0) / (x1 - x0)))
L = cv2.resize(Lfull, (nw, nh), interpolation=cv2.INTER_AREA)
L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(L).astype(np.float32)
L = np.clip(L + 0.8 * (L - cv2.GaussianBlur(L, (0, 0), 1.2)), 0, 255) / 255
lo, hi = np.percentile(L, [1, 99.5]); L = np.clip((L - lo) / (hi - lo), 0, 1)
# fade mask
xs = OX + (np.arange(nw) + 0.5) * PITCH; ys = OY + (np.arange(nh) + 0.5) * PITCH
X, Y = np.meshgrid(xs, ys)
ss = lambda t: (lambda c: c * c * (3 - 2 * c))(np.clip(t, 0, 1))
de = np.minimum.reduce([X - OX, OX + W_MM - X, Y - OY, OY + nh * PITCH - Y])
# soften corners: use distance to an inset rounded rectangle
r = FADE; cx = np.clip(X, OX + r, OX + W_MM - r); cy = np.clip(Y, OY + r, OY + nh*PITCH - r)
dcorner = r - np.hypot(X - cx, Y - cy)
a_edge = ss(np.minimum(de, dcorner) / FADE)
dx = np.maximum.reduce([COIL[0] - X, np.zeros_like(X), X - COIL[2]]); dy = np.maximum.reduce([COIL[1] - Y, np.zeros_like(Y), Y - COIL[3]])
a_coil = ss((np.hypot(dx, dy) - 2.0) / 6.0)          # safety net: no copper within 2 mm of the coil (photo is ~11 mm away now)
A = (a_edge * a_coil) ** 1.2
L = L * A
def diffuse(t, nlev, kernel, div):
    f = t.copy(); o = np.zeros(t.shape, np.uint8); H, W = t.shape
    for y in range(H):
        d = 1 if y % 2 == 0 else -1
        for x in (range(W) if d == 1 else range(W-1, -1, -1)):
            q = int(np.clip(round(f[y, x]), 0, nlev)); o[y, x] = q; e = (f[y, x] - q) / div
            for ddx, ddy, k in kernel:
                XX, YY = x + ddx*d, y + ddy
                if 0 <= XX < W and YY < H: f[YY, XX] += e * k
    return o
FS = [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)]
a4 = diffuse(L * 3, 3, FS, 16)
a4[A < 0.02] = 0
def save(mask, name):
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).convert("1").save(f"{OUT}/{name}.png", dpi=(25.4/PITCH,)*2)
save(a4 >= 1, "fade4_F_Cu"); save(a4 == 2, "fade4_F_Mask_openings"); save(a4 == 3, "fade4_F_Silkscreen")
rgb = np.zeros((nh, nw, 3), np.uint8)
for i, c in enumerate([DARK, MID, GOLD, SILK]): rgb[a4 == i] = c
Image.fromarray(rgb).resize((nw*6, nh*6), Image.NEAREST).save(f"{OUT}/preview_fade4.png")
print(nw, nh, "copper within 2 mm of coil:", int(((a4 >= 1) & (a_coil == 0)).sum()))
