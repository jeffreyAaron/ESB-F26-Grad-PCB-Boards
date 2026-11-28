# plain 4-tone dither of the grad portrait for the back, faded into the bare-board purple at the edges
import sys, numpy as np, cv2
from PIL import Image
OUT = sys.argv[1]
PITCH = 0.2
CROP = (0, 0, 376, 634)          # drop the sliver of another person on the right edge
W_DOTS, H_DOTS = 225, 380        # 45.0 x 76.0 mm, same height as the group photo
FADE = 9.0
DARK, MID, GOLD, SILK = (58, 26, 86), (96, 48, 138), (226, 186, 104), (238, 236, 228)
im = cv2.imread(f"{OUT}/source.png"); x0, y0, x1, y1 = CROP; im = im[y0:y1, x0:x1]
L = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (W_DOTS, H_DOTS), interpolation=cv2.INTER_AREA)
L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(L).astype(np.float32)
L = np.clip(L + 0.8 * (L - cv2.GaussianBlur(L, (0, 0), 1.2)), 0, 255) / 255
lo, hi = np.percentile(L, [1, 99.5]); L = np.clip((L - lo) / (hi - lo), 0, 1)
W, H = W_DOTS * PITCH, H_DOTS * PITCH
X, Y = np.meshgrid((np.arange(W_DOTS) + 0.5) * PITCH, (np.arange(H_DOTS) + 0.5) * PITCH)
ss = lambda t: (lambda c: c * c * (3 - 2 * c))(np.clip(t, 0, 1))
r = FADE; cx = np.clip(X, r, W - r); cy = np.clip(Y, r, H - r)
A = ss((r - np.hypot(X - cx, Y - cy)) / FADE) ** 1.2
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
a4 = diffuse(L * 3, 3, [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], 16)
a4[A < 0.02] = 0
def save(mask, name):
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).convert("1").save(f"{OUT}/{name}.png", dpi=(25.4/PITCH,)*2)
save(a4 >= 1, "fade4_F_Cu"); save(a4 == 2, "fade4_F_Mask_openings"); save(a4 == 3, "fade4_F_Silkscreen")
rgb = np.zeros((H_DOTS, W_DOTS, 3), np.uint8)
for i, c in enumerate([DARK, MID, GOLD, SILK]): rgb[a4 == i] = c
Image.fromarray(rgb).resize((W_DOTS*6, H_DOTS*6), Image.NEAREST).save(f"{OUT}/preview_fade4.png")
print(W, H)
