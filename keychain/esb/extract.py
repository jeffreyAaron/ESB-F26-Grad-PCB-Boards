"""keychain/esb/source.png (ESB cube logo, thin lavender lines) -> sketch.png: black lines on white, upscaled."""
import os, numpy as np, cv2
from PIL import Image
D = os.path.dirname(os.path.abspath(__file__))
im = Image.open(f"{D}/source.png").convert("RGBA"); a = np.asarray(im).astype(int)
r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
ink = (al > 128) & ((r + g + b) < 3 * 225)             # coloured line pixels (background is white or transparent)
n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
ink = np.isin(lab, [i for i in range(1, n) if st[i, 4] >= 30])
sk = np.pad(np.where(ink, 0, 255).astype(np.uint8), 60, constant_values=255)   # logo touches the image edges
sk = cv2.resize(sk, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(f"{D}/sketch.png", np.where(sk > 128, 255, 0).astype(np.uint8))
print("ink px", int(ink.sum()), "pieces", n - 1, "sketch", sk.shape)
