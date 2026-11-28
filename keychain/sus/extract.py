"""keychain/sus/source.png -> sketch.png: the black outlines plus the pink blush strokes, without the text."""
import cv2, numpy as np, os
D = os.path.dirname(os.path.abspath(__file__))
im = cv2.imread(f"{D}/source.png"); b, g, r = [im[:, :, i].astype(int) for i in range(3)]
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); H, W = gray.shape
black = (gray < 90) & (np.abs(r - b) < 40)
black[:, :55] = 0; black[:, 530:] = 0; black[477:] = 0          # dark side bars, text below
blush = (r > 200) & (g > 75) & (g < 200) & (b > 60) & (b < 200) & (r - g > 60)
blush[477:] = 0
n, lab, st, _ = cv2.connectedComponentsWithStats(blush.astype(np.uint8), 8)
blush = np.isin(lab, [i for i in range(1, n) if st[i, 4] >= 15])
sk = np.full((H, W), 255, np.uint8); sk[black] = 0; sk[blush] = 0
sk = cv2.resize(sk, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)     # same pixel scale as the dog sketch for trace.py
cv2.imwrite(f"{D}/sketch.png", np.where(sk > 128, 255, 0).astype(np.uint8))
print("outline px", black.sum(), "blush px", blush.sum())
