# GrabCut cut-out of the person -> cutout_mask.png (255 = person)
import cv2, numpy as np, sys
D = sys.argv[1]
im = cv2.imread(f"{D}/source.jpg"); H, W = im.shape[:2]
m = np.full((H, W), cv2.GC_BGD, np.uint8)
m[190:, 30:770] = cv2.GC_PR_BGD
cv2.ellipse(m, (410, 400), (150, 200), 0, 0, 360, cv2.GC_PR_FGD, -1)   # head
m[520:, 80:720] = cv2.GC_PR_FGD                                         # shoulders / suit
cv2.ellipse(m, (408, 410), (80, 110), 0, 0, 360, cv2.GC_FGD, -1)        # face core
m[600:, 250:560] = cv2.GC_FGD                                           # chest
bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
cv2.grabCut(im, m, None, bg, fg, 8, cv2.GC_INIT_WITH_MASK)
out = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(out)
out = np.where(lab == 1 + np.argmax(st[1:, 4]), 255, 0).astype(np.uint8)
out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
out[730:, 735:] = 0                                                     # plant in the corner
cv2.imwrite(f"{D}/cutout_mask.png", out)
v = im.copy(); v[out == 0] = (v[out == 0] * 0.25 + np.array([200, 0, 200]) * 0.75).astype(np.uint8)
cv2.imwrite(sys.argv[2], v)
# then (by hand): source.jpg / cutout_mask.png = 560x560+130+200 crop of the full-frame versions, resized to 800x800
