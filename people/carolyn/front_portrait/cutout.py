# GrabCut cut-out of Carolyn (busy lab background) -> cutout_mask_full.png (255 = person)
import cv2, numpy as np, sys
D = sys.argv[1]
im = cv2.imread(f"{D}/source_full.jpg"); H, W = im.shape[:2]
m = np.full((H, W), cv2.GC_BGD, np.uint8)
m[110:, 190:680] = cv2.GC_PR_BGD
cv2.ellipse(m, (400, 285), (125, 165), 0, 0, 360, cv2.GC_PR_FGD, -1)   # head + hair
pts = np.array([[330, 430], [470, 430], [560, 470], [640, 560], [640, 800], [220, 800], [235, 560]], np.int32)
cv2.fillPoly(m, [pts], cv2.GC_PR_FGD)                                    # shoulders / shirt
cv2.ellipse(m, (385, 300), (70, 100), 0, 0, 360, cv2.GC_FGD, -1)        # face core
cv2.fillPoly(m, [np.array([[300, 560], [520, 540], [560, 800], [280, 800]], np.int32)], cv2.GC_FGD)   # torso core
bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
cv2.grabCut(im, m, None, bg, fg, 10, cv2.GC_INIT_WITH_MASK)
out = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(out)
out = np.where(lab == 1 + np.argmax(st[1:, 4]), 255, 0).astype(np.uint8)
# limit to a hand-drawn silhouette so the black equipment frame behind her shoulders is dropped
lim = np.zeros((H, W), np.uint8)
cv2.ellipse(lim, (400, 280), (135, 160), 0, 0, 360, 255, -1)
cv2.fillPoly(lim, [np.array([[335, 410], [470, 410], [505, 440], [555, 478], [590, 520], [608, 600], [618, 800],
                             [215, 800], [228, 640], [245, 560], [280, 500], [318, 460]], np.int32)], 255)
out &= lim
out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
cv2.imwrite(f"{D}/cutout_mask_full.png", out)
v = im.copy(); v[out == 0] = (v[out == 0] * 0.25 + np.array([200, 0, 200]) * 0.75).astype(np.uint8)
cv2.imwrite(sys.argv[2], v)
# then: source.jpg / cutout_mask.png = 560x560+120+95 crop of the full-frame versions, resized to 800x800
