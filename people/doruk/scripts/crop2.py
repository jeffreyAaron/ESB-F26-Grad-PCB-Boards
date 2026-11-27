# generic: crop note blocks (union of rects, optional whiteout rects), auto-invert dark-background notes, trim, collapse gaps
import pymupdf, json, numpy as np, cv2, sys
pdf, bj, out = sys.argv[1:4]
B = json.load(open(bj)); p = pymupdf.open(pdf)[0]; DPI = 600; k = DPI/72
pitch = {}
for n, v in B.items():
    rs, pt = v[0], v[1]; wo = v[2] if len(v) > 2 else []
    opt = v[3] if len(v) > 3 else {}
    U = [min(r[0] for r in rs), min(r[1] for r in rs), max(r[2] for r in rs), max(r[3] for r in rs)]
    pm = p.get_pixmap(dpi=DPI, clip=pymupdf.Rect(*U), colorspace=pymupdf.csGRAY, alpha=False)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width).copy()
    def img_ink(xref, bbox):
        """ink mask of an embedded image, placed where it sits in this crop"""
        im = pymupdf.Pixmap(p.parent, xref)
        if im.alpha or im.n > 1: im = pymupdf.Pixmap(pymupdf.csGRAY, im)
        g = np.frombuffer(im.samples, np.uint8).reshape(im.height, im.width)
        x0, y0, x1, y1 = [int(round((v_-o)*k)) for v_, o in zip(bbox, [U[0], U[1], U[0], U[1]])]
        g = cv2.resize(g, (x1-x0, y1-y0), interpolation=cv2.INTER_AREA)
        full = np.full(a.shape, 255, np.uint8)
        X0, Y0 = max(0, x0), max(0, y0); X1, Y1 = min(a.shape[1], x1), min(a.shape[0], y1)
        full[Y0:Y1, X0:X1] = g[Y0-y0:Y1-y0, X0-x0:X1-x0]
        return full
    if "only_image" in opt: a = img_ink(*opt["only_image"])
    if "minus_image" in opt:
        star = cv2.dilate((img_ink(*opt["minus_image"]) < 140).astype(np.uint8), np.ones((5, 5), np.uint8))
        a[star > 0] = 255
    inverted = np.median(a) < 128
    if inverted:
        a = 255 - a                                        # white-on-purple notes -> dark ink on white
        ink0 = a < 140; Hh, Ww = a.shape; e = int(0.06 * max(Hh, Ww))
        for yy in [*range(e), *range(Hh - e, Hh)]:
            if ink0[yy].mean() > 0.5: a[yy] = 255          # frame edge of the purple box (rows)
        for xx in [*range(e), *range(Ww - e, Ww)]:
            if ink0[:, xx].mean() > 0.5: a[:, xx] = 255    # (columns)
    m = np.zeros_like(a)
    for r in rs:
        x0, y0, x1, y1 = [int(round((v_-o)*k)) for v_, o in zip(r, [U[0], U[1], U[0], U[1]])]; m[y0:y1, x0:x1] = 1
    for r in wo:
        x0, y0, x1, y1 = [int(round((v_-o)*k)) for v_, o in zip(r, [U[0], U[1], U[0], U[1]])]; m[max(0,y0):y1, max(0,x0):x1] = 0
    a[m == 0] = 255
    ink = (a < 140).astype(np.uint8)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8); H, W = ink.shape; keep = np.zeros_like(ink)
    for i in range(1, nl):
        x, y, w, h, ar = st[i]
        if ar < 25: continue
        if inverted and max(w, h) > 30 * max(1, min(w, h)) and max(w, h) > 0.5 * max(W, H): continue   # frame edge of the purple box
        if (x <= 2 or y <= 2 or x+w >= W-2 or y+h >= H-2) and ar < 4000: continue
        keep[lab == i] = 1
    ys, xs = np.nonzero(keep); g = a[ys.min():ys.max()+1, xs.min():xs.max()+1].copy(); g[keep[ys.min():ys.max()+1, xs.min():xs.max()+1] == 0] = 255
    rows = (g < 140).any(1); lim = int(1.1*pt*k); kp = int(0.7*pt*k); idx = []; y = 0
    while y < len(rows):
        if rows[y]: idx.append(y); y += 1; continue
        s = y
        while y < len(rows) and not rows[y]: y += 1
        idx += list(range(s, y)) if y - s <= lim else list(range(s, s + kp))
    cv2.imwrite(f"{out}/{n}_t.png", g[idx]); pitch[n] = pt
    print(n, g[idx].shape)
json.dump(pitch, open(f"{out}/pitch.json", "w"))
