import pymupdf, json, numpy as np, cv2, sys
B = {  # name: list of rects in PDF pt (union), pitch estimate pt
 "brian":   [[[4,4,198,108]], 8.5],
 "evana":   [[[9,104,208,241]], 13.7],
 "sarah":   [[[9,243,226,403]], 13.5],
 "veronica":[[[7,399,146,530]], 8.6],
 "sanjana": [[[204,6,521,129]], 14.5],
 "rob":     [[[221,129,508,265],[221,265,318,410]], 12],
 "amisha":  [[[320,266,533,399]], 13],
 "ella":    [[[530,14,703,240]], 14],
 "anthony": [[[537,248,694,428]], 16.8],
 "isa":     [[[154,403,538,474]], 12.5],
 "ava":     [[[151,475,541,531]], 16],
 "ethan":   [[[539,431,711,533]], 16],
 "thomas":  [[[0,536,395,797]], 18],
 "jeffrey": [[[0,797,380,912]], 17],
}
doc = pymupdf.open(sys.argv[1]); p = doc[0]
DPI = 600; k = DPI/72
for n,(rs,pitch) in B.items():
    xs=[r[0] for r in rs]+[r[2] for r in rs]; ys=[r[1] for r in rs]+[r[3] for r in rs]
    U=[min(xs),min(ys),max(xs),max(ys)]
    pm = p.get_pixmap(dpi=DPI, clip=pymupdf.Rect(*U), colorspace=pymupdf.csGRAY, alpha=False)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width).copy()
    m = np.zeros_like(a)
    for r in rs:
        x0,y0,x1,y1=[int(round((v-o)*k)) for v,o in zip(r,[U[0],U[1],U[0],U[1]])]
        m[y0:y1,x0:x1]=1
    a[m==0]=255
    cv2.imwrite(f"{sys.argv[2]}/{n}.png", a)
    print(n, a.shape, "dark%", round((a<128).mean()*100,1), "pitch", pitch)
json.dump({n:v[1] for n,v in B.items()}, open(f"{sys.argv[2]}/pitch.json","w"))
