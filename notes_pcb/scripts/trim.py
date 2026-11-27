import cv2, numpy as np, json, glob, os, sys
P=float(sys.argv[1]); pitch=json.load(open('crops/pitch.json'))
k=600/72  # px per pt
out={}
for n in pitch:
    a=cv2.imread(f'crops/{n}.png',0)
    ink=(a<140).astype(np.uint8)
    nl,lab,st,_=cv2.connectedComponentsWithStats(ink,8)
    H,W=ink.shape; keep=np.zeros_like(ink)
    for i in range(1,nl):
        x,y,w,h,ar=st[i]
        touches = x<=2 or y<=2 or x+w>=W-2 or y+h>=H-2
        if ar<25: continue
        if touches and ar<4000: continue
        keep[lab==i]=1
    ys,xs=np.nonzero(keep); x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
    g=a[y0:y1,x0:x1].copy(); g[keep[y0:y1,x0:x1]==0]=255
    cv2.imwrite(f'crops/{n}_t.png',g)
    s=P/pitch[n]/k   # mm per px
    out[n]=dict(w=round((x1-x0)*s,1),h=round((y1-y0)*s,1),mmpp=s)
    print(f"{n:9s} {out[n]['w']:6.1f} x {out[n]['h']:5.1f} mm  area {out[n]['w']*out[n]['h']:6.0f}")
print('total', sum(v['w']*v['h'] for v in out.values()))
json.dump(out,open('sizes.json','w'))
