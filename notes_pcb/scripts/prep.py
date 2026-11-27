# rework rob (move doodle beside sign-off), collapse big blank row gaps in every note
import cv2, numpy as np, json
pitch=json.load(open('crops/pitch.json')); k=600/72
def ink(a): return a<140
# rob: split at text/doodle
a=cv2.imread('crops/rob_t.png',0); H,W=a.shape
rows=ink(a).any(1)
# find last text row: doodle is the tallest bottom component group; split at largest blank gap in lower half
runs=[];y=0
while y<H:
    if not rows[y]:
        s=y
        while y<H and not rows[y]: y+=1
        runs.append((s,y))
    else: y+=1
gaps=[r for r in runs if r[0]>H*0.5]; g=max(gaps,key=lambda r:r[1]-r[0])
top=a[:g[0]]; dood=a[g[1]:]
# doodle bbox in x
cols=np.nonzero(ink(dood).any(0))[0]; dood=dood[:,cols.min():cols.max()+1]
# put doodle to the right of the last ~2 text paragraphs ("DO GREAT"/"rob") which are short
trows=np.nonzero(ink(top).any(1))[0]
# width of the bottom 25% of text
yb=int(top.shape[0]*0.72); wb=np.nonzero(ink(top[yb:]).any(0))[0].max()
x0=wb+int(30*k)
newH=max(top.shape[0], yb+dood.shape[0]); newW=max(top.shape[1], x0+dood.shape[1])
c=np.full((newH,newW),255,np.uint8); c[:top.shape[0],:top.shape[1]]=top
c[yb:yb+dood.shape[0], x0:x0+dood.shape[1]]=np.minimum(c[yb:yb+dood.shape[0], x0:x0+dood.shape[1]],dood)
cv2.imwrite('crops/rob_t.png',c)
# collapse blank gaps > 1.1*pitch to 0.7*pitch
for n,p in pitch.items():
    a=cv2.imread(f'crops/{n}_t.png',0); rows=ink(a).any(1); H=a.shape[0]
    lim=int(1.1*p*k); keep=int(0.7*p*k); out=[]; y=0
    while y<H:
        if rows[y]: out.append(y); y+=1; continue
        s=y
        while y<H and not rows[y]: y+=1
        out+=list(range(s,y)) if y-s<=lim else list(range(s,s+keep))
    cv2.imwrite(f'crops/{n}_t.png',a[out])
    print(n,H,'->',len(out))
