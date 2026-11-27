import cv2, json, random, sys
pitch=json.load(open('crops/pitch.json')); k=600/72
dims={n:cv2.imread(f'crops/{n}_t.png',0).shape for n in pitch}   # (h,w) px
BW,BH=203.2,152.4; M=2.5; GAP=float(sys.argv[1]) if len(sys.argv)>1 else 3.0
PH=dict(x0=59.6,x1=143.6,y0=96.4)  # photo reserve (back view)
OBS=[(0,0,10.5,10.5),(BW-10.5,0,BW,10.5),(0,BH-10.5,10.5,BH),(BW-10.5,BH-10.5,BW,BH),
     (PH['x0'],PH['y0'],PH['x1'],BH),(145.0,96.4,179.5,BH)]
def size(n,P): h,w=dims[n]; s=P/pitch[n]/k; return w*s,h*s
def cut(free,o):
    ox0,oy0,ox1,oy1=o; out=[]
    for (x0,y0,x1,y1) in free:
        if ox0>=x1 or ox1<=x0 or oy0>=y1 or oy1<=y0: out.append((x0,y0,x1,y1)); continue
        if ox0>x0: out.append((x0,y0,ox0,y1))
        if ox1<x1: out.append((ox1,y0,x1,y1))
        if oy0>y0: out.append((x0,y0,x1,oy0))
        if oy1<y1: out.append((x0,oy1,x1,y1))
    out=[r for r in out if r[2]-r[0]>0.5 and r[3]-r[1]>0.5]
    return [r for r in out if not any(q!=r and q[0]<=r[0] and q[1]<=r[1] and q[2]>=r[2] and q[3]>=r[3] for q in out)]
def pack(P,order):
    g=GAP/2
    free=[(M-g,M-g,BW-M+g,BH-M+g)]
    for o in OBS: free=cut(free,(o[0]-g,o[1]-g,o[2]+g,o[3]+g))
    place={}
    for n in order:
        w,h=size(n,P); w+=GAP; h+=GAP; best=None
        for (x0,y0,x1,y1) in free:
            if x1-x0>=w and y1-y0>=h:
                sc=(y0, x0)  # top-left (bottom-left rule) -> tidy rows
                if best is None or sc<best[0]: best=(sc,x0,y0)
        if best is None: return None
        _,x,y=best; place[n]=(x+g,y+g,w-GAP,h-GAP); free=cut(free,(x,y,x+w,y+h))
    return place
random.seed(int(sys.argv[2]) if len(sys.argv)>2 else 1)
names=list(pitch); bestP=0;bestL=None
orders=[sorted(names,key=lambda n:-size(n,1)[0]*size(n,1)[1]), sorted(names,key=lambda n:-size(n,1)[1]), sorted(names,key=lambda n:-size(n,1)[0])]
for _ in range(4000):
    o=names[:]; random.shuffle(o); orders.append(o)
for o in orders:
    lo,hi=max(bestP,1.5),4.0
    if not pack(lo+1e-3,o): continue
    for _ in range(18):
        m=(lo+hi)/2
        if pack(m,o): lo=m
        else: hi=m
    if lo>bestP: bestP,bestL,bo=lo,pack(lo,o),o
print('best pitch mm',round(bestP,3))
json.dump(dict(P=bestP,place=bestL,order=bo),open('layout.json','w'),indent=1)
for n,v in bestL.items(): print(n,[round(t,1) for t in v])
