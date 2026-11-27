# compose back-view bitmap at RES mm/px, enforce min stroke, write preview
import cv2, numpy as np, json, sys
from skimage.morphology import skeletonize
RES=0.02; BW,BH=203.2,152.4; MINW=0.16
L=json.load(open('layout.json')); pitch=json.load(open('crops/pitch.json')); k=600/72
W,H=int(round(BW/RES)),int(round(BH/RES))
canvas=np.zeros((H,W),np.uint8)
for n,(x,y,w,h) in L['place'].items():
    a=cv2.imread(f'crops/{n}_t.png',0)
    pw,ph=int(round(w/RES)),int(round(h/RES))
    g=cv2.resize(a,(pw,ph),interpolation=cv2.INTER_AREA if pw<a.shape[1] else cv2.INTER_CUBIC)
    b=(g<150).astype(np.uint8)
    X,Y=int(round(x/RES)),int(round(y/RES))
    canvas[Y:Y+ph,X:X+pw]|=b
# min stroke: union with skeleton thickened to MINW
sk=skeletonize(canvas>0).astype(np.uint8)
r=int(round(MINW/RES/2))
disk=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
canvas=np.maximum(canvas,cv2.dilate(sk,disk))
# drop specks smaller than 0.2mm across
nl,lab,st,_=cv2.connectedComponentsWithStats(canvas,8)
small=np.isin(lab,[i for i in range(1,nl) if st[i,4]<(0.18/RES)**2])
canvas[small]=0
np.save('back_view.npy',canvas)
cv2.imwrite('back_view_mask.png',(1-canvas)*255)
# preview with board colours (back view)
DARK=(86,26,58); SILK=(228,236,238)
pv=np.zeros((H,W,3),np.uint8); pv[:]=DARK; pv[canvas>0]=SILK
pv=cv2.resize(pv,(W//4,H//4),interpolation=cv2.INTER_AREA)
s=4*RES
def R(x0,y0,x1,y1,c): cv2.rectangle(pv,(int(x0/s),int(y0/s)),(int(x1/s),int(y1/s)),c,2)
pass  # photo reserve
pass # parts
for cx,cy in [(5,5),(198.2,5),(5,147.4),(198.2,147.4)]: cv2.circle(pv,(int(cx/s),int(cy/s)),int(1.6/s),(20,20,20),-1)
cv2.imwrite(sys.argv[1],pv)
print('ink %',round(canvas.mean()*100,1))
