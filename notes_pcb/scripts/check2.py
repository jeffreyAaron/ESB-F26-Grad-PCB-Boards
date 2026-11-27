import re, cv2, numpy as np, glob
R=0.04; W,H=int(203.2/R),int(152.4/R)
def layer(tag):
    g=open(glob.glob(f'gerb/*-{tag}.g*')[0]).read()
    i=g.find('G04 --- '); body=g[i:] if i>=0 else ''
    img=np.zeros((H,W),np.uint8); pol=1
    for t in re.finditer(r'%LP([DC])\*%|G36\*(.*?)G37\*',body,re.S):
        if t.group(1): pol=1 if t.group(1)=='D' else 0; continue
        pts=[(int(x)/1e6-30,-int(y)/1e6-30) for x,y in re.findall(r'X(-?\d+)Y(-?\d+)D0',t.group(2))]
        cv2.fillPoly(img,[np.array([[(203.2-x)/R,y/R] for x,y in pts],np.int32)],pol)
    return img
cu,mk,sk=layer('B_Cu'),layer('B_Mask'),layer('B_Silkscreen')
pv=np.zeros((H,W,3),np.uint8); pv[:]=(86,26,58)
pv[cu>0]=(138,48,96); pv[(mk>0)&(cu>0)]=(104,186,226); pv[(sk>0)&(mk==0)]=(228,236,238)
e=lambda a:cv2.erode(a,np.ones((3,3),np.uint8)); print('silk on gold (interior)',int(((e(sk)>0)&(e(mk)>0)).sum()),'mask open w/o copper',int(((e(mk)>0)&(cu==0)).sum()))
for cx,cy in [(5,5),(198.2,5),(5,147.4),(198.2,147.4)]: cv2.circle(pv,(int(cx/R),int(cy/R)),int(1.6/R),(20,20,20),-1)
cv2.imwrite('/Users/jeffrey/Desktop/ESB_grad_project/notes_pcb/preview_back.png',cv2.resize(pv,(W//2,H//2),interpolation=cv2.INTER_AREA))
