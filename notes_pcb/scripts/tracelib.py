import cv2, numpy as np
BW=203.2
def write_svg(m, res, ox, oy, path, note):
    """m: back-view bitmap (1=feature) whose pixel (0,0) sits at back-view mm (ox,oy). Writes mirrored board-coord polygons."""
    cs,h=cv2.findContours(m.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    h=h[0]; dep=[]
    for i in range(len(cs)):
        d=0;p=h[i][3]
        while p>=0: d+=1;p=h[p][3]
        dep.append(d)
    out=[]
    for c,d in sorted(zip(cs,dep),key=lambda t:t[1]):
        a=cv2.approxPolyDP(c,0.6,True).reshape(-1,2).astype(float)
        if len(a)<3: continue
        xs=BW-(ox+(a[:,0]+0.5)*res); ys=oy+(a[:,1]+0.5)*res
        out.append((d," ".join(f"{x:.3f},{y:.3f}" for x,y in zip(xs,ys))))
    with open(path,'w') as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" width="203.2mm" height="152.4mm" viewBox="0 0 203.2 152.4">\n')
        f.write(f'<!-- {note}. Board coordinates as seen from the FRONT (bottom-side art is mirrored). data-level: even=feature, odd=hole -->\n')
        f.write('<path fill="black" fill-rule="evenodd" d="'+" ".join("M"+p+"Z" for _,p in out)+'"/>\n')
        for d,p in out: f.write(f'<polygon data-level="{d}" points="{p}" fill="none"/>\n')
        f.write('</svg>\n')
    return len(out)
