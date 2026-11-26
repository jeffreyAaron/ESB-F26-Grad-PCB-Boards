#!/usr/bin/env python3
"""Export JLC gerbers + drills from the KiCad board and merge the dithered campanile art into them.

The art lives outside KiCad (as 0.2 mm-per-dot bitmaps) so the board file stays light:
  campanile_pcb/board_203x152/art_F_Cu.png             -> F.Cu         (black = copper)
  campanile_pcb/board_203x152/art_F_Mask_openings.png  -> F.Mask       (black = opening -> ENIG gold)
  campanile_pcb/board_203x152/art_F_Silkscreen.png     -> F.Silkscreen (black = white ink)
Usage:  python3 fab/make_gerbers.py        (writes fab/gerbers/ and fab/ESB_grad_project_JLC.zip)
"""
import os, re, sys, glob, shutil, subprocess, zipfile
import numpy as np
from PIL import Image

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "ESB_grad_project.kicad_pcb")
ART = os.path.join(PROJ, "campanile_pcb", "board_203x152")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PROJ, "fab", "gerbers")
CLI = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"
# where the art bitmap sits, in KiCad page coordinates (mm): board top-left (30, 30) + art offset
BOARD_X, BOARD_Y, BW = 30.0, 30.0, 203.2
PITCH = 0.2
LAYERS = "F.Cu,B.Cu,F.Mask,B.Mask,F.Paste,B.Paste,F.Silkscreen,B.Silkscreen,Edge.Cuts"
MERGE = {"F_Cu": "art_F_Cu.png", "F_Mask": "art_F_Mask_openings.png", "F_Silkscreen": "art_F_Silkscreen.png"}

def rects(mask):
    """Greedy merge of dot runs into rectangles: horizontal runs, extended down while identical."""
    h, w = mask.shape; used = np.zeros_like(mask); out = []
    for y in range(h):
        row = mask[y] & ~used[y]; x = 0
        while x < w:
            if not row[x]: x += 1; continue
            x1 = x
            while x1 < w and row[x1]: x1 += 1
            y1 = y + 1
            while y1 < h and mask[y1, x:x1].all() and not used[y1, x:x1].any() and \
                  (x == 0 or not (mask[y1, x-1] and not used[y1, x-1])) and (x1 == w or not mask[y1, x1]):
                y1 += 1
            used[y:y1, x:x1] = True; out.append((x, y, x1, y1)); x = x1
    return out

def merge(gbr, png):
    m = np.asarray(Image.open(png).convert("L")) < 128
    h, w = m.shape
    x0 = BOARD_X + BW - w * PITCH; y0 = BOARD_Y           # art is right-aligned on the board
    txt = open(gbr).read()
    fs = re.search(r"%FSLAX(\d)(\d)Y\d\d\*%", txt); dec = int(fs.group(2)); k = 10 ** dec
    assert "%MOMM*%" in txt, "expected mm units"
    c = lambda v: str(int(round(v * k)))
    out = ["G04 --- dithered campanile art merged by fab/make_gerbers.py ---*", "%TD*%", "%LPD*%", "G01*"]
    rs = rects(m)
    for (a, b, a1, b1) in rs:
        X0, X1 = x0 + a * PITCH, x0 + a1 * PITCH
        Y0, Y1 = -(y0 + b * PITCH), -(y0 + b1 * PITCH)     # KiCad gerbers: Y up = -page Y
        out += ["G36*", f"X{c(X0)}Y{c(Y0)}D02*", f"X{c(X1)}Y{c(Y0)}D01*", f"X{c(X1)}Y{c(Y1)}D01*",
                f"X{c(X0)}Y{c(Y1)}D01*", f"X{c(X0)}Y{c(Y0)}D01*", "G37*"]
    i = txt.rstrip().rfind("M02*"); assert i > 0
    open(gbr, "w").write(txt[:i] + "\n".join(out) + "\nM02*\n")
    return len(rs)

def main():
    if os.path.isdir(OUT): shutil.rmtree(OUT)
    os.makedirs(OUT)
    subprocess.run([CLI, "pcb", "export", "gerbers", "--layers", LAYERS, "--subtract-soldermask",
                    "--check-zones", "-o", OUT + "/", PCB], check=True)
    subprocess.run([CLI, "pcb", "export", "drill", "--format", "excellon", "--excellon-separate-th",
                    "--generate-map", "--map-format", "gerberx2", "-o", OUT + "/", PCB], check=True)
    for tag, png in MERGE.items():
        g = glob.glob(os.path.join(OUT, f"*-{tag}.g*"))
        assert len(g) == 1, (tag, g)
        print(f"merged {merge(g[0], os.path.join(ART, png))} art rectangles into {os.path.basename(g[0])}")
    z = os.path.join(os.path.dirname(OUT), "ESB_grad_project_JLC.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(OUT)): zf.write(os.path.join(OUT, f), f)
    print("wrote", z)

if __name__ == "__main__":
    main()
