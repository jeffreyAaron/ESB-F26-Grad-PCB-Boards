#!/usr/bin/env python3
"""Export JLC gerbers + drills from the KiCad board and merge the dithered campanile art into them.

The art lives outside KiCad (as 0.2 mm-per-dot bitmaps) so the board file stays light:
  campanile_pcb/board_305x229/art_F_Cu.png             -> F.Cu         (black = copper)
  campanile_pcb/board_305x229/art_F_Mask_openings.png  -> F.Mask       (black = opening -> ENIG gold)
  campanile_pcb/board_305x229/art_F_Silkscreen.png     -> F.Silkscreen (black = white ink)
the dithered group photo on the back (same 0.2 mm dots, mirrored because it is read from the bottom):
  group_pcb/fade4_F_Cu.png / fade4_F_Mask_openings.png / fade4_F_Silkscreen.png -> B.Cu / B.Mask / B.Silkscreen
  portrait_back/fade4_*.png (grad portrait, same treatment)                         -> B.Cu / B.Mask / B.Silkscreen
the break-off keychain row for the card (keychain/<row>/keychain_Cu.svg, keychain_Mask.svg -> F/B.Cu, F/B.Mask),
and the traced handwritten notes (vector polygons, already mirrored for the bottom side):
  notes_pcb/notes_B_Silkscreen.svg                      -> B.Silkscreen (data-level even = ink, odd = hole)
Usage:  python3 fab/make_gerbers.py [person]     person = reagan (default), carolyn, doruk
        writes fab/gerbers_<person>/ and fab/ESB_grad_<person>_JLC.zip (Reagan keeps fab/gerbers + ESB_grad_project_JLC.zip)
Other people's boards live in people/<person>/ (board file, notes svg, optional front_portrait/ with front_*.png).
"""
import os, re, sys, glob, shutil, subprocess, zipfile
import numpy as np
from PIL import Image

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSON = sys.argv[1] if len(sys.argv) > 1 else "reagan"
PEOPLE = {
    "reagan": dict(pcb="ESB_grad_project.kicad_pcb", notes="notes_pcb/notes_B_Silkscreen.svg", front_portrait="portrait_pcb",
                   back_portrait="portrait_back", back_portrait_at=(39.3, 148.4), out="fab/gerbers", zip="fab/ESB_grad_project_JLC.zip"),
}
for _p in ("carolyn", "doruk"):
    _fp = f"people/{_p}/front_portrait"
    PEOPLE[_p] = dict(pcb=f"people/{_p}/ESB_grad_{_p}.kicad_pcb", notes=f"people/{_p}/notes_B_Silkscreen.svg",
                      front_portrait=_fp if os.path.exists(os.path.join(PROJ, _fp, "front_F_Cu_clear.png")) else None,
                      back_portrait=None, back_portrait_at=None, out=f"fab/gerbers_{_p}", zip=f"fab/ESB_grad_{_p}_JLC.zip")
# Doruk: face portrait (people/doruk/back_portrait/portrait_fade.py, 61.2 x 76.0 mm) right of the group photo as seen
# from the back (back-view x 219.0-280.2) -> board x 24.6-85.8
PEOPLE["doruk"].update(back_portrait="people/doruk/back_portrait", back_portrait_at=(24.6, 148.4))
# keychain row per card (keychain/row.py): 14 cubes on Reagan and Carolyn; Doruk: 10 cubes + dog, monkey, crewmate
for _p, _r in (("reagan", "row_cubes"), ("carolyn", "row_cubes"), ("doruk", "row_mixed")): PEOPLE[_p]["keychain"] = f"keychain/{_r}"
CFG = {k: (os.path.join(PROJ, v) if isinstance(v, str) else v) for k, v in PEOPLE[PERSON].items()}
PCB, OUT = CFG["pcb"], CFG["out"]
ART = os.path.join(PROJ, "campanile_pcb", "board_305x229")   # campanile_pcb/campanile_art.py
CLI = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"
# where the art bitmap sits, in KiCad page coordinates (mm): board top-left (30, 30) + art offset
BOARD_X, BOARD_Y, BW = 30.0, 30.0, 304.8      # 12 x 9 in card
PITCH = 0.2
LAYERS = "F.Cu,B.Cu,F.Mask,B.Mask,F.Paste,B.Paste,F.Silkscreen,B.Silkscreen,Edge.Cuts"
NOTES = CFG["notes"]
GROUP = os.path.join(PROJ, "group_pcb")
PORTRAIT = CFG["back_portrait"]
# (layer tag, png, top-left in board mm or None = right-aligned at the top, mirror)
# group photo: centred, board x 92.4-212.4, y 148.4-224.4; edges fade out (group_fade.py)
FRONT_PORTRAIT = CFG["front_portrait"]
MERGE = [("F_Cu", os.path.join(ART, "art_F_Cu.png"), None, False),
         ("F_Mask", os.path.join(ART, "art_F_Mask_openings.png"), None, False),
         ("F_Silkscreen", os.path.join(ART, "art_F_Silkscreen.png"), None, False),
         ("B_Cu", os.path.join(GROUP, "fade4_F_Cu.png"), (92.4, 148.4), True),
         ("B_Mask", os.path.join(GROUP, "fade4_F_Mask_openings.png"), (92.4, 148.4), True),
         ("B_Silkscreen", os.path.join(GROUP, "fade4_F_Silkscreen.png"), (92.4, 148.4), True)]
if FRONT_PORTRAIT:
    # front portrait (portrait_pcb/portrait_front.py): board x 12-109.6, y 63-160.6, sits on the field copper;
    # its dark dots are cut out of the field with clear polarity
    MERGE += [("F_Cu", os.path.join(FRONT_PORTRAIT, "front_F_Cu_clear.png"), (12.0, 63.0), False, True),
              ("F_Mask", os.path.join(FRONT_PORTRAIT, "front_F_Mask_openings.png"), (12.0, 63.0), False),
              ("F_Silkscreen", os.path.join(FRONT_PORTRAIT, "front_F_Silkscreen.png"), (12.0, 63.0), False)]
if PORTRAIT:
    # back portrait, mirrored; Reagan's sits at board x 39.3-84.3, y 148.4-224.4 (right of the group photo seen from the back)
    AT = CFG["back_portrait_at"]
    MERGE += [("B_Cu", os.path.join(PORTRAIT, "fade4_F_Cu.png"), AT, True),
              ("B_Mask", os.path.join(PORTRAIT, "fade4_F_Mask_openings.png"), AT, True),
              ("B_Silkscreen", os.path.join(PORTRAIT, "fade4_F_Silkscreen.png"), AT, True)]

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

def merge(gbr, png, at=None, mirror=False, clear=False):
    m = np.asarray(Image.open(png).convert("L")) < 128
    if mirror: m = m[:, ::-1]
    h, w = m.shape
    if at is None: x0 = BOARD_X + BW - w * PITCH; y0 = BOARD_Y      # campanile art: right-aligned at the top
    else: x0 = BOARD_X + at[0]; y0 = BOARD_Y + at[1]
    txt = open(gbr).read()
    fs = re.search(r"%FSLAX(\d)(\d)Y\d\d\*%", txt); dec = int(fs.group(2)); k = 10 ** dec
    assert "%MOMM*%" in txt, "expected mm units"
    c = lambda v: str(int(round(v * k)))
    out = ["G04 --- dithered art merged by fab/make_gerbers.py ---*", "%TD*%", "%LPC*%" if clear else "%LPD*%", "G01*"]
    rs = rects(m)
    for (a, b, a1, b1) in rs:
        X0, X1 = x0 + a * PITCH, x0 + a1 * PITCH
        Y0, Y1 = -(y0 + b * PITCH), -(y0 + b1 * PITCH)     # KiCad gerbers: Y up = -page Y
        out += ["G36*", f"X{c(X0)}Y{c(Y0)}D02*", f"X{c(X1)}Y{c(Y0)}D01*", f"X{c(X1)}Y{c(Y1)}D01*",
                f"X{c(X0)}Y{c(Y1)}D01*", f"X{c(X0)}Y{c(Y0)}D01*", "G37*"]
    if clear: out.append("%LPD*%")
    i = txt.rstrip().rfind("M02*"); assert i > 0
    open(gbr, "w").write(txt[:i] + "\n".join(out) + "\nM02*\n")
    return len(rs)

def merge_notes(gbr, svg, label="handwritten notes"):
    """Append traced note outlines as G36 regions; holes are drawn with clear polarity, outermost first."""
    txt = open(gbr).read()
    dec = int(re.search(r"%FSLAX\d(\d)Y\d\d\*%", txt).group(1)); k = 10 ** dec
    c = lambda v: str(int(round(v * k)))
    polys = re.findall(r'<polygon data-level="(\d+)" points="([^"]+)"', open(svg).read())
    out = [f"G04 --- {label} merged by fab/make_gerbers.py ---*", "G01*"]
    pol = None
    for lev, pts in sorted(polys, key=lambda t: int(t[0])):
        want = "%LPD*%" if int(lev) % 2 == 0 else "%LPC*%"
        if want != pol: out.append(want); pol = want
        xy = [tuple(map(float, q.split(","))) for q in pts.split()]
        xy.append(xy[0])
        out.append("G36*")
        for j, (x, y) in enumerate(xy):
            out.append(f"X{c(BOARD_X + x)}Y{c(-(BOARD_Y + y))}D0{2 if j == 0 else 1}*")
        out.append("G37*")
    out.append("%LPD*%")
    i = txt.rstrip().rfind("M02*"); assert i > 0
    open(gbr, "w").write(txt[:i] + "\n".join(out) + "\nM02*\n")
    return len(polys)

def main():
    if os.path.isdir(OUT): shutil.rmtree(OUT)
    os.makedirs(OUT)
    subprocess.run([CLI, "pcb", "export", "gerbers", "--layers", LAYERS, "--subtract-soldermask",
                    "--check-zones", "-o", OUT + "/", PCB], check=True)
    subprocess.run([CLI, "pcb", "export", "drill", "--format", "excellon", "--excellon-separate-th",
                    "--generate-map", "--map-format", "gerberx2", "-o", OUT + "/", PCB], check=True)
    for tag, png, at, mirror, *clear in MERGE:
        g = glob.glob(os.path.join(OUT, f"*-{tag}.g*"))
        assert len(g) == 1, (tag, g)
        print(f"merged {merge(g[0], png, at, mirror, bool(clear and clear[0]))} rectangles from {os.path.basename(png)} into {os.path.basename(g[0])}")
    g = glob.glob(os.path.join(OUT, "*-B_Silkscreen.g*")); assert len(g) == 1, g
    print(f"merged {merge_notes(g[0], NOTES)} note outlines into {os.path.basename(g[0])}")
    # break-off keychain row (keychain/row.py): gold + purple, same on both sides
    KC = CFG["keychain"]
    bm = "keychain_Mask_B.svg" if os.path.exists(os.path.join(KC, "keychain_Mask_B.svg")) else "keychain_Mask.svg"   # back art (letters read right)
    for tag, f in (("F_Cu", "keychain_Cu.svg"), ("B_Cu", "keychain_Cu.svg"), ("F_Mask", "keychain_Mask.svg"), ("B_Mask", bm)):
        g = glob.glob(os.path.join(OUT, f"*-{tag}.g*")); assert len(g) == 1, g
        print(f"merged {merge_notes(g[0], os.path.join(KC, f), 'keychain')} keychain outlines into {os.path.basename(g[0])}")
    z = CFG["zip"]
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(OUT)): zf.write(os.path.join(OUT, f), f)
    print("wrote", z)

if __name__ == "__main__":
    main()
