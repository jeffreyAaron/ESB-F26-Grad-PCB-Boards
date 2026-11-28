"""Scale a card's KiCad board about its top-left corner (page 30, 30): text (position, size, stroke),
the field copper zone, comments and the reference image. Mounting holes stay 5 mm in from each corner of the
new board size. Edge.Cuts and the keychain footprints are rebuilt by keychain/apply_to_board.py, so run
`apply_to_board.py BOARD clean` before this and `... add` after regenerating the keychain row.

  KiCad-python fab/scale_board.py BOARD SCALE NEW_W NEW_H"""
import sys, pcbnew
path, S, NW, NH = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
b = pcbnew.LoadBoard(path)
O = 30.0
mm, iu = pcbnew.ToMM, pcbnew.FromMM
def sp(p): return pcbnew.VECTOR2I(iu(O + (mm(p.x) - O) * S), iu(O + (mm(p.y) - O) * S))
n_txt = n_other = 0
for d in list(b.GetDrawings()):
    if b.GetLayerName(d.GetLayer()) == "Edge.Cuts": continue
    if d.GetClass() == "PCB_TEXT":
        d.SetPosition(sp(d.GetPosition()))
        ts = d.GetTextSize(); d.SetTextSize(pcbnew.VECTOR2I(int(ts.x * S), int(ts.y * S)))
        d.SetTextThickness(int(d.GetTextThickness() * S)); n_txt += 1
    elif d.GetClass() == "PCB_SHAPE":
        d.SetStart(sp(d.GetStart())); d.SetEnd(sp(d.GetEnd())); n_other += 1
    elif d.GetClass() == "PCB_REFERENCE_IMAGE":
        d.SetPosition(sp(d.GetPosition()))
        try: d.GetReferenceImage().SetImageScale(d.GetReferenceImage().GetImageScale() * S)
        except Exception: pass
        n_other += 1
for z in b.Zones():
    ol = z.Outline()
    for i in range(ol.TotalVertices()):
        v = ol.CVertex(i); ol.SetVertex(i, sp(v))
    z.UnFill()
holes = {"H1": (O + 5, O + 5), "H2": (O + NW - 5, O + 5), "H3": (O + 5, O + NH - 5), "H4": (O + NW - 5, O + NH - 5)}
for f in b.GetFootprints():
    r = f.GetReference()
    if r in holes: f.SetPosition(pcbnew.VECTOR2I(iu(holes[r][0]), iu(holes[r][1])))
pcbnew.SaveBoard(path, b)
print(f"{path}: scaled {n_txt} texts, {n_other} other items, {len(list(b.Zones()))} zones; holes at", holes)
