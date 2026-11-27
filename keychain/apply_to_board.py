"""Put the keychain into a KiCad board: replace Edge.Cuts with the outline from geometry.json and add the plated
keyring hole + mouse-bite holes as board-only footprints.  Run twice (KiCad's Python gets confused after removals):
   python3 apply_to_board.py BOARD clean    then    python3 apply_to_board.py BOARD add"""
import pcbnew, json, sys, os
P, STAGE = sys.argv[1], sys.argv[2]
G = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry.json")))
b = pcbnew.LoadBoard(P); O = 30.0; mm = pcbnew.FromMM
V = lambda x, y: pcbnew.VECTOR2I(mm(O + x), mm(O + y))
if STAGE == "clean":
    old = [d for d in b.GetDrawings() if d.GetLayer() == pcbnew.Edge_Cuts]
    fps = [f for f in b.GetFootprints() if f.GetReference().startswith(("KEY", "MB"))]
    for d in old: b.Remove(d)
    for f in fps: b.Remove(f)
    b.Save(P); print("removed", len(old), "edge items,", len(fps), "keychain footprints"); sys.exit()
def loop(pts):
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        if abs(x0 - x1) < 1e-6 and abs(y0 - y1) < 1e-6: continue
        s = pcbnew.PCB_SHAPE(b); s.SetShape(pcbnew.SHAPE_T_SEGMENT); s.SetLayer(pcbnew.Edge_Cuts)
        s.SetStart(V(x0, y0)); s.SetEnd(V(x1, y1)); s.SetWidth(mm(0.05)); b.Add(s)
loop([tuple(p) for p in G["outline"][:-1]])
for sl in G["slots"]: loop([tuple(p) for p in sl[:-1]])
def hole(ref, x, y, drill, pad, plated):
    f = pcbnew.FOOTPRINT(b); f.SetReference(ref); f.SetValue("keychain"); f.Reference().SetVisible(False); f.Value().SetVisible(False)
    f.SetBoardOnly(True); f.SetExcludedFromBOM(True); f.SetExcludedFromPosFiles(True)
    p = pcbnew.PAD(f)
    p.SetAttribute(pcbnew.PAD_ATTRIB_PTH if plated else pcbnew.PAD_ATTRIB_NPTH)
    p.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
    p.SetSize(pcbnew.VECTOR2I(mm(pad), mm(pad))); p.SetDrillSize(pcbnew.VECTOR2I(mm(drill), mm(drill)))
    p.SetLayerSet(pcbnew.PAD.PTHMask() if plated else pcbnew.PAD.UnplatedHoleMask())
    if plated: p.SetNumber("1")
    f.Add(p); b.Add(f); f.SetPosition(V(x, y))
for i, k in enumerate(G["keys"]): hole(f"KEY{i+1}", *k, G["key_drill"], G["key_pad"], True)
for i, (x, y) in enumerate(G["bites"]): hole(f"MB{i+1}", x, y, G["bite_d"], G["bite_d"], False)
b.Save(P); print("outline", len(G["outline"]), "pts,", len(G["slots"]), "slot,", len(G["bites"]), "mouse bites,", len(G["keys"]), "keyring holes")
