"""Card size and the areas on the back the notes must avoid (back-view mm: x from the left as you look at the back)."""
BW, BH = 304.8, 228.6            # 12 x 9 in
M, GAP = 2.5, 3.5                # margin to the board edge, gap between notes
CORNERS = [(0, 0, 10.5, 10.5), (BW - 10.5, 0, BW, 10.5), (0, BH - 10.5, 10.5, BH), (BW - 10.5, BH - 10.5, BW, BH)]
GROUP = (90.4, 145.9, 214.4, BH)                 # group photo (back-view x 92.4-212.4, y 148.4-224.4) + 2 mm
BACK_PORTRAIT = {"reagan": (218.5, 145.9, 267.5, BH),      # back-view x 220.5-265.5
                 "doruk": (217.0, 145.9, 282.2, BH),       # back-view x 219.0-280.2
                 "carolyn": None}
DIRS = {"reagan": "notes_pcb", "carolyn": "people/carolyn", "doruk": "people/doruk"}
def obstacles(person):
    return CORNERS + [GROUP] + ([BACK_PORTRAIT[person]] if BACK_PORTRAIT[person] else [])
