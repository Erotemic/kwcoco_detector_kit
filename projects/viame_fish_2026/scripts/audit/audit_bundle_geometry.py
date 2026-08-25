"""Geometry/integrity audit of a kwcoco bundle. stdlib only."""
import json, math, sys, collections

def audit(path, label):
    d = json.load(open(path))
    imgs = {im["id"]: im for im in d["images"]}
    anns = d["annotations"]
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    print(f"images {len(imgs):,}   annotations {len(anns):,}   "
          f"categories {[c['name'] for c in d.get('categories',[])]}")

    bad = collections.Counter()
    ws, hs, areas, rels = [], [], [], []
    per_img = collections.Counter()
    oob_margin = []
    for a in anns:
        per_img[a["image_id"]] += 1
        bb = a.get("bbox")
        if bb is None or len(bb) != 4:
            bad["missing/malformed bbox"] += 1; continue
        x, y, w, h = [float(v) for v in bb]
        if any(math.isnan(v) or math.isinf(v) for v in (x, y, w, h)):
            bad["NaN/inf coords"] += 1; continue
        if w <= 0 or h <= 0:
            bad["zero/negative w or h"] += 1; continue
        if w < 1 or h < 1:      bad["sub-pixel (<1px) side"] += 1
        if w < 2 or h < 2:      bad["tiny (<2px) side"] += 1
        im = imgs.get(a["image_id"])
        if im is None:
            bad["annotation references missing image"] += 1; continue
        W, H = float(im.get("width") or 0), float(im.get("height") or 0)
        if W <= 0 or H <= 0:
            bad["image missing width/height"] += 1; continue
        if x < -0.5 or y < -0.5:                 bad["negative origin"] += 1
        over_x, over_y = (x + w) - W, (y + h) - H
        if over_x > 0.5 or over_y > 0.5:
            bad["extends past image edge"] += 1
            oob_margin.append(max(over_x, over_y))
        # what the model actually receives after ConvertBoxes(normalize=True)
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        nw, nh = w / W, h / H
        if not (0 <= cx <= 1 and 0 <= cy <= 1):  bad["normalized center outside [0,1]"] += 1
        if nw > 1 or nh > 1:                     bad["normalized w/h > 1"] += 1
        if nw <= 0 or nh <= 0:                   bad["normalized w/h <= 0"] += 1
        ar = a.get("area")
        if ar is not None and ar > 0 and abs(ar - w * h) / (w * h) > 0.5:
            bad["area field disagrees with bbox >50%"] += 1
        ws.append(w); hs.append(h); areas.append(w * h); rels.append(nw * nh)

    def pct(v, p):
        if not v: return float("nan")
        v = sorted(v); return v[min(len(v) - 1, int(len(v) * p / 100))]

    print(f"\nbox sides (px):  w  p1 {pct(ws,1):7.1f}  p50 {pct(ws,50):7.1f}  p99 {pct(ws,99):8.1f}  max {max(ws):8.1f}")
    print(f"                 h  p1 {pct(hs,1):7.1f}  p50 {pct(hs,50):7.1f}  p99 {pct(hs,99):8.1f}  max {max(hs):8.1f}")
    print(f"box area frac of image: p1 {pct(rels,1)*100:6.3f}%  p50 {pct(rels,50)*100:6.3f}%  "
          f"p99 {pct(rels,99)*100:6.2f}%  max {max(rels)*100:6.2f}%")
    empty = len(imgs) - len(per_img)
    print(f"anns/image: mean {len(anns)/max(1,len(imgs)):.2f}  max {max(per_img.values()) if per_img else 0}  "
          f"images with 0 anns {empty:,} ({100*empty/max(1,len(imgs)):.1f}%)")
    if oob_margin:
        print(f"edge overflow px: p50 {pct(oob_margin,50):.1f}  p99 {pct(oob_margin,99):.1f}  max {max(oob_margin):.1f}")
    print("\nFINDINGS:")
    if not bad: print("  none")
    for k, v in bad.most_common():
        print(f"  {v:>9,}  ({100*v/max(1,len(anns)):5.2f}%)  {k}")
    return bad, len(anns)

if __name__ == "__main__":
    audit(sys.argv[1], sys.argv[2])
