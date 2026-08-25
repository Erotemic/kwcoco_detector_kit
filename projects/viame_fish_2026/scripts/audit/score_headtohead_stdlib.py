"""AP@0.5 head-to-head over saved predictions. stdlib only."""
import json, collections, sys

B='/home/local/KHQ/jon.crall/ssd-data/fish_kcd'
truth=json.load(open(f'{B}/bundle/test.kwcoco.json'))
gt=collections.defaultdict(list)
for a in truth['annotations']:
    x,y,w,h=a['bbox']; gt[a['image_id']].append([x,y,x+w,y+h])
gt_ids=set(im['id'] for im in truth['images'])
n_gt=sum(len(v) for v in gt.values())

def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    if x2<=x1 or y2<=y1: return 0.0
    i=(x2-x1)*(y2-y1)
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)

def score(path,label,thr=0.5):
    d=json.load(open(path))
    pred=collections.defaultdict(list)
    unknown=0
    for a in d['annotations']:
        if a['image_id'] not in gt_ids: unknown+=1; continue
        x,y,w,h=a['bbox']; pred[a['image_id']].append((a.get('score',1.0),[x,y,x+w,y+h]))
    rows=[]
    for gid,ps in pred.items():
        ps.sort(key=lambda t:-t[0])
        boxes=gt.get(gid,[]); used=[False]*len(boxes)
        for s,pb in ps:
            best,bi=0.0,-1
            for i,tb in enumerate(boxes):
                if used[i]: continue
                v=iou(pb,tb)
                if v>best: best,bi=v,i
            if best>=thr and bi>=0:
                used[bi]=True; rows.append((s,1))
            else:
                rows.append((s,0))
    rows.sort(key=lambda t:-t[0])
    tp=fp=0; pr=[]
    for s,is_tp in rows:
        tp+=is_tp; fp+= (1-is_tp)
        pr.append((tp/n_gt, tp/(tp+fp)))
    # monotonic precision envelope + all-point interpolation
    mono=[]; best=0.0
    for r,p in reversed(pr):
        best=max(best,p); mono.append((r,best))
    mono.reverse()
    ap=0.0; prev_r=0.0
    for r,p in mono:
        ap += (r-prev_r)*p; prev_r=r
    TP=sum(t for _,t in rows); FP=len(rows)-TP
    prec=TP/max(1,len(rows)); rec=TP/n_gt
    f1=2*prec*rec/max(1e-9,prec+rec)
    print(f"{label:22} AP@0.5 {ap:.4f}   preds {len(rows):>7,}  TP {TP:>6,}  FP {FP:>6,}  "
          f"P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}" + (f"  [{unknown} preds on unknown images]" if unknown else ""))
    return ap

print(f"ground truth: {len(gt_ids):,} images, {n_gt:,} annotations (held-out test split)")
print("both prediction sets floored at score >= 0.5 -- AP AT A 0.5 SCORE FLOOR\n")
a=score(f'{B}/rfdetr_test_inference/rfdetr_test_preds.kwcoco.json',"RF-DETR (VIAME 720)")
b=score(f'{B}/headtohead/deim_preds_min05.kwcoco.json',        "DEIMv2 gen001 @1024")
print(f"\ndelta (DEIMv2 - RF-DETR): {b-a:+.4f}")
