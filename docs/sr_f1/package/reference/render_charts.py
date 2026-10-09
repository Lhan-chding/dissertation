"""Deterministic Pillow chart renderer for generated numeric data, not model-produced art.
Runtime must verify the native Qwen processor's resampling and readability before training.
Font files are NOT distributed in this package. Record local font path/hash at execution.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FONT_PATH='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def render(task: dict,path: Path) -> dict:
    world=task['world'];chart=task['chart'];style=task['style']
    im=Image.new('RGB',(1024,768),'white');d=ImageDraw.Draw(im)
    normal=ImageFont.truetype(FONT_PATH,22);small=ImageFont.truetype(FONT_PATH,18)
    title=ImageFont.truetype(FONT_BOLD,27)
    # Palette/layout changes are controlled dataset factors, not a model-performance filter.
    palette={'Alpha':(42,100,157),'Beta':(197,100,38),'Gamma':(91,145,91)}
    if style=='equivalent':palette={'Alpha':(100,82,147),'Beta':(37,130,132),'Gamma':(157,113,58)}
    if style=='heldout':palette={'Alpha':(160,51,93),'Beta':(59,132,79),'Gamma':(93,102,163)}
    left,top,right,bottom=92,145,970,660
    d.text((left,26),'Quarterly quantities',font=title,fill='black')
    d.text((left,70),'Unit: count',font=small,fill='black')
    series=list(world['series'])
    if style=='equivalent':series=list(reversed(series))
    lx=400
    for s in series:
        d.rectangle((lx,66,lx+24,83),fill=palette[s]);d.text((lx+33,61),s,font=normal,fill='black');lx+=165
    d.text((left,105),'Category order: January, February, March, April',font=small,fill='black')
    ys=lambda val:bottom-(bottom-top)*val/100
    for val in range(0,101,5):
        y=ys(val);major=val%10==0
        d.line((left,y,right,y),fill=(207,211,215) if major else (239,240,242),width=1)
        if major:d.text((left-16,y),str(val),font=small,anchor='rm',fill='black')
    d.line((left,top,left,bottom,right,bottom),fill='black',width=2)
    xs=[left+(i+.5)*(right-left)/4 for i in range(4)]
    for i,c in enumerate(world['categories']):d.text((xs[i],bottom+24),c,font=normal,anchor='mt',fill='black')
    if chart=='grouped_bar':
        width=41 if len(series)==3 else 51;gap=9
        for i,x in enumerate(xs):
            start=x-(len(series)*width+(len(series)-1)*gap)/2
            for k,s in enumerate(series):
                x0=start+k*(width+gap); y=ys(world['series'][s][i])
                d.rectangle((x0,y,x0+width,bottom-1),fill=palette[s],outline='black',width=1)
    elif chart=='line':
        for s in series:
            points=[(xs[i],ys(v)) for i,v in enumerate(world['series'][s])]
            d.line(points,fill=palette[s],width=3)
            for x,y in points:
                if style=='heldout':d.rectangle((x-6,y-6,x+6,y+6),fill=palette[s],outline='black')
                else:d.ellipse((x-6,y-6,x+6,y+6),fill=palette[s],outline='black')
    else:raise ValueError(chart)
    path.parent.mkdir(parents=True,exist_ok=True);im.save(path,format='PNG',optimize=False)
    return {'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'width':1024,'height':768,'pixels_per_unit_before_processor':(bottom-top)/100,
            'font_sha256':hashlib.sha256(Path(FONT_PATH).read_bytes()).hexdigest()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,default=Path('manifests/TASKS_GOLD_AUDIT_ONLY.jsonl'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--examples-only',action='store_true');args=p.parse_args()
    records=[json.loads(x) for x in args.tasks.read_text().splitlines()]
    if args.examples_only:
        first={}
        for r in records:
            if r['pool']=='TRAIN':first.setdefault(r['family'],r['root_id'])
        records=[r for r in records if r['root_id']==first.get(r['family'])]
    seen=set();receipt=[]
    for r in records:
        if r['image_file'] in seen:continue
        seen.add(r['image_file']);receipt.append(render(r,args.out/r['image_file']))
    if not args.examples_only:
        blank=args.out/'images'/'diagnostic_blank.png'
        blank.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(1024,768),'white').save(blank)
        receipt.append({'file':str(blank),'sha256':hashlib.sha256(blank.read_bytes()).hexdigest(),'type':'diagnostic_blank_not_scientific_world'})
    (args.out/'RENDER_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'rendered_distinct_images':len(receipt),'examples_only':args.examples_only}))
