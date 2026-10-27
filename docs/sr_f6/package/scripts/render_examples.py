"""Render development charts only. No gold, selected set, arm, or collision cue.
This is a common reference renderer, not evidence of real-processor readability.
Font files are read locally and are never copied into the artifact package.
"""
import json,sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1]
fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
if not fontpath.is_file():raise SystemExit('Set a readable local DejaVuSans path; do not distribute font binaries.')
def font(s):return ImageFont.truetype(str(fontpath),s)
def render(t,out):
    im=Image.new('RGB',(1280,896),'white');d=ImageDraw.Draw(im)
    left,right,top,bottom=110,1225,125,755
    d.text((left,35),'Product measurements',font=font(28),fill='black')
    # Same colors, scale, geometry and labels for every factor; never encode P0/P1.
    cols={'Alpha':(63,110,156),'Beta':(184,112,48)}
    for j,(s,c) in enumerate(cols.items()):
        x=840+j*170;d.rectangle([x,55,x+24,79],fill=c);d.text((x+35,50),s,font=font(23),fill='black')
    for v in range(0,101,10):
        y=bottom-(bottom-top)*v/100
        d.line((left,y,right,y),fill=(215,215,215),width=1)
        d.text((left-15,y),str(v),font=font(21),fill='black',anchor='rm')
    w=t['world'];n=len(w['categories']);space=(right-left)/n;bw=space*.28
    for i,c in enumerate(w['categories']):
        center=left+space*(i+.5)
        for j,s in enumerate(('Alpha','Beta')):
            x=center+(j-.5)*bw*1.12
            y=bottom-(bottom-top)*w['series'][s][i]/100
            d.rectangle((x-bw/2,y,x+bw/2,bottom),fill=cols[s])
        d.text((center,bottom+25),c,font=font(22),fill='black',anchor='mt')
    d.line((left,top,left,bottom,right,bottom),fill='black',width=2)
    d.text((left,825),'Counts   |   Read categories in printed left-to-right order',font=font(22),fill='black')
    out.parent.mkdir(exist_ok=True,parents=True);im.save(out)
    return out.name
if __name__=='__main__':
    roots=json.loads((R/'examples/DEV_ROOTS_PRIVATE.json').read_text());manifest=[]
    for root in roots:
        # Anchor plus P1 first edit, each with its own one-question prompt.
        for variant in (0,1):
            t=root['tasks']['P1'][variant];p=R/'examples/charts'/f"{t['qid']}.png";render(t,p)
            manifest.append({'qid':t['qid'],'image':str(p.relative_to(R)),'size':[1280,896],
                 'scope':'DEV_CAL only; no actual processor/model validation'})
    (R/'examples/charts/MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'rendered {len(manifest)} development images')
