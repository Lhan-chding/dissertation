"""Inspectable demo stimuli. Production must bind to a frozen renderer revision."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def render(world,path,style='demo-r0'):
    im=Image.new('RGB',(1200,700),'white');d=ImageDraw.Draw(im)
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try:font=ImageFont.truetype(fontpath,23);small=ImageFont.truetype(fontpath,20)
    except OSError:raise RuntimeError('DejaVu Sans unavailable: register a replacement, do not silently use tiny fallback')
    left,right,top,bottom=100,1160,110,590;unit=(bottom-top)/100
    for x in range(0,101,10):
        y=bottom-unit*x;d.line((left,y,right,y),fill='#dddddd',width=1)
        d.text((left-14,y),str(x),font=small,fill='black',anchor='rm')
    d.line((left,top,left,bottom,right,bottom),fill='black',width=2)
    n=len(world['categories']);stride=(right-left)/n;colors=({'Alpha':'#335d91','Beta':'#a75d35'} if style!='r1' else {'Alpha':'#674c8c','Beta':'#3b7965'})
    for i,c in enumerate(world['categories']):
        center=left+stride*(i+.5)
        for j,s in enumerate(('Alpha','Beta')):
            cx=center+(-18 if j==0 else 18);v=world['series'][s][i]
            d.rectangle((cx-14,bottom-unit*v,cx+14,bottom),fill=colors[s])
        d.text((center,bottom+15),c,font=font,fill='black',anchor='mt')
    d.text((left,45),'Chart values',font=font,fill='black')
    for j,s in enumerate(('Alpha','Beta')):
        x=800+j*150;d.rectangle((x,45,x+24,65),fill=colors[s]);d.text((x+34,42),s,font=small,fill='black')
    Path(path).parent.mkdir(parents=True,exist_ok=True);im.save(path)
    return {'width':1200,'height':700,'pixels_per_value_unit':unit,'min_main_gap_px':6*unit,
            'numeric_bar_labels':False,'category_count':n,'style':style,'font_files_distributed':False}
