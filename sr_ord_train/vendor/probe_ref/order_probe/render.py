"""F6 reference chart style, held fixed across conditions and paired names.
Original implementation archived in sources/historical_reference_renderer.py.
Font binaries are local dependencies and are never distributed.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def render(task: dict, output: str | Path, font_path: str) -> None:
    font_file = Path(font_path)
    if not font_file.is_file(): raise FileNotFoundError('Use an installed font; do not bundle its binary')
    def font(size): return ImageFont.truetype(str(font_file), size)
    im=Image.new('RGB',(1280,896),'white'); d=ImageDraw.Draw(im)
    left,right,top,bottom=110,1225,125,755
    d.text((left,35),'Product measurements',font=font(28),fill='black')
    colors={'Alpha':(63,110,156),'Beta':(184,112,48)}
    for j,(name,color) in enumerate(colors.items()):
        x=840+j*170; d.rectangle([x,55,x+24,79],fill=color)
        d.text((x+35,50),name,font=font(23),fill='black')
    for val in range(0,101,10):
        y=bottom-(bottom-top)*val/100
        d.line((left,y,right,y),fill=(215,215,215),width=1)
        d.text((left-15,y),str(val),font=font(21),fill='black',anchor='rm')
    world=task['world']; n=len(world['categories']); space=(right-left)/n; bw=space*.28
    for i,name in enumerate(world['categories']):
        center=left+space*(i+.5)
        for j,series in enumerate(('Alpha','Beta')):
            x=center+(j-.5)*bw*1.12; y=bottom-(bottom-top)*world['series'][series][i]/100
            d.rectangle((x-bw/2,y,x+bw/2,bottom),fill=colors[series])
        d.text((center,bottom+25),name,font=font(22),fill='black',anchor='mt')
    d.line((left,top,left,bottom,right,bottom),fill='black',width=2)
    d.text((left,825),'Counts   |   Read categories in printed left-to-right order',font=font(22),fill='black')
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); im.save(output)
