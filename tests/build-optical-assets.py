"""Code-generated optical branding and DPI-ready native control surfaces."""
from pathlib import Path
from PIL import Image, ImageDraw
import math
root=Path(__file__).resolve().parents[1]
assets=root/'assets'
art=Image.open(assets/'keynote-source.png').convert('RGBA')
for size in (400,600,800):
    im=Image.new('RGB',(size,size));px=im.load()
    for y in range(size):
        for x in range(size):
            r=((x/size-.48)/.32)**2+((y/size-.52)/.29)**2
            halo=math.exp(-r*2)*.7
            px[x,y]=(round(6+halo*13),round(8+halo*27),round(13+halo*45))
    overlay=art.resize((size,size),Image.Resampling.LANCZOS);im.paste(overlay,(0,0),overlay)
    im.save(assets/f'keynote-{size}.png')
art.resize((800,800),Image.Resampling.LANCZOS).save(root/'demo/optical-core.png')
colors={'normal':((35,44,57),(18,24,33)),'primary':((249,252,255),(193,212,229)),'quiet':((6,8,13),(6,8,13))}
for kind,(top,bottom) in colors.items():
    for state in ('idle','active','pressed','disabled','focus'):
        n=144;im=Image.new('RGB',(n,n),'#06080d');surface=Image.new('RGB',(n,n));d=ImageDraw.Draw(surface)
        # A flat interior survives Tk's nine-slice stretch without gradient seams.
        t=.4 if state=='active' else .8 if state=='pressed' else .6
        color=tuple(round(a*(1-t)+b*t) for a,b in zip(top,bottom))
        if state=='disabled':color=(16,21,30)
        d.rectangle((0,0,n,n),fill=color)
        mask=Image.new('L',(n,n));ImageDraw.Draw(mask).rounded_rectangle((0,0,n-1,n-1),radius=36,fill=255)
        im.paste(surface,(0,0),mask);d=ImageDraw.Draw(im)
        if state=='focus':d.rounded_rectangle((3,3,n-4,n-4),radius=33,outline='#c3ddf4',width=4)
        elif kind=='normal' and state!='disabled':d.rounded_rectangle((1,1,n-2,n-2),radius=35,outline='#46566a',width=2)
        im.resize((48,48),Image.Resampling.LANCZOS).save(assets/f'button-{kind}-{state}.png')
print('Built optical artwork at 3 densities and five native button states.')
