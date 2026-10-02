"""Build bundled launcher geometry; no image library needed at application runtime."""
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import math
from fontTools.ttLib import TTFont

root=Path(__file__).resolve().parents[1]
assets=root/'assets'
bg='#06080d'
for size in (400,600,800):
    n=size*3
    im=Image.new('RGB',(n,n),bg)
    glow=Image.new('RGB',(n,n),bg)
    gd=ImageDraw.Draw(glow)
    c=n/2;r=n*.4
    gd.ellipse((c-r*.7,c-r*.7,c+r*.7,c+r*.7),fill='#102034')
    glow=glow.filter(ImageFilter.GaussianBlur(n*.12))
    im=glow
    d=ImageDraw.Draw(im)
    for k,(angle,flat) in enumerate(((.12,.34),(.95,.46),(-.9,.46))):
        pts=[]
        for i in range(289):
            a=i*math.tau/288;x,y=r*math.cos(a),r*flat*math.sin(a)
            pts.append((c+x*math.cos(angle)-y*math.sin(angle),c+x*math.sin(angle)+y*math.cos(angle)))
        d.line(pts,fill=('#526f95','#8ca8cb','#c1d5ee')[k],width=max(3,round(n*.003)))
    q=r*.29
    for coords,fill in [([(c,c-q),(c+q,c-q*.4),(c,c+q*.2),(c-q,c-q*.4)],'#4a617e'), ([(c-q,c-q*.4),(c,c+q*.2),(c,c+q*1.2),(c-q,c+q*.6)],'#16283f'), ([(c,c+q*.2),(c+q,c-q*.4),(c+q,c+q*.6),(c,c+q*1.2)],'#29435e')]:
        d.polygon(coords,fill=fill)
        d.line(coords+[coords[0]],fill='#c3d9f4',width=max(3,round(n*.003)))
    im.resize((size,size),Image.Resampling.LANCZOS).save(assets/f'orbit-{size}.png')

# Opaque corner background avoids Tk alpha/clam composition seams.
colors={'normal':['#161e2b','#253a4c','#344f62','#10151e'],'primary':['#dce9ff','#ffffff','#a8c8f5','#10151e'],'quiet':[bg,'#182334','#253a4c',bg]}
for name,values in colors.items():
    for state,color in zip(('idle','active','pressed','disabled'),values):
        im=Image.new('RGB',(144,144),bg)
        ImageDraw.Draw(im).rounded_rectangle((0,0,143,143),radius=36,fill=color)
        im.resize((48,48),Image.Resampling.LANCZOS).save(assets/f'button-{name}-{state}.png')
    im=Image.new('RGB',(144,144),bg)
    ImageDraw.Draw(im).rounded_rectangle((3,3,140,140),radius=33,fill=values[0],outline='#86a9d3',width=4)
    im.resize((48,48),Image.Resampling.LANCZOS).save(assets/f'button-{name}-focus.png')

font=TTFont(assets/'ZCOOLQingKeHuangYou-Regular.ttf')
font.flavor='woff2'
(root/'demo/public/fonts').mkdir(parents=True,exist_ok=True)
font.save(root/'demo/public/fonts/ZCOOLQingKeHuangYou.woff2')
print('Built antialiased brand assets and bundled Chinese display font.')
