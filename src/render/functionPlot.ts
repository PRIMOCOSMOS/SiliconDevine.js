import * as T from 'three';
import { numericColor } from './numericPalette.js';

/** A world-space function plot. Curve, axes and samples share exactly one transform. */
export class FunctionPlot {
    readonly group=new T.Group();
    readonly marker=new T.Mesh(new T.SphereGeometry(.075,12,8),new T.MeshBasicMaterial({color:'#effaff'}));
    readonly inputMarker=new T.Mesh(new T.BoxGeometry(.12,.12,.12),new T.MeshBasicMaterial({color:'#effaff'}));
    readonly transfer=new T.LineSegments(new T.BufferGeometry().setFromPoints(Array.from({length:4},()=>new T.Vector3())),new T.LineBasicMaterial({color:'#c2e6f0',transparent:true,opacity:.78}));
    readonly extent:number;
    readonly yExtent:number;
    readonly halfWidth=1.35;
    readonly halfHeight:number;
    readonly points:T.Vector3[];
    constructor(parent:T.Group,position:[number,number,number],curve:(x:number)=>number,extent:number,gap:number){
        this.extent=Math.max(.5,extent);
        this.halfHeight=Math.min(1.1,Math.max(.38,gap/2-.45));
        const samples=Array.from({length:129},(_,i)=>-this.extent+i*this.extent/64);
        this.yExtent=Math.max(.5,...samples.map(x=>Math.abs(curve(x))).filter(Number.isFinite));
        this.points=samples.map(x=>new T.Vector3(x/this.extent*this.halfWidth,curve(x)/this.yExtent*this.halfHeight,0));
        const path=new T.CurvePath<T.Vector3>();
        for(let i=1;i<this.points.length;i++)if(Number.isFinite(this.points[i-1].y)&&Number.isFinite(this.points[i].y))path.add(new T.LineCurve3(this.points[i-1],this.points[i]));
        const tube=new T.TubeGeometry(path,128,.018,5,false),colors:number[]=[];
        // Tube vertices are colored by their actual ordinate, using the same scale as the sample marker.
        const vertices=tube.attributes.position;
        for(let i=0;i<vertices.count;i++){const c=numericColor(vertices.getY(i)/this.halfHeight*this.yExtent,this.yExtent,new T.Color());colors.push(c.r,c.g,c.b);}
        tube.setAttribute('color',new T.Float32BufferAttribute(colors,3));
        this.group.add(new T.Mesh(tube,new T.MeshBasicMaterial({vertexColors:true,transparent:true,opacity:.95})));
        const axes:T.Vector3[]=[new T.Vector3(-1.5,0,0),new T.Vector3(1.5,0,0),new T.Vector3(0,-this.halfHeight-.12,0),new T.Vector3(0,this.halfHeight+.12,0)];
        for(const sign of [-1,1]){axes.push(new T.Vector3(sign*this.halfWidth,-.05,0),new T.Vector3(sign*this.halfWidth,.05,0),new T.Vector3(-.05,sign*this.halfHeight,0),new T.Vector3(.05,sign*this.halfHeight,0));}
        this.group.add(new T.LineSegments(new T.BufferGeometry().setFromPoints(axes),new T.LineBasicMaterial({color:'#98afc1',transparent:true,opacity:.65})));
        const label=document.createElement('canvas');label.width=768;label.height=384;const c=label.getContext('2d')!;
        const labelHeight=(this.halfHeight+.32)*2;
        const px=(x:number)=>(x/3.45+.5)*768,py=(y:number)=>(.5-y/labelHeight)*384;
        c.fillStyle='#d1e3ef';c.font='26px Segoe UI';c.textAlign='center';
        const fmt=(v:number)=>Number(v.toPrecision(3)).toString();
        c.fillText('x',px(1.6),py(0)+8);c.fillText('f(x)',px(.3),py(this.halfHeight+.15));c.fillText('0',px(-.1),py(-.16));
        c.font='22px Segoe UI';c.fillText(fmt(-this.extent),px(-this.halfWidth),py(-.17));c.fillText(fmt(this.extent),px(this.halfWidth),py(-.17));
        c.textAlign='right';c.fillText(fmt(this.yExtent),px(-.09),py(this.halfHeight)+6);c.fillText(fmt(-this.yExtent),px(-.09),py(-this.halfHeight)+6);
        const texture=new T.CanvasTexture(label);texture.colorSpace=T.SRGBColorSpace;
        const caption=new T.Mesh(new T.PlaneGeometry(3.45,(this.halfHeight+.32)*2),new T.MeshBasicMaterial({map:texture,transparent:true,depthWrite:false,side:T.DoubleSide}));caption.position.z=.015;
        this.group.add(caption,this.marker,this.inputMarker,this.transfer);this.group.position.set(...position);parent.add(this.group);
    }
    update(x:number,y:number,active:boolean,showIdle:boolean){
        this.group.visible=active||showIdle;
        const valid=active&&Number.isFinite(x)&&Number.isFinite(y);
        this.marker.visible=this.inputMarker.visible=this.transfer.visible=valid;
        if(!valid)return;
        const px=x/this.extent*this.halfWidth,py=y/this.yExtent*this.halfHeight;
        this.inputMarker.position.set(px,0,.04);this.marker.position.set(px,py,.04);
        numericColor(x,this.extent,(this.inputMarker.material as T.MeshBasicMaterial).color);
        numericColor(y,this.yExtent,(this.marker.material as T.MeshBasicMaterial).color);
        const p=this.transfer.geometry.attributes.position;p.setXYZ(0,px,0,.025);p.setXYZ(1,px,py,.025);p.setXYZ(2,px,py,.025);p.setXYZ(3,0,py,.025);p.needsUpdate=true;
    }
}
