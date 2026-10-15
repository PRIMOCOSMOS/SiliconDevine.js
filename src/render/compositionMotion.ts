import * as T from 'three';
import {valueAt,type Model} from '../core/model.js';
import type {TensorPlane} from '../core/layout.js';
import {industrialTheme as theme} from './industrialTheme.js';

export const branchColors=theme.branches;
/** Discrete registration arcs express module ownership; crystal cells remain data.
 * All tower rails, arc segments and ticks share one draw and one animation clock.
 */
export class CompositionMotion {
 private ids=new Map<string,number>();
 private material?:T.ShaderMaterial;
 constructor(root:T.Group,model:Model,planes:Map<string,TensorPlane>){
  const positions:number[]=[],colors:number[]=[],owners:number[]=[],phases:number[]=[];
  const branches=[...new Set(model.nodes.map(n=>(n.attrs?.compositionVisual as any)?.branch).filter(Boolean))];
  for(const n of model.nodes){
   const v=n.attrs?.compositionVisual as {kind:string;branch?:string;stages?:{tensor:string}[];distribution?:{mean:string;std:string}}|undefined;
   if(!v)continue;
   const color=v.branch?branchColors[branches.indexOf(v.branch)%branchColors.length]:theme.frame;
   const ps=(v.stages?.length?v.stages.map(s=>planes.get(s.tensor)):n.outputs.map(id=>planes.get(id))).filter((p):p is TensorPlane=>!!p);
   if(!ps.length)continue;
   const owner=this.ids.size;this.ids.set(n.id,owner);
   const c=new T.Color(theme.frame).lerp(new T.Color(color),.4);
   const line=(a:number[],b:number[],phase:number)=>{positions.push(...a,...b);colors.push(c.r,c.g,c.b,c.r,c.g,c.b);owners.push(owner,owner);phases.push(phase,phase);};
   if(v.stages?.length){
    const r=Math.max(...ps.map(p=>Math.hypot(p.width,p.depth)))/2+.65;
    const first=ps[0],last=ps.at(-1)!;
    for(const side of [-1,1])line([first.center[0]+side*r,first.center[1]+.6,first.center[2]],
      [last.center[0]+side*r,last.center[1]-.6,last.center[2]],0);
    ps.forEach((p,i)=>{
     const [x,y0,z]=p.center,y=y0-.25,phase=i/ps.length;
     const point=(a:number,radius=r)=>[x+Math.cos(a)*radius,y,z+Math.sin(a)*radius];
     for(let j=0;j<64;j++)if(j%16<12)line(point(j/64*Math.PI*2),point((j+1)/64*Math.PI*2),phase);
     for(let j=0;j<8;j++){const a=j/8*Math.PI*2;line(point(a,r+.15),point(a,r+(j%2===0?.6:.35)),phase);}
    });
   }
   if(v.distribution){
    const mu=model.tensors.find(t=>t.id===v.distribution!.mean),sd=model.tensors.find(t=>t.id===v.distribution!.std);
    const mean=mu?valueAt(mu,0):NaN,std=sd?valueAt(sd,0):NaN;
    if(!Number.isFinite(mean)||!Number.isFinite(std)||std<=0)continue;
    const p=ps[0],x=p.center[0],y=p.center[1]+.45,z=p.center[2];
    const curve=new T.Line(new T.BufferGeometry().setFromPoints(Array.from({length:49},(_,i)=>{
      const a=i/48*6-3;return new T.Vector3(x+a*.65,y+Math.exp(-a*a/2)*2.1,z);
    })),new T.LineBasicMaterial({color,transparent:true,opacity:.85,depthWrite:false}));
    curve.userData.distribution={mean,variance:std*std,coordinate:0,domain:[mean-3*std,mean+3*std]};root.add(curve);
    line([x-1.95,y,z],[x+1.95,y,z],0);line([x,y,z],[x,y+2.1,z],0);
    for(let j=-3;j<=3;j++)line([x+j*.65,y-.1,z],[x+j*.65,y+.1,z],0);
   }
  }
  if(!positions.length)return;
  const geometry=new T.BufferGeometry();
  for(const [name,array,size] of [['position',positions,3],['color',colors,3],['ownerId',owners,1],['phase',phases,1]] as const)
    geometry.setAttribute(name,new T.Float32BufferAttribute(array,size));
  this.material=new T.ShaderMaterial({transparent:true,depthWrite:false,uniforms:{time:{value:0},selectedOwner:{value:-1}},
   vertexShader:`attribute vec3 color;attribute float ownerId;attribute float phase;varying vec3 tint;varying float owner;varying float progress;
    void main(){tint=color;owner=ownerId;progress=phase;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
   fragmentShader:`uniform float time;uniform float selectedOwner;varying vec3 tint;varying float owner;varying float progress;
    void main(){float selected=1.-step(.25,abs(owner-selectedOwner));float wave=pow(.5+.5*sin(time*2.-progress*6.283),3.);
    vec3 c=mix(tint,vec3(.90,.92,.52),selected*.48);gl_FragColor=vec4(c,.32+selected*(.22+.3*wave));}`});
  const marks=new T.LineSegments(geometry,this.material);marks.userData.architectureBatch=true;root.add(marks);
 }
 update(time:number,active:string){if(this.material){this.material.uniforms.time.value=time;this.material.uniforms.selectedOwner.value=this.ids.get(active)??-1;}}
}
