import * as T from 'three';
import { createArrowStream, createCrystalTensor } from './crystalPrimitives.js';
import type { TensorPlane } from '../core/layout.js';
import type { Model, Operation } from '../core/model.js';

/** Semantic colors in structure mode. Numerical mode retains the signed value palette. */
export const structureColors:Record<string,string>={repeat:'#a5b9f5',attention:'#6de0da',experts:'#ddb879',router:'#e3a2bb',recurrent:'#8ed4b1',norm:'#b7aacd',projection:'#8bbcdf',ffn:'#d2b281',gate:'#c5a4dc',merge:'#efe0ae',embedding:'#96cbbb',rope:'#b9a3e3'};
export class StructureMotion {
    private glyphs=new Map<string,T.LineSegments>();
    private circuits=new Map<string,[number,number,number][][]>();
    private flows:ReturnType<typeof createArrowStream>[];
    constructor(parent:T.Group,private nodes:Operation[],private planes:Map<string,TensorPlane>,scopes:NonNullable<Model['architecture']>['scopes']) {
        this.flows=Array.from({length:8},()=>createArrowStream(parent,'#defaf7',.07,3));
        for(const n of nodes) {
            const p=planes.get(n.outputs[0]);if(!p)continue;
            const points:T.Vector3[]=[];
            const segment=(a:number[],b:number[])=>points.push(new T.Vector3(...a as [number,number,number]),new T.Vector3(...b as [number,number,number]));
            const rectangle=(y:number,w=1.5,d=.8)=>{segment([-w,y,-d],[w,y,-d]);segment([w,y,-d],[w,y,d]);segment([w,y,d],[-w,y,d]);segment([-w,y,d],[-w,y,-d]);};
            const ring=(r:number,y:number,start=0,end=Math.PI*2)=>{for(let i=0;i<48;i++){const a=start+(end-start)*i/48,b=start+(end-start)*(i+1)/48;segment([r*Math.cos(a),y,r*Math.sin(a)],[r*Math.cos(b),y,r*Math.sin(b)]);}};
            const kind=String(n.attrs?.kind??'projection');
            const inner=scopes[String(n.attrs?.scopeRef)];
            if(inner){
                const nodes=inner.nodes.slice(0,10),producers=new Map(nodes.flatMap(n=>n.outputs.map(id=>[id,n.id]))),levels=new Map<string,number>();
                for(const child of nodes)levels.set(child.id,1+Math.max(0,...child.inputs.map(id=>levels.get(producers.get(id)??'')??0)));
                const max=Math.max(...levels.values()),locations=new Map<string,[number,number,number]>();
                for(const child of nodes){const level=levels.get(child.id)!,peers=nodes.filter(o=>levels.get(o.id)===level),j=peers.indexOf(child);locations.set(child.id,[(j-(peers.length-1)/2)*1.15,.4+(max-level)*2.5/Math.max(1,max-1),0]);}
                const paths:[number,number,number][][]=[];
                for(const child of nodes){const at=locations.get(child.id)!;for(const input of child.inputs){const from=locations.get(producers.get(input)??'');if(from){segment(from,at);paths.push([from,at].map(q=>[p.center[0]+q[0]*1.4,p.center[1]+q[1],p.center[2]+q[2]*1.25]));}}
                    const [x,y,z]=at,w=.29;segment([x-w,y,z-.15],[x+w,y,z-.15]);segment([x+w,y,z-.15],[x+w,y,z+.15]);segment([x+w,y,z+.15],[x-w,y,z+.15]);segment([x-w,y,z+.15],[x-w,y,z-.15]);
                }
                this.circuits.set(n.id,paths);
                const fragments=createCrystalTensor(parent,nodes.length*3,.17,{bodyOpacity:.3,edgeOpacity:.72});
                const positions=nodes.flatMap(child=>{const at=locations.get(child.id)!;return [-.16,0,.16].map(x=>[p.center[0]+at[0]*1.4+x,p.center[1]+at[1],p.center[2]] as [number,number,number]);});fragments.update(positions.map(()=>0),positions);
                nodes.forEach((child,i)=>{const c=new T.Color(structureColors[String(child.attrs?.kind)]??'#8bbcdf');for(let k=0;k<3;k++){fragments.body.setColorAt(i*3+k,c);fragments.edge.setColorAt(i*3+k,c);}});

                // One texture per module; labels are embedded next to their real internal steps.
                const canvas=document.createElement('canvas');canvas.width=768;canvas.height=640;const ctx=canvas.getContext('2d')!;
                ctx.fillStyle='#c8e4f0';ctx.font='46px Segoe UI';
                for(const child of nodes){const at=locations.get(child.id)!;const label=({norm:'RMSNorm',merge:'残差 +',attention:child.name.includes('MLA')?'MLA':'Attention',experts:'MoE',ffn:'SwiGLU',repeat:child.name.split(' · ')[0]} as Record<string,string>)[String(child.attrs?.kind)]??child.name.slice(0,9);ctx.fillText(label,390+at[0]*95,90+(2.9-at[1])*170,340);}
                if(n.attrs?.repeat){ctx.fillStyle='#f0d5a9';ctx.font='30px Segoe UI';ctx.fillText('× '+n.attrs.repeat+' · 独立实例',220,620);}
                const texture=new T.CanvasTexture(canvas);texture.colorSpace=T.SRGBColorSpace;
                const plate=new T.Mesh(new T.PlaneGeometry(5.6,3.7),new T.MeshBasicMaterial({map:texture,transparent:true,depthWrite:false,side:T.DoubleSide}));plate.position.set(p.center[0],p.center[1]+1.65,p.center[2]+.2);parent.add(plate);
            }

            if(inner) {}
            else if(kind==='repeat') { for(let i=0;i<4;i++)rectangle(.38+i*.25);segment([-1.5,.38,-.8],[-1.5,1.13,-.8]);segment([1.5,.38,.8],[1.5,1.13,.8]); }
            else if(kind==='experts'||kind==='router') { const count=kind==='router'?4:8;for(let i=0;i<count;i++){const x=(i-(count-1)/2)*.38;segment([0,1.3,0],[x,.6,.2]);segment([x-.12,.6,.08],[x+.12,.6,.08]);segment([x+.12,.6,.08],[x+.12,.6,.32]);segment([x+.12,.6,.32],[x-.12,.6,.32]);segment([x-.12,.6,.32],[x-.12,.6,.08]);} }
            else if(kind==='attention') { ring(1,.65);ring(.48,.65);for(let i=0;i<8;i++){const a=i*Math.PI/4;segment([Math.cos(a),.65,Math.sin(a)],[0,1.1,0]);} }
            else if(kind==='recurrent'||kind==='rope') {ring(1,.65,.2,Math.PI*1.9);segment([.95,.65,-.3],[.7,.65,-.4]);segment([.95,.65,-.3],[1,.65,-.57]);if(kind==='recurrent')rectangle(.95,.5,.5);}
            else if(kind==='norm') {ring(.8,.65);segment([-1.2,.65,0],[1.2,.65,0]);}
            else if(kind==='merge'||kind==='gate') {segment([-.85,1.2,0],[0,.5,0]);segment([.85,1.2,0],[0,.5,0]);ring(.22,.5);}
            else {rectangle(.65);for(let i=-2;i<=2;i++)segment([i*.48,.65,-.8],[i*.48,.65,.8]);}
            const mesh=new T.LineSegments(new T.BufferGeometry().setFromPoints(points),new T.LineBasicMaterial({color:structureColors[kind]??'#8bbcdf',transparent:true,opacity:.52,depthWrite:false}));
            mesh.scale.set(1.4,1,1.25);mesh.position.set(...p.center);parent.add(mesh);this.glyphs.set(n.id,mesh);
        }
    }
    update(active:string,time:number) {
        for(const [id,mesh] of this.glyphs)(mesh.material as T.LineBasicMaterial).opacity=id===active ? .78+.18*Math.sin(time*2):.45;
        const node=this.nodes.find(n=>n.id===active),out=node&&this.planes.get(node.outputs[0]);
        const inputs=node?[...node.inputs,...Object.values(node.parameters??{})].map(id=>this.planes.get(id)).filter((p):p is TensorPlane=>!!p):[];
        this.flows.forEach((f,i)=>{
            f.group.visible=false;if(!out||!node)return;
            const phase=(time*.25+i*.11)%1;
            const paths=this.circuits.get(node.id);
            if(paths?.length&&i>=inputs.length){const path=paths[(i-inputs.length+Math.floor(time*.3))%paths.length];f.update(path,phase,.85);return;}
            if(inputs[i]){f.update([inputs[i].center,out.center],phase);return;}
            const kind=String(node.attrs?.kind),slot=i-inputs.length;
            // Schematic functional motion has a fixed cost and carries no invented scores.
            const world=(x:number,y:number,z:number):[number,number,number]=>[out.center[0]+x*1.4,out.center[1]+y,out.center[2]+z*1.25];
            if(['recurrent','attention','rope'].includes(kind)&&slot===0){
                const points=Array.from({length:25},(_,k)=>world(Math.cos(k*Math.PI/12),.65,Math.sin(k*Math.PI/12)));
                f.update(points,phase,.7);
            }else if(['experts','router'].includes(kind)&&slot<4){
                const x=(slot-1.5)*.65;
                f.update([world(0,1.3,0),world(x,.6,.2)],phase,.35+.65*(.5+.5*Math.sin(time*1.5+slot)));
            }
        });
    }
}
