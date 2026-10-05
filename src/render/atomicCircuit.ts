import * as T from 'three';
import { layoutModel, type TensorPlane } from '../core/layout.js';
import { valueAt, type Model } from '../core/model.js';
import { operatorVisual } from '../core/operators.js';
import { createCrystalTensor } from './crystalPrimitives.js';
import { connectionFabric, type Connection } from './operationMotion.js';
import { MechanismMotion } from './mechanismMotion.js';

/** The same coordinate layout, crystals, scalar dependencies, signed weights and
 * calculation motion as the full viewer. Only one bounded inset is alive at once. */
export class AtomicCircuit {
    readonly group = new T.Group();
    readonly planes;
    readonly tensors;
    private fabrics = new Map<string, ReturnType<typeof connectionFabric>>();
    private mechanism: MechanismMotion;
    private crystals: {plane:TensorPlane; crystal:ReturnType<typeof createCrystalTensor>; values:number[]}[]=[];
    readonly cells: number;
    readonly connections: number;
    constructor(readonly model:Model, parent:T.Group, center:[number,number,number]) {
        this.planes=layoutModel(model,24);
        // Preview has no in-graph text, so remove the full viewer's label spacing.
        // Translate entire tensor planes; never squash scalar cubes or tensor axes.
        const gap=Math.min(1,Math.max(.3,...[...this.planes.values()].map(p=>(p.height+.6)/4.1)));
        for(const p of this.planes.values()){
            const delta=p.center[1]*(gap-1);p.center[1]+=delta;
            p.positions=p.positions.map(v=>[v[0],v[1]+delta,v[2]]);
        }
        this.tensors=new Map(model.tensors.map(t=>[t.id,t]));
        let cells=0,connections=0;
        for(const p of this.planes.values()) {
            const values=p.indices.map(i=>valueAt(p.tensor,i));
            const crystal=createCrystalTensor(this.group,p.indices.length,.26,{parameter:p.tensor.role==='parameter',valueEdges:true,bodyOpacity:.3,edgeOpacity:.65});
            crystal.update(values,p.positions);this.crystals.push({plane:p,crystal,values});cells+=p.indices.length;
        }
        for(const n of model.nodes) {
            const links:Connection[]=[];
            for(const id of n.outputs){const out=this.planes.get(id)!;
                out.indices.forEach((index,j)=>{
                    for(const [term,d] of operatorVisual(n.op).dependencies(n,out.tensor,index,this.tensors).entries()){
                        if(connections>=8192)break;
                        const source=this.planes.get(d.tensor),k=source?.indices.indexOf(d.index)??-1;
                        if(!source||k<0)continue;
                        links.push({from:source.positions[k],to:out.positions[j],weight:d.weight,
                            weightScale:d.parameter?this.tensors.get(d.parameter)?.stats?.absmax??1:1,output:j,term,parameterIndex:d.parameterIndex});connections++;
                    }
                });
            }
            this.fabrics.set(n.id,connectionFabric(this.group,links));
        }
        this.mechanism=new MechanismMotion(this.group);
        const box=new T.Box3().setFromObject(this.group),size=box.getSize(new T.Vector3()),mid=box.getCenter(new T.Vector3());
        // Fit in the existing functional-unit envelope; expansion uses unscaled layout.
        this.group.scale.setScalar(Math.min(4.3/Math.max(4.3,size.x),2.7/Math.max(2.7,size.y),1.9/Math.max(1.9,size.z)));
        this.group.position.set(center[0]-mid.x*this.group.scale.x,center[1]+1.65-mid.y*this.group.scale.y,center[2]-mid.z*this.group.scale.z);
        parent.add(this.group);this.cells=cells;this.connections=connections;
    }
    update(time:number) {
        const nodes=this.model.nodes.filter(n=>!['reshape','transpose','permute','slice'].includes(n.op));
        const active=nodes[Math.floor(time/2.4)%Math.max(1,nodes.length)],phase=(time*.4)%1;
        if(!active)return;
        const out=this.planes.get(active.outputs[0])!,index=Math.floor(phase*out.indices.length),term=time*2%8;
        const width=Number(this.tensors.get(active.parameters?.weight??'')?.shape.at(-1)??1);
        const parameter=active.op==='linear'?(out.indices[index]%Number(out.tensor.shape.at(-1)))*width+term%width:-1;
        for(const [id,f] of this.fabrics)f.update(time*.4,index,term,id===active.id?1:0,.45,id===active.id?parameter:-1);
        for(const {plane,crystal,values} of this.crystals){
            const selected=plane.tensor.id===active.parameters?.weight?plane.indices.indexOf(Math.floor(parameter)):plane.tensor.id===out.tensor.id?index:-1;
            crystal.update(values,plane.positions,{active:selected>=0?[selected]:[],focus:selected});
        }
        this.mechanism.update(active,out,out.indices[index],phase,this.planes,this.tensors);
    }
    dispose(){
        this.group.removeFromParent();
        const geometries=new Set<T.BufferGeometry>(),materials=new Set<T.Material>();
        this.group.traverse(o=>{const m=o as T.Mesh;if(m.geometry)geometries.add(m.geometry);if(m.material)(Array.isArray(m.material)?m.material:[m.material]).forEach(v=>materials.add(v));});
        geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());
    }
}
