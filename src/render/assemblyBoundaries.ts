import * as T from 'three';
import type {Assembly} from '../core/assemblies.js';
import type {TensorPlane} from '../core/layout.js';

/** Open corner frames identify ownership without enclosing tensors in opaque boxes.
 * All idle frames share one draw; selection uses one additional draw.
 */
export class AssemblyBoundaries {
    readonly bounds=new Map<string,T.Box3>();
    readonly idle:T.LineSegments;
    readonly selected:T.LineSegments;
    active?:string;
    constructor(parent:T.Group,readonly assemblies:Assembly[],planes:Map<string,TensorPlane>,decorations=new Map<string,T.Object3D>()){
        const positions:number[]=[],colors:number[]=[];
        for(const a of assemblies){
            const box=new T.Box3();
            for(const id of a.owned){const p=planes.get(id);if(!p)continue;
                box.expandByPoint(new T.Vector3(p.center[0]-p.width/2,p.center[1]-.2,p.center[2]-p.depth/2));
                box.expandByPoint(new T.Vector3(p.center[0]+p.width/2,p.center[1]+p.height,p.center[2]+p.depth/2));
            }
            for(const id of a.nodes){const visual=decorations.get(id);if(visual){visual.updateWorldMatrix(true,true);box.expandByObject(visual);}}
            if(box.isEmpty())continue;
            box.expandByScalar(.3);this.bounds.set(a.id,box);
            const points=this.corners(box);positions.push(...points);
            const c=new T.Color(a.color);for(let i=0;i<points.length/3;i++)colors.push(c.r,c.g,c.b);
        }
        const geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new T.Float32BufferAttribute(colors,3));
        this.idle=new T.LineSegments(geometry,new T.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.28,depthWrite:false}));
        this.selected=new T.LineSegments(new T.BufferGeometry(),new T.LineBasicMaterial({color:'#d8f8ff',transparent:true,opacity:.9,depthWrite:false}));
        this.selected.visible=false;parent.add(this.idle,this.selected);
    }
    private corners(b:T.Box3){
        const s=b.getSize(new T.Vector3()),length=[Math.min(.8,s.x*.23),Math.min(1,s.y*.35),Math.min(.8,s.z*.23)],points:number[]=[];
        for(const x of [0,1])for(const y of [0,1])for(const z of [0,1]){
            const bits=[x,y,z],p=[x?b.max.x:b.min.x,y?b.max.y:b.min.y,z?b.max.z:b.min.z];
            for(let axis=0;axis<3;axis++){const q=[...p];q[axis]+=(bits[axis]?-1:1)*length[axis];points.push(...p,...q);}
        }
        // Two rear rails make multi-plane ownership legible without drawing across data.
        points.push(b.min.x,b.min.y,b.min.z,b.min.x,b.max.y,b.min.z,b.max.x,b.min.y,b.min.z,b.max.x,b.max.y,b.min.z);
        return points;
    }
    update(id?:string){
        if(this.active===id)return;this.active=id;
        const box=id?this.bounds.get(id):undefined;this.selected.visible=!!box;
        (this.idle.material as T.LineBasicMaterial).opacity=id?.12:.28;
        this.selected.geometry.dispose();this.selected.geometry=new T.BufferGeometry();
        if(box)this.selected.geometry.setAttribute('position',new T.Float32BufferAttribute(this.corners(box),3));
    }
}
