import * as T from 'three';
import {RoundedBoxGeometry} from 'three/addons/geometries/RoundedBoxGeometry.js';
import type {TensorPlane} from '../core/layout.js';

/** Opaque machined rims support transparent tensors. Never used as scalar cells.
 * One instanced draw for every rim in the scope; no per-tensor material or light.
 */
export class MetalScaffold {
    readonly mesh:T.InstancedMesh;
    private entries:{id:string;color:T.Color}[]=[];
    private selection='';
    constructor(root:T.Group,planes:Map<string,TensorPlane>){
        const grain=new Uint8Array(64*64*4);
        for(let y=0;y<64;y++)for(let x=0;x<64;x++){
            const i=(y*64+x)*4,v=190+((y*37+x%3*7)%53);
            grain[i]=grain[i+1]=grain[i+2]=v;grain[i+3]=255;
        }
        const roughness=new T.DataTexture(grain,64,64,T.RGBAFormat);
        roughness.wrapS=roughness.wrapT=T.RepeatWrapping;
        roughness.magFilter=T.LinearFilter;roughness.minFilter=T.LinearMipmapLinearFilter;
        roughness.generateMipmaps=true;roughness.needsUpdate=true;
        const material=new T.MeshStandardMaterial({color:'#ffffff',metalness:.94,roughness:.36,
            roughnessMap:roughness,envMapIntensity:1.3});
        const geometry=new RoundedBoxGeometry(1,1,1,1,.12);
        this.mesh=new T.InstancedMesh(geometry,material,planes.size*8);
        this.mesh.userData.metalScaffold=true;
        this.mesh.instanceMatrix.setUsage(T.StaticDrawUsage);
        const transform=new T.Object3D();let index=0;
        const put=(id:string,color:T.Color,x:number,y:number,z:number,w:number,h:number,d:number)=>{
            transform.position.set(x,y,z);transform.scale.set(w,h,d);transform.updateMatrix();
            this.mesh.setMatrixAt(index,transform.matrix);this.mesh.setColorAt(index++,color);
            this.entries.push({id,color});
        };
        for(const p of planes.values()){
            const [x,y,z]=p.center,w=Math.max(.8,p.width)+.18,d=Math.max(.5,p.depth)+.18;
            const rim=Math.min(.18,Math.max(.095,Math.min(w,d)*.065));
            const silver=new T.Color(p.tensor.role==='parameter'?'#cbbb9f':'#b8c4cd');
            const graphite=new T.Color('#303e48');
            for(const side of [-1,1]){
                // A dark lower flange and bright bevel create a readable thickness.
                put(p.tensor.id,graphite,x,y-.25,z+side*d/2,w+.12,.14,rim+.06);
                put(p.tensor.id,silver,x,y-.16,z+side*d/2,w,.12,rim);
                put(p.tensor.id,graphite,x+side*w/2,y-.25,z,rim+.06,.14,d+.12);
                put(p.tensor.id,silver,x+side*w/2,y-.16,z,rim,.12,d);
            }
        }
        this.mesh.instanceMatrix.needsUpdate=true;if(this.mesh.instanceColor)this.mesh.instanceColor.needsUpdate=true;
        this.mesh.computeBoundingSphere();root.add(this.mesh);
    }
    update(members:Set<string>,selected:boolean){
        const key=selected?[...members].sort().join('|'):'';
        if(key===this.selection)return;this.selection=key;
        const c=new T.Color();
        this.entries.forEach((e,i)=>{
            c.copy(e.color).multiplyScalar(selected&&!members.has(e.id)?.48:1);
            this.mesh.setColorAt(i,c);
        });
        if(this.mesh.instanceColor)this.mesh.instanceColor.needsUpdate=true;
    }
}
