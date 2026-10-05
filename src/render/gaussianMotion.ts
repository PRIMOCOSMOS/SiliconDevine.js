import * as T from 'three';
import {coordinates,valueAt,type Operation,type Tensor} from '../core/model.js';
import type {TensorPlane} from '../core/layout.js';
import {numericColor} from './numericPalette.js';

/** Analytic density morph plus the captured epsilon, never a browser-generated sample. */
export class GaussianMotion {
    private group=new T.Group();
    private points=new Float32Array(65*3);
    private curve:T.InstancedMesh;
    private dummy=new T.Object3D();
    private marker=new T.Mesh(new T.BoxGeometry(.23,.23,.23),new T.MeshBasicMaterial({color:'#dce9ff'}));
    private stem=new T.Line(new T.BufferGeometry().setFromPoints([new T.Vector3(),new T.Vector3()]),new T.LineBasicMaterial({color:'#d3c2ff',transparent:true,opacity:.75}));
    state?:{index:number;coordinate:number[];mean:number;std:number;epsilon:number;sample:number;phase:number};
    constructor(parent:T.Group){
        this.curve=new T.InstancedMesh(new T.CylinderGeometry(1,1,1,6),new T.MeshBasicMaterial({color:'#c5b8ff',transparent:true,opacity:.92,depthWrite:false}),64);
        const axis=new T.Line(new T.BufferGeometry().setFromPoints([new T.Vector3(-2.2,0,0),new T.Vector3(2.2,0,0)]),new T.LineBasicMaterial({color:'#6c87b0',transparent:true,opacity:.5}));
        this.group.add(this.curve,this.marker,this.stem,axis);parent.add(this.group);this.group.visible=false;
        this.curve.frustumCulled=false;this.stem.frustumCulled=false;
    }
    update(n:Operation|undefined,out:TensorPlane|undefined,index:number,phase:number,tensors:Map<string,Tensor>){
        this.group.visible=false;this.state=undefined;
        const unit=n?.attrs?.gaussian as Record<string,string>|undefined;
        if(!unit||!out)return;
        const mu=valueAt(tensors.get(unit.mean)!,index),sigma=valueAt(tensors.get(unit.std)!,index),epsilon=valueAt(tensors.get(unit.noise)!,index),z=valueAt(tensors.get(unit.sample)!,index);
        if(![mu,sigma,epsilon,z].every(Number.isFinite)||sigma<=0)return;
        const blend=.5-.5*Math.cos(phase*Math.PI*2),center=mu*blend,width=1+(sigma-1)*blend;
        const extent=Math.max(4,Math.abs(mu)+4*sigma,Math.abs(epsilon),Math.abs(z)),scale=2.1/extent;
        this.group.position.set(out.center[0],out.center[1]+.85,out.center[2]+out.depth/2+.45);
        this.group.visible=true;
        const densityScale=.95*Math.min(1,sigma);
        for(let j=0;j<65;j++){
            const standard=-3.5+j*7/64;
            this.points[j*3]=(center+standard*width)*scale;
            this.points[j*3+1]=Math.exp(-.5*standard*standard)/width*densityScale;
            this.points[j*3+2]=0;
        }
        for(let j=0;j<64;j++){
            const a=new T.Vector3().fromArray(this.points,j*3),b=new T.Vector3().fromArray(this.points,(j+1)*3),d=b.clone().sub(a);
            this.dummy.position.copy(a).add(b).multiplyScalar(.5);this.dummy.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),d.clone().normalize());this.dummy.scale.set(.018,d.length(),.018);this.dummy.updateMatrix();this.curve.setMatrixAt(j,this.dummy.matrix);
        }
        this.curve.instanceMatrix.needsUpdate=true;
        const current=center+width*epsilon,x=current*scale,height=Math.exp(-.5*epsilon*epsilon)/width*densityScale;
        this.marker.position.set(x,height,0);
        numericColor(current,extent,(this.marker.material as T.MeshBasicMaterial).color);
        const a=this.stem.geometry.attributes.position as T.BufferAttribute;a.setXYZ(0,x,0,0);a.setXYZ(1,x,height,0);a.needsUpdate=true;
        this.state={index,coordinate:coordinates(index,tensors.get(unit.sample)!.shape as number[]),mean:mu,std:sigma,epsilon,sample:z,phase:blend};
    }
}
