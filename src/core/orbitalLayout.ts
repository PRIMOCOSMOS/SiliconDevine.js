import type {Model,Operation} from './model.js';
import type {TensorPlane} from './layout.js';

type Visual={family?:string;branch?:string;stages?:{tensor:string}[]};
/** Arrange proven independent sibling calls around a dependency-ranked trunk.
 * No model names, encoder/decoder fields, or modality counts enter this policy.
 */
export function orbitalLayout(model:Model,planes:Map<string,TensorPlane>){
 const meta=(n:Operation)=>n.attrs?.compositionVisual as Visual|undefined;
 const families=new Map<string,Operation[]>();
 for(const n of model.nodes){const v=meta(n);if(v?.family&&v.branch)families.set(v.family,[...families.get(v.family)??[],n]);}
 const parallel=new Set([...families.values()].filter(ns=>ns.length>1).flat().map(n=>n.id));
 if(!parallel.size)return false;
 const branchKeys=[...new Set(model.nodes.filter(n=>parallel.has(n.id)).map(n=>meta(n)!.branch!))];
 const width=Math.max(4,...[...planes.values()].map(p=>Math.max(p.width,p.depth)));
 const radius=Math.max(12,(width+5)/(2*Math.sin(Math.PI/Math.max(3,branchKeys.length))));
 const producer=new Map(model.nodes.flatMap(n=>n.outputs.map(id=>[id,n] as const)));
 const ends=new Map<string,number>(),starts=new Map<string,number>(),visiting=new Set<string>();
 const heights=new Map(model.nodes.map(n=>[n.id,Math.max(1,meta(n)?.stages?.length??1)]));
 function rank(n:Operation):number{
  if(ends.has(n.id))return ends.get(n.id)!;
  if(visiting.has(n.id))return 0; // Validation handles cycles; remain finite on partial scopes.
  visiting.add(n.id);
  const deps=n.inputs.map(id=>producer.get(id)).filter((q):q is Operation=>!!q&&q.id!==n.id);
  const start=Math.max(0,...deps.map(rank));starts.set(n.id,start);const end=start+heights.get(n.id)!;
  ends.set(n.id,end);visiting.delete(n.id);return end;
 }
 model.nodes.forEach(rank);
 // Family members share entry rank, but never cross an actual dependency.
 for(const ns of families.values())if(ns.length>1){const start=Math.max(...ns.map(n=>starts.get(n.id)!));
  for(const n of ns)starts.set(n.id,start);
 }
 // Recompute dependency ends after sibling alignment so downstream mergers stay below.
 ends.clear();
 function alignedEnd(n:Operation):number{
  if(ends.has(n.id))return ends.get(n.id)!;
  const deps=n.inputs.map(id=>producer.get(id)).filter((q):q is Operation=>!!q&&q.id!==n.id);
  const start=Math.max(starts.get(n.id)??0,...deps.map(alignedEnd));starts.set(n.id,start);
  const end=start+heights.get(n.id)!;ends.set(n.id,end);return end;
 }
 model.nodes.forEach(alignedEnd);
 const gap=Math.max(2.6,...[...planes.values()].map(p=>p.height+1.8));
 for(const n of model.nodes){
  const v=meta(n),orbital=parallel.has(n.id),a=2*Math.PI*branchKeys.indexOf(v?.branch??'')/branchKeys.length+Math.PI/12;
  const x=orbital?Math.cos(a)*radius:0,z=orbital?Math.sin(a)*radius:0;
  const stageIds=v?.stages?.map(s=>s.tensor)??[];
  const extras=n.outputs.filter(id=>!stageIds.includes(id));
  const old=planes.get(n.outputs[0])?.center.slice()??[0,0,0];
  stageIds.forEach((id,i)=>{const p=planes.get(id);if(p)p.center=[x,-(starts.get(n.id)!+i+1)*gap,z];});
  const ps=extras.map(id=>planes.get(id)).filter((p):p is TensorPlane=>!!p);
  const w=Math.max(1,...ps.map(p=>p.width)),d=Math.max(1,...ps.map(p=>p.depth));
  // Exterior ports live beside the final stage; keep them distinct from that activation.
  ps.forEach((p,i)=>p.center=[x+(i%2-(Math.min(ps.length,2)-1)/2)*(w+1),-ends.get(n.id)!*gap,
    z+(Math.floor(i/2)+(stageIds.length?1:0))*(d+1)]);
  const center=planes.get(n.outputs[0])?.center;
  if(center)for(const p of planes.values())if(p.owner===n.id&&!n.outputs.includes(p.tensor.id)&&!model.inputs.includes(p.tensor.id))
    p.center=p.center.map((q,i)=>q+center[i]-old[i]) as [number,number,number];
  if(orbital)for(const id of n.inputs){const p=planes.get(id);
    if(p&&model.inputs.includes(id)&&model.nodes.filter(q=>q.inputs.includes(id)).length===1)p.center=[x,-starts.get(n.id)!*gap,z];
  }
 }
 return true;
}
