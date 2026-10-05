import {coordinates,flatIndex,valueAt,type Tensor} from './model.js';
import type {OperatorVisual,Dependency} from './operators.js';
const pair=(v:unknown,f=1):number[]=>Array.isArray(v)?v:[Number(v??f),Number(v??f)];
export function installAPIOperators(register:(name:string,visual:OperatorVisual)=>void){
    for(const shuffle of [true,false])register(shuffle?'pixel_shuffle':'pixel_unshuffle',{label:shuffle?'通道重排到空间':'空间重排到通道',color:'#8fbddd',formula:shuffle?'[…, C·r², H, W] → […, C, H·r, W·r]':'[…, C, H·r, W·r] → […, C·r², H, W]',detail:'exact',dependencies:(n,out,i,t)=>{
        const x=t.get(n.inputs[0])!,xs=x.shape as number[],c=coordinates(i,out.shape as number[]),r=Number(n.attrs?.factor??1),[ch,y,z]=c.slice(-3);
        const source=shuffle?[ch*r*r+(y%r)*r+z%r,Math.floor(y/r),Math.floor(z/r)]:[Math.floor(ch/(r*r)),y*r+Math.floor(ch%(r*r)/r),z*r+ch%r];
        return [{tensor:x.id,index:flatIndex([...c.slice(0,-3),...source],xs),weight:1}];
    }});
    const curves:Record<string,(x:number,a:Record<string,unknown>)=>number>={
        abs:Math.abs,
        relu6:x=>Math.min(6,Math.max(0,x)),hardsigmoid:x=>Math.max(0,Math.min(1,x/6+.5)),
        hardswish:x=>x*Math.max(0,Math.min(1,x/6+.5)),hardtanh:(x,a)=>Math.max(Number(a.min_val??-1),Math.min(Number(a.max_val??1),x)),
        softsign:x=>x/(1+Math.abs(x)),selu:x=>1.0507009873554805*(x>0?x:1.6732632423543772*Math.expm1(x)),
        celu:(x,a)=>Math.max(0,x)+Math.min(0,Number(a.alpha??1)*Math.expm1(x/Number(a.alpha??1))),
        mish:x=>x*Math.tanh(Math.max(0,x)+Math.log1p(Math.exp(-Math.abs(x)))),
        logsigmoid:x=>Math.min(0,x)-Math.log1p(Math.exp(-Math.abs(x))),
    };
    for(const [name,curve] of Object.entries(curves))register(name,{label:name,color:'#baa8ef',formula:`y = ${name}(x)`,detail:'exact',curve,dependencies:(n,_,i)=>[{tensor:n.inputs[0],index:i}]});
    register('prelu',{label:'可学习负半轴 PReLU',color:'#baa8ef',formula:'y = max(0,x) + a[c] min(0,x)',detail:'exact',dependencies:(n,out,i,t)=>{
        const w=t.get(n.parameters?.weight??'');if(!w)return [];
        const c=coordinates(i,out.shape as number[]),wi=Number(w.shape[0])===1?0:c[1]??0,x=t.get(n.inputs[0])!;
        return [{tensor:x.id,index:i,weight:valueAt(x,i)>=0?1:valueAt(w,wi),parameter:w.id,parameterIndex:wi}];
    }});
    register('log_softmax',{label:'对数 Softmax',color:'#baa8ef',formula:'yᵢ = xᵢ − log Σ exp(xⱼ)',detail:'exact',dependencies:(n,out,i)=>{
        const s=out.shape as number[],axis=(Number(n.attrs?.dim??-1)+s.length)%s.length,c=coordinates(i,s);
        return Array.from({length:s[axis]},(_,j)=>({tensor:n.inputs[0],index:flatIndex(c.map((v,k)=>k===axis?j:v),s)}));
    }});
    register('pad',{label:'边界填充 / 裁剪',color:'#8fbddd',formula:'按边界规则映射坐标；常量区域保持指定值',detail:'exact',dependencies:(n,out,i,t)=>{
        const x=t.get(n.inputs[0])!,s=x.shape as number[],c=coordinates(i,out.shape as number[]),pad=n.attrs?.pad as number[]??[],mode=n.attrs?.mode??'constant';
        for(let j=0;j<pad.length/2;j++){
            const axis=s.length-1-j,size=s[axis];c[axis]-=pad[2*j];
            if(c[axis]<0||c[axis]>=size){
                if(mode==='constant')return [];
                if(mode==='replicate')c[axis]=Math.max(0,Math.min(size-1,c[axis]));
                else if(mode==='circular')c[axis]=((c[axis]%size)+size)%size;
                else if(mode==='reflect'){const period=2*(size-1),v=((c[axis]%period)+period)%period;c[axis]=v<size?v:period-v;}
                else return [];
            }
        }
        return [{tensor:x.id,index:flatIndex(c,s),weight:1}];
    }});
    for(const fold of [false,true])register(fold?'fold2d':'unfold2d',{label:fold?'分块重组 · 重叠求和':'滑动窗口分块',color:'#8fbddd',formula:fold?'X[n,c,h,w] = Σ 对应 patch 元素':'P[n,c·kH·kW+k,l] = X[n,c,h(l,k),w(l,k)]',detail:'exact',dependencies:(n,out,i,t)=>{
        const x=t.get(n.inputs[0])!,xs=x.shape as number[],os=out.shape as number[],a=n.attrs??{},k=pair(a.kernel_size),d=pair(a.dilation),p=pair(a.padding,0),s=pair(a.stride);
        const image=fold?os:xs,h=image.at(-2)!,w=image.at(-1)!,cols=Math.floor((w+2*p[1]-d[1]*(k[1]-1)-1)/s[1]+1),rows=Math.floor((h+2*p[0]-d[0]*(k[0]-1)-1)/s[0]+1);
        const c=coordinates(i,os),deps:Dependency[]=[];
        if(!fold){
            const [b,ck,l]=c,ch=Math.floor(ck/(k[0]*k[1])),ky=Math.floor(ck%(k[0]*k[1])/k[1]),kx=ck%k[1];
            const y=Math.floor(l/cols)*s[0]-p[0]+ky*d[0],z=l%cols*s[1]-p[1]+kx*d[1];
            if(y>=0&&y<h&&z>=0&&z<w)deps.push({tensor:x.id,index:flatIndex([b,ch,y,z],xs),weight:1});
        }else{
            const unbatched=os.length===3,[b,ch,y,z]=unbatched?[0,...c]:c;
            for(let ky=0;ky<k[0];ky++)for(let kx=0;kx<k[1];kx++){
                const r=(y+p[0]-ky*d[0])/s[0],q=(z+p[1]-kx*d[1])/s[1];
                if(Number.isInteger(r)&&Number.isInteger(q)&&r>=0&&r<rows&&q>=0&&q<cols){const at=[ch*k[0]*k[1]+ky*k[1]+kx,r*cols+q];deps.push({tensor:x.id,index:flatIndex(unbatched?at:[b,...at],xs),weight:1});}
            }
        }
        return deps;
    }});
}
