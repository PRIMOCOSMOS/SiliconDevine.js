import * as T from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

/** Branding only. This scene never receives scientific model data. */
export class KeynoteArtwork {
    readonly renderer:T.WebGLRenderer;
    readonly scene=new T.Scene();
    readonly camera=new T.PerspectiveCamera(34,1,.1,50);
    private sculpture=new T.Group();
    private lens=new T.Group();
    private crystal:T.Mesh;
    private environment:T.WebGLRenderTarget;
    constructor(canvas:HTMLCanvasElement){
        this.renderer=new T.WebGLRenderer({canvas,alpha:true,antialias:true,powerPreference:'low-power'});
        this.renderer.setClearColor('#06080d',0);
        this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
        this.renderer.toneMapping=T.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.2;
        const room=new RoomEnvironment(),generator=new T.PMREMGenerator(this.renderer);
        this.environment=generator.fromScene(room,.03,0.1,30);this.scene.environment=this.environment.texture;
        room.dispose();generator.dispose();
        this.camera.position.set(0,.15,10.8);
        this.scene.add(this.sculpture);this.sculpture.add(this.lens);
        const silver=new T.MeshStandardMaterial({color:'#d5e5f1',metalness:1,roughness:.22});
        const graphite=new T.MeshStandardMaterial({color:'#586980',metalness:.95,roughness:.2});
        const ring=new T.Mesh(new T.TorusGeometry(2.22,.16,20,160,Math.PI*1.83),silver);
        ring.rotation.z=.34;this.lens.add(ring);
        const inset=new T.Mesh(new T.TorusGeometry(1.94,.042,10,144),graphite);inset.position.z=.14;this.lens.add(inset);
        const rim=new T.Mesh(new T.TorusGeometry(2.22,.018,8,160,Math.PI*1.25),new T.MeshBasicMaterial({color:'#a9daf9'}));rim.position.z=.14;rim.rotation.z=.5;this.lens.add(rim);
        const secondary=new T.Mesh(new T.TorusGeometry(2.07,.07,12,144,Math.PI*1.7),graphite);
        secondary.rotation.set(.75,-.7,-.4);this.sculpture.add(secondary);
        const glass=new T.MeshPhysicalMaterial({color:'#a9d7ed',metalness:.3,roughness:.12,clearcoat:1,clearcoatRoughness:.04,transparent:true,opacity:.72,side:T.DoubleSide,iridescence:.3,iridescenceIOR:1.3});
        this.crystal=new T.Mesh(new T.OctahedronGeometry(.95),glass);this.sculpture.add(this.crystal);
        const edge=new T.LineSegments(new T.EdgesGeometry(this.crystal.geometry),new T.LineBasicMaterial({color:'#d1efff',transparent:true,opacity:.68}));this.crystal.add(edge);
        const core=new T.Mesh(new T.OctahedronGeometry(.27),new T.MeshStandardMaterial({color:'#f0faff',emissive:'#609cc2',emissiveIntensity:.6,metalness:.45,roughness:.15}));this.crystal.add(core);
        const key=new T.DirectionalLight('#d8e9ff',4.2);key.position.set(-3,4,5);
        const blue=new T.DirectionalLight('#627fcd',2.8);blue.position.set(4,-1,-2);
        this.scene.add(key,blue);
    }
    resize(w:number,h:number){if(w<1||h<1)return;this.renderer.setSize(w,h,false);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();}
    render(time=0,px=0,py=0){
        this.lens.rotation.set(.56+Math.sin(time*.18)*.05,-.4,-.4);
        this.sculpture.rotation.set(py*.05,px*.07,Math.sin(time*.12)*.028);
        this.crystal.rotation.set(.18,time*.065,.2);this.crystal.position.y=Math.sin(time*.4)*.055;
        this.renderer.render(this.scene,this.camera);
    }
    point(a:number){this.scene.updateMatrixWorld(true);return new T.Vector3(2.22*Math.cos(a),2.22*Math.sin(a),.18).applyMatrix4(this.lens.matrixWorld).project(this.camera).toArray();}
    dispose(){const materials=new Set<T.Material>();this.scene.traverse(o=>{const mesh=o as T.Mesh;mesh.geometry?.dispose();if(mesh.material)(Array.isArray(mesh.material)?mesh.material:[mesh.material]).forEach(m=>materials.add(m));});materials.forEach(m=>m.dispose());this.environment.dispose();this.renderer.dispose();}
}

export function mountKeynoteArtwork(host:HTMLElement){
    const canvas=document.createElement('canvas');canvas.setAttribute('aria-hidden','true');host.append(canvas);
    let art:KeynoteArtwork|undefined,frame=0,visible=false,last=0,elapsed=0,px=0,py=0,failed=false;
    const reduced=matchMedia('(prefers-reduced-motion: reduce)');
    const moving=()=>!reduced.matches&&document.body.dataset.effects==='full';
    const ready=()=>visible&&!document.hidden&&!failed;
    function draw(time:number){frame=0;if(!ready()||!art)return;if(time-last>=50){elapsed+=Math.min((time-last)/1000,.08);last=time;art.render(elapsed,px,py);}if(moving())frame=requestAnimationFrame(draw);}
    function sync(){cancelAnimationFrame(frame);frame=0;if(!ready())return;
        try{art??=new KeynoteArtwork(canvas);art.resize(host.clientWidth,host.clientHeight);art.render(moving()?elapsed:0);if(moving()){last=performance.now();frame=requestAnimationFrame(draw);}}
        catch{failed=true;canvas.remove();host.classList.add('keynote-fallback');}
    }
    const observer=new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;sync();});observer.observe(host);
    const resize=new ResizeObserver(sync);resize.observe(host);
    const effects=new MutationObserver(sync);effects.observe(document.body,{attributes:true,attributeFilter:['data-effects','data-catalog']});
    host.addEventListener('pointermove',e=>{const r=host.getBoundingClientRect();px=(e.clientX-r.left)/r.width-.5;py=(e.clientY-r.top)/r.height-.5;});
    host.addEventListener('pointerleave',()=>{px=py=0;});
    document.addEventListener('visibilitychange',sync);reduced.addEventListener('change',sync);
    window.addEventListener('pagehide',event=>{cancelAnimationFrame(frame);frame=0;if(event.persisted)return;observer.disconnect();resize.disconnect();effects.disconnect();document.removeEventListener('visibilitychange',sync);reduced.removeEventListener('change',sync);art?.dispose();});
    window.addEventListener('pageshow',sync);
}
