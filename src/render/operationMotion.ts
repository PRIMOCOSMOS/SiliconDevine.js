import * as T from 'three';
import type { Position3 } from './crystalPrimitives.js';
import { numericPalette } from './numericPalette.js';
export interface Connection {
    from: Position3;
    to: Position3;
    weight?: number;
    weightScale?: number;
    output?: number;
    term?: number;
    parameterIndex?: number;
    via?: Position3[];
}
/** A connection is a weighted dependency, not a proxy neuron. GPU pulses travel
 * with overlapping, smooth output/term envelopes; every weight retains a visible baseline. */
export function connectionFabric(parent: T.Group, links: Connection[]) {
    const geometry = new T.BufferGeometry(), positions: number[] = [], along: number[] = [], weights: number[] = [], outputs: number[] = [], terms: number[] = [], weighted: number[] = [], parameters: number[] = [];
    for (const l of links) {
        const points = [l.from, ...l.via ?? [], l.to], w = (l.weight ?? 0) / Math.max(l.weightScale ?? 1, Number.EPSILON);
        for (let i = 0; i < points.length - 1; i++) {
            positions.push(...points[i], ...points[i + 1]);
            along.push(i / (points.length - 1), (i + 1) / (points.length - 1));
            weights.push(w, w);
            weighted.push(l.weight === undefined ? 0 : 1, l.weight === undefined ? 0 : 1);
            outputs.push(l.output ?? -1, l.output ?? -1);
            terms.push(l.term ?? -1, l.term ?? -1);
            parameters.push(l.parameterIndex ?? -1, l.parameterIndex ?? -1);
        }
    }
    geometry.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('along', new T.Float32BufferAttribute(along, 1));
    geometry.setAttribute('weight', new T.Float32BufferAttribute(weights, 1));
    geometry.setAttribute('outputIndex', new T.Float32BufferAttribute(outputs, 1));
    geometry.setAttribute('termIndex', new T.Float32BufferAttribute(terms, 1));
    geometry.setAttribute('weighted', new T.Float32BufferAttribute(weighted, 1));
    geometry.setAttribute('parameterIndex', new T.Float32BufferAttribute(parameters, 1));
    const paletteUniforms = Object.fromEntries(Object.entries(numericPalette).map(([key, value]) => [key, { value: new T.Color(value) }]));
    const material = new T.ShaderMaterial({ transparent: true, depthWrite: false, uniforms: { ...paletteUniforms, phase: { value: 0 }, selectedOutput: { value: -1 }, selectedTerm: { value: -1 }, strength: { value: 1 }, baseline: { value: 1 }, outputCount: { value: Math.max(1, links.reduce((n, l) => Math.max(n, (l.output ?? 0) + 1), 0)) }, density: { value: Math.max(.3, 1 / Math.sqrt(Math.max(1, links.length / 128))) } }, vertexShader: `
 attribute float along,weight,outputIndex,termIndex,weighted,parameterIndex;
 varying float vAlong,vWeight,vOutput,vTerm,vWeighted,vParameter;
 void main(){vParameter=parameterIndex;vAlong=along;vWeight=weight;vOutput=outputIndex;vTerm=termIndex;vWeighted=weighted;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}
 `, fragmentShader: `
 uniform float phase,selectedOutput,selectedTerm,strength,baseline,density,outputCount,selectedParameter;
 uniform vec3 negative,negativeMid,zero,positiveMid,positive,relation,focus;
 varying float vAlong,vWeight,vOutput,vTerm,vWeighted,vParameter;
 void main(){
  float delta=mod(abs(vOutput-selectedOutput),outputCount);
  float chosen=exp(-pow(min(delta,outputCount-delta)/1.4,2.));
  float current=chosen*exp(-pow((vTerm-selectedTerm)/2.5,2.));
  float linked=selectedParameter>=0.&&vParameter>=0.?exp(-pow((vParameter-selectedParameter)/.85,2.)):0.;
  chosen=max(chosen*.5,linked);
  current=max(current*.35,linked);
  float head=fract(phase-vOutput*.037-vTerm*.021);
  float pulse=exp(-pow(abs(vAlong-head)/.16,2.));
  float a=4.*min(1.,abs(vWeight));
  float magnitude=log(a+sqrt(a*a+1.))/2.094712547;
  vec3 middle=vWeight<0.?negativeMid:positiveMid,end=vWeight<0.?negative:positive;
  vec3 tint=magnitude<.5?mix(zero,middle,magnitude*2.):mix(middle,end,(magnitude-.5)*2.);
  tint=mix(relation,tint,vWeighted);
  // Pulses retain the signed hue; white is an accent rather than a wash.
  tint=mix(tint,focus,min(.32,pulse*(.12+chosen*.2)*strength));
  float opacity=(.2+.14*magnitude)*density*baseline+strength*(chosen*.1+current*.15+pulse*(.01+chosen*.5));
  gl_FragColor=vec4(tint,min(.95,opacity));
  #include <colorspace_fragment>
 }
 ` });
    const mesh = new T.LineSegments(geometry, material);
    material.uniforms.selectedParameter = { value: -1 };
    parent.add(mesh);
    return { mesh, update: (phase: number, output = -1, term = -1, strength = 1, baseline = 1, parameter = -1) => { material.uniforms.selectedParameter.value = parameter; material.uniforms.phase.value = phase; material.uniforms.selectedOutput.value = output; material.uniforms.selectedTerm.value = term; material.uniforms.strength.value = strength; material.uniforms.baseline.value = baseline; }, dispose: () => { parent.remove(mesh); geometry.dispose(); material.dispose(); } };
}
