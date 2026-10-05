import * as T from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { numericColor } from './numericPalette.js';
export type Position3 = [
    number,
    number,
    number
];
export type CrystalSelection = number | {
    focus?: number;
    active?: number[];
    hidden?: number[];
};
const positive = new T.Color('#78cde7'), negative = new T.Color('#e9af83'), selected = new T.Color('#e3fbff');
/** Uniform tensor cells: values affect tint, never geometry. Four instanced draws per tensor. */
export function createCrystalTensor(parent: T.Group, count: number, size = .42, style?: {
    valueEdges?: boolean;
    parameter?: boolean;
    bodyOpacity?: number;
    edgeOpacity?: number;
    valueScale?: number | (() => number);
}) {
    const group = new T.Group();
    group.userData.crystalTensor = true;
    group.userData.parameterTensor = !!style?.parameter;
    parent.add(group);
    const cube = new T.BoxGeometry(size, size, size);
    const edgeParts: T.BufferGeometry[] = [];
    for (let axis = 0; axis < 3; axis++)
        for (const a of [-1, 1])
            for (const b of [-1, 1]) {
                const dimensions = [.012, .012, .012];
                dimensions[axis] = size;
                const position = [0, 0, 0];
                position[(axis + 1) % 3] = a * size / 2;
                position[(axis + 2) % 3] = b * size / 2;
                edgeParts.push(new T.BoxGeometry(...dimensions as Position3).translate(...position as Position3));
            }
    const edges = mergeGeometries(edgeParts);
    if(style?.parameter) { cube.scale(1,.42,1); edges.scale(1,.42,1); }
    edgeParts.forEach(g => g.dispose());
    const bodyMaterial = new T.MeshPhysicalMaterial({ color: '#ffffff', roughness: .13, metalness: .08, clearcoat: 1, clearcoatRoughness: .08, transparent: true, opacity: .12, depthWrite: false });
    if(style?.parameter) {bodyMaterial.metalness=.5;bodyMaterial.roughness=.32;}
    const edgeMaterial = new T.MeshBasicMaterial({ color: '#a3d4e4', transparent: true, opacity: .25, depthWrite: false });
    if (style?.bodyOpacity !== undefined)
        bodyMaterial.opacity = style.bodyOpacity;
    if (style?.edgeOpacity !== undefined)
        edgeMaterial.opacity = style.edgeOpacity;
    if (style?.valueEdges)
        edgeMaterial.color.set('#ffffff');
    const activeMaterial = bodyMaterial.clone();
    activeMaterial.opacity = .68;
    activeMaterial.emissive.set('#70bfd2');
    activeMaterial.emissiveIntensity = .13;
    const activeEdgeMaterial = new T.MeshBasicMaterial({ transparent: true, opacity: .94, depthWrite: false });
    const body = new T.InstancedMesh(cube, bodyMaterial, count), edge = new T.InstancedMesh(edges, edgeMaterial, count);
    const lit = new T.InstancedMesh(cube, activeMaterial, count), litEdge = new T.InstancedMesh(edges, activeEdgeMaterial, count);
    const meshes = [body, edge, lit, litEdge];
    meshes.forEach(mesh => { mesh.instanceMatrix.setUsage(T.DynamicDrawUsage); mesh.frustumCulled = false; group.add(mesh); });
    const transform = new T.Object3D(), tint = new T.Color();
    function update(values: number[], positions: Position3[], selection: CrystalSelection = -1) {
        const state = typeof selection === 'number' ? { focus: selection } : selection;
        const active = new Set(state.active ?? []), hidden = new Set(state.hidden ?? []);
        const colorScale = typeof style?.valueScale === 'function' ? style.valueScale() : style?.valueScale;
        let visibleCount = 0, activeCount = 0;
        for (let i = 0; i < count; i++) {
            if (hidden.has(i) || !positions[i])
                continue;
            transform.position.set(...positions[i]);
            transform.updateMatrix();
            if (colorScale !== undefined)
                numericColor(values[i] ?? 0, colorScale, tint);
            else
                tint.copy((values[i] ?? 0) < 0 ? negative : positive).multiplyScalar(.5 + .5 * Math.min(1, Math.abs(values[i] ?? 0)));
            body.setMatrixAt(visibleCount, transform.matrix);
            body.setColorAt(visibleCount, tint);
            edge.setMatrixAt(visibleCount, transform.matrix);
            if (style?.valueEdges)
                edge.setColorAt(visibleCount, tint);
            visibleCount++;
            if (i === state.focus || active.has(i)) {
                lit.setMatrixAt(activeCount, transform.matrix);
                lit.setColorAt(activeCount, style?.valueScale !== undefined ? tint.clone().lerp(selected, i === state.focus ? .28 : .1) : i === state.focus ? selected : tint.clone().multiplyScalar(.65));
                litEdge.setMatrixAt(activeCount, transform.matrix);
                litEdge.setColorAt(activeCount++, i === state.focus ? selected : tint);
            }
        }
        body.count = edge.count = visibleCount;
        lit.count = litEdge.count = activeCount;
        body.visible = edge.visible = visibleCount > 0;
        lit.visible = litEdge.visible = activeCount > 0;
        meshes.forEach(mesh => {
            mesh.instanceMatrix.needsUpdate = true;
            if (mesh.instanceColor)
                mesh.instanceColor.needsUpdate = true;
        });
        group.userData.focus = state.focus ?? -1;
        group.userData.active = [...active];
    }
    return { group, update, body, edge, lit, litEdge };
}
/** Arrow trains carry direction even while paused. Paths and tangent orientation share one curve. */
export function createArrowStream(parent: T.Group, color = '#c4f3fa', size = .15, count = 3) {
    const group = new T.Group();
    group.userData.directionalFlow = true;
    parent.add(group);
    const headGeometry = new T.ConeGeometry(size, size * 2.8, 4);
    const shaftGeometry = new T.CylinderGeometry(size * .14, size * .14, size * 3, 5).translate(0, -size * 2.4, 0);
    const geometry = mergeGeometries([headGeometry, shaftGeometry]);
    headGeometry.dispose();
    shaftGeometry.dispose();
    const material = new T.MeshBasicMaterial({ color, transparent: true, opacity: .92, depthWrite: false });
    const arrows = Array.from({ length: count }, () => new T.Object3D());
    const instances = new T.InstancedMesh(geometry, material, count);
    instances.instanceMatrix.setUsage(T.DynamicDrawUsage);
    instances.frustumCulled = false;
    group.add(instances);
    const track = new T.Line(new T.BufferGeometry(), new T.LineBasicMaterial({ color, transparent: true, opacity: .2, depthWrite: false }));
    group.add(track);
    const up = new T.Vector3(0, 1, 0);
    let key = '', curve: T.CatmullRomCurve3 | undefined;
    function update(points: Position3[] | T.CatmullRomCurve3, phase: number, strength = 1) {
        if (Array.isArray(points)) {
            const nextKey = points.flat().join(',');
            if (nextKey !== key) {
                key = nextKey;
                curve = new T.CatmullRomCurve3(points.map(p => new T.Vector3(...p)), false, 'centripetal');
                const previous = track.geometry;
                track.geometry = new T.BufferGeometry().setFromPoints(curve.getPoints(32));
                previous.dispose();
            }
        }
        else {
            curve = points;
            track.visible = false;
        }
        const valid = !!curve && curve.getLength() > .001 && strength > .001;
        group.visible = valid;
        if (!valid || !curve)
            return;
        material.opacity = .35 + .6 * strength;
        arrows.forEach((arrow, i) => {
            const t = .08 + (((phase + i / count) % 1 + 1) % 1) * .84;
            arrow.position.copy(curve!.getPointAt(t));
            arrow.quaternion.setFromUnitVectors(up, curve!.getTangentAt(t).normalize());
            arrow.updateMatrix();
            instances.setMatrixAt(i, arrow.matrix);
        });
        instances.instanceMatrix.needsUpdate = true;
    }
    return { group, arrows, update };
}
