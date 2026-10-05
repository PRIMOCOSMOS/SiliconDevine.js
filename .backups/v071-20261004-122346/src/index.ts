export { SiliconDevineViewer } from './render/viewer.js';
export type { ViewerOptions, ViewerStats } from './render/viewer.js';
export { defineModel, ModelBuilder } from './core/builder.js';
export { compileDescription } from './core/description.js';
export type { ModelDescription } from './core/description.js';
export { validateModel, numel, valueAt, coordinates, flatIndex, topologicalNodes } from './core/model.js';
export type { Model, Tensor, Operation, Dimension } from './core/model.js';
export { registerOperator, operatorVisual } from './core/operators.js';
export type { OperatorVisual, Dependency } from './core/operators.js';
export { numericPalette } from './render/numericPaletteTokens.js';
export { LiveConnection } from './core/liveClient.js';
export type { LiveStatus } from './core/liveClient.js';
export { moduleView } from './core/hierarchy.js';
export { aggregatePartition } from './core/layout.js';
export { defineArchitecture } from './core/architecture.js';
export type { ArchitectureScopes } from './core/architecture.js';

export { inspectSupport } from './core/support.js';
