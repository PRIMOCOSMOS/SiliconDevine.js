import { defineModel } from './builder.js';
import type { Model } from './model.js';
export interface ModelDescription {
    format: 'silicondevine-recipe';
    version: 1;
    name: string;
    seed?: number;
    input: {
        name?: string;
        shape: number[];
        values?: number[];
    };
    layers: ({
        name: string;
        type: 'linear';
        outFeatures: number;
    } | {
        name: string;
        type: 'relu' | 'gelu' | 'sigmoid' | 'tanh';
    })[];
}
/** Friendly sequential recipe; full graph IR handles branches and custom operators. */
export function compileDescription(input: unknown): Model {
    const d = input as ModelDescription;
    if (!d || d.format !== 'silicondevine-recipe' || d.version !== 1 || typeof d.name !== 'string' || !d.input || !Array.isArray(d.input.shape) || !Array.isArray(d.layers) || d.layers.length > 64)
        throw Error('无效描述：需要名称、输入形状和最多 64 层的 layers。');
    const model = defineModel(d.name, d.seed).input(d.input.name ?? 'x', d.input.shape, d.input.values);
    for (const layer of d.layers) {
        if (typeof layer.name !== 'string')
            throw Error('每层需要唯一名称。');
        if (layer.type === 'linear')
            model.linear(layer.name, layer.outFeatures);
        else if (['relu', 'gelu', 'sigmoid', 'tanh'].includes(layer.type))
            model.activation(layer.name, layer.type);
        else
            throw Error(`描述式构建器暂不支持 ${(layer as {
                type: string;
            }).type}；请使用标准图格式。`);
    }
    return model.build();
}
