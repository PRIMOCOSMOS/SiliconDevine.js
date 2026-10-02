import { Color } from 'three';
import { numericPalette } from './numericPaletteTokens.js';
export { numericPalette } from './numericPaletteTokens.js';
/** Signed scientific colour scale. State and topology never change its meaning. */
const colors = Object.fromEntries(Object.entries(numericPalette).map(([key, value]) => [key, new Color(value)])) as Record<keyof typeof numericPalette, Color>;
export const valueExtent = (values: readonly number[]) => values.reduce((m, v) => Number.isFinite(v) ? Math.max(m, Math.abs(v)) : m, 0) || 1;
/** Symmetric asinh: preserves zero/sign and reveals small values without clipping outliers. */
export const valueMagnitude = (value: number, extent: number) => Math.asinh(4 * Math.min(1, Math.abs(value) / Math.max(extent, Number.EPSILON))) / Math.asinh(4);
export function numericColor(value: number, extent: number, target: Color) {
    if (!Number.isFinite(value))
        return target.copy(colors.unknown);
    const amount = valueMagnitude(Number.isFinite(value) ? value : 0, extent);
    const mid = value < 0 ? colors.negativeMid : colors.positiveMid, end = value < 0 ? colors.negative : colors.positive;
    return amount < .5 ? target.copy(colors.zero).lerp(mid, amount * 2) : target.copy(mid).lerp(end, (amount - .5) * 2);
}
