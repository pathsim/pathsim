/**
 * AMPLIFIER BLOCK
 * (blocks/Amplifier.js)
 *
 * Multiplies input signal by a constant gain.
 */

import { Block } from './Block.js';

/**
 * Amplifier block - multiplies input by constant gain
 *
 * Implements y = gain * u
 */
export class Amplifier extends Block {
    /**
     * Create a new Amplifier block
     * @param {number} [gain=1.0] - Amplification gain
     */
    constructor(gain = 1.0) {
        super();
        this.gain = gain;
    }

    /**
     * Update output based on input
     * @param {number} t - Evaluation time
     */
    update(t) {
        const u = this.inputs.get(0);
        const y = this.gain * u;
        this.outputs.set(0, y);
    }

    /**
     * Convert to dictionary for serialization
     * @returns {Object} Dictionary representation
     */
    toDict() {
        return {
            ...super.toDict(),
            params: { gain: this.gain }
        };
    }
}
