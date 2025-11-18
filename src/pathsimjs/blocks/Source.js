/**
 * SOURCE BLOCK
 * (blocks/Source.js)
 *
 * Generates time-dependent output signal.
 */

import { Block } from './Block.js';

/**
 * Source block - generates time-dependent signal
 *
 * Implements y = f(t)
 */
export class Source extends Block {
    /**
     * Create a new Source block
     * @param {Function} func - Function of time: f(t) => value
     */
    constructor(func = (t) => 0.0) {
        super();
        this.func = func;
    }

    /**
     * Update output based on current time
     * @param {number} t - Evaluation time
     */
    update(t) {
        const y = this.func(t);
        this.outputs.set(0, y);
    }

    /**
     * Convert to dictionary for serialization
     * @returns {Object} Dictionary representation
     */
    toDict() {
        return {
            ...super.toDict(),
            params: { func: this.func.toString() }
        };
    }
}

/**
 * Constant block - outputs constant value
 */
export class Constant extends Block {
    /**
     * Create a new Constant block
     * @param {number} [value=0.0] - Constant value
     */
    constructor(value = 0.0) {
        super();
        this.value = value;
        // Set initial output
        this.outputs.set(0, value);
    }

    /**
     * Update output (constant, so no change)
     * @param {number} t - Evaluation time
     */
    update(t) {
        this.outputs.set(0, this.value);
    }

    /**
     * Convert to dictionary for serialization
     * @returns {Object} Dictionary representation
     */
    toDict() {
        return {
            ...super.toDict(),
            params: { value: this.value }
        };
    }
}
