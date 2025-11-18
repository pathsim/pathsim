/**
 * ADDER BLOCK
 * (blocks/Adder.js)
 *
 * Sums multiple input signals with optional sign operations.
 */

import { Block } from './Block.js';

/**
 * Adder block - sums inputs with optional operations
 *
 * Implements y = sum(op[i] * u[i])
 * where op[i] is either +1 or -1 based on operations string
 */
export class Adder extends Block {
    /**
     * Create a new Adder block
     * @param {string} [operations='++'] - String of operations ('+' or '-')
     */
    constructor(operations = '++') {
        super();

        // Parse operations string
        this.operations = operations.split('').map(op => {
            if (op === '+') return 1;
            if (op === '-') return -1;
            throw new Error(`Invalid operation: ${op}`);
        });

        // Set input register size
        this.inputs.resize(this.operations.length);
    }

    /**
     * Update output based on inputs
     * @param {number} t - Evaluation time
     */
    update(t) {
        let sum = 0.0;

        for (let i = 0; i < this.operations.length; i++) {
            const u = this.inputs.get(i);
            sum += this.operations[i] * u;
        }

        this.outputs.set(0, sum);
    }

    /**
     * Convert to dictionary for serialization
     * @returns {Object} Dictionary representation
     */
    toDict() {
        const ops = this.operations.map(o => o === 1 ? '+' : '-').join('');
        return {
            ...super.toDict(),
            params: { operations: ops }
        };
    }
}
