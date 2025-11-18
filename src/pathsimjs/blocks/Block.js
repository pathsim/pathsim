/**
 * BASE BLOCK CLASS
 * (blocks/Block.js)
 *
 * This module defines the base Block class that is the parent to all other blocks
 * and can serve as a base for new or custom blocks.
 */

import { Register } from '../utils/register.js';
import { PortReference } from '../utils/portreference.js';

/**
 * Base Block class
 *
 * Defines the fundamental structure and behavior of all blocks in the simulation.
 * Blocks have inputs, outputs, and can optionally have internal state managed by a solver.
 */
export class Block {
    // Static properties
    static _nInMax = null;   // Maximum number of input ports (null = unlimited)
    static _nOutMax = null;  // Maximum number of output ports (null = unlimited)
    static _portMapIn = {};  // String aliases for input ports
    static _portMapOut = {}; // String aliases for output ports

    /**
     * Create a new Block
     */
    constructor() {
        // Determine register sizes
        const nIn = Math.max(Object.keys(this.constructor._portMapIn).length, 1);
        const nOut = Math.max(Object.keys(this.constructor._portMapOut).length, 1);

        // Registers to hold input and output values
        this.inputs = new Register(nIn, this.constructor._portMapIn);
        this.outputs = new Register(nOut, this.constructor._portMapOut);

        // Initialize integration engine as null by default
        this.engine = null;

        // Flag to set block active
        this._active = true;

        // Internal discrete events (for mixed signal blocks)
        this.events = [];

        // Operators for algebraic and dynamic components
        this.opAlg = null;
        this.opDyn = null;

        // Unique ID for serialization
        this.__id = Symbol('blockId');
    }

    /**
     * Get the length of the algebraic path
     * @returns {number} Length (1 for instant blocks, 0 for stateful blocks)
     */
    get length() {
        return this._active ? 1 : 0;
    }

    /**
     * Enable array indexing syntax for port selection
     * @param {number|string|Array} key - Port index/indices or label(s)
     * @returns {PortReference} Port reference object
     */
    getItem(key) {
        let ports;

        if (typeof key === 'number' || typeof key === 'string') {
            ports = [key];
        } else if (Array.isArray(key)) {
            // Validate port types
            for (const k of key) {
                if (typeof k !== 'number' && typeof k !== 'string') {
                    throw new Error(`Port '${k}' must be number or string but is ${typeof k}`);
                }
            }

            // Check for duplicates
            if (new Set(key).size < key.length) {
                throw new Error('Ports cannot be duplicates!');
            }

            ports = key;
        } else {
            throw new Error(`Port must be number, string, or array but is ${typeof key}`);
        }

        return new PortReference(this, ports);
    }

    /**
     * Call operator to get all block state
     * @returns {Object} {inputs, outputs, states}
     */
    getAll() {
        const inputs = this.inputs.toArray();
        const outputs = this.outputs.toArray();
        const states = this.engine ? this.engine.get() : [];
        return { inputs, outputs, states };
    }

    /**
     * Boolean evaluation of block
     * @returns {boolean} Whether block is active
     */
    get active() {
        return this._active;
    }

    /**
     * Get size information (number of blocks, number of states)
     * @returns {Array} [nBlocks, nStates]
     */
    get size() {
        const nx = this.engine ? this.engine.length : 0;
        return [1, nx];
    }

    /**
     * Get shape information (number of input/output ports)
     * @returns {Array} [nInputs, nOutputs]
     */
    get shape() {
        return [this.inputs.length, this.outputs.length];
    }

    /**
     * Activate the block and all internal events
     */
    on() {
        this._active = true;
        for (const event of this.events) {
            event.on();
        }
    }

    /**
     * Deactivate the block and all internal events
     */
    off() {
        this._active = false;
        for (const event of this.events) {
            event.off();
        }
        this.reset();
    }

    /**
     * Reset the block's inputs, outputs, and internal state
     */
    reset() {
        this.inputs.reset();
        this.outputs.reset();

        if (this.engine) {
            this.engine.reset();
        }

        if (this.opAlg) {
            this.opAlg.reset();
        }

        if (this.opDyn) {
            this.opDyn.reset();
        }
    }

    /**
     * Linearize the algebraic and dynamic components
     * @param {number} t - Evaluation time
     */
    linearize(t) {
        const { inputs, outputs, states } = this.getAll();

        if (!this.engine) {
            // Stateless block - linearize only algebraic operator
            if (this.opAlg) {
                this.opAlg.linearize(inputs);
            }
        } else {
            // Stateful block - linearize both operators
            if (this.opAlg) {
                this.opAlg.linearize(states, inputs, t);
            }
            if (this.opDyn) {
                this.opDyn.linearize(states, inputs, t);
            }
        }
    }

    /**
     * Revert linearization
     */
    delinearize() {
        if (this.opAlg) {
            this.opAlg.reset();
        }
        if (this.opDyn) {
            this.opDyn.reset();
        }
    }

    /**
     * Set the numerical integration solver
     * @param {Function} Solver - Solver class
     * @param {Object} parent - Parent solver instance
     * @param {Object} solverArgs - Additional solver arguments
     */
    setSolver(Solver, parent, solverArgs = {}) {
        // Override in subclasses that need solvers
    }

    /**
     * Revert to previous timestep state
     */
    revert() {
        if (this.engine) {
            this.engine.revert();
        }
    }

    /**
     * Buffer current state before timestep
     * @param {number} dt - Timestep
     */
    buffer(dt) {
        if (this.engine) {
            this.engine.buffer(dt);
        }
    }

    /**
     * Sample data at current time
     * @param {number} t - Current time
     * @param {number} dt - Timestep
     */
    sample(t, dt) {
        // Override in blocks that need sampling (Scope, Delay, etc.)
    }

    /**
     * Plot block data
     * @param  {...any} args - Arguments for plotting
     */
    plot(...args) {
        // Override in blocks that have visualization (Scope, Spectrum, etc.)
    }

    /**
     * Update block outputs based on inputs (algebraic evaluation)
     * @param {number} t - Evaluation time
     */
    update(t) {
        // No internal algebraic operator - early exit
        if (!this.opAlg) {
            return 0.0;
        }

        // Get block inputs
        const u = this.inputs.toArray();

        let y;
        if (this.engine) {
            // Stateful block
            const x = this.engine.get();
            y = this.opAlg(x, u, t);
        } else {
            // Stateless block
            y = this.opAlg(u);
        }

        // Update outputs
        this.outputs.updateFromArray(y);
    }

    /**
     * Solve implicit update equation (for implicit solvers)
     * @param {number} t - Evaluation time
     * @param {number} dt - Timestep
     * @returns {number} Solver residual norm
     */
    solve(t, dt) {
        return 0.0;
    }

    /**
     * Advance block state by one timestep
     * @param {number} t - Evaluation time
     * @param {number} dt - Timestep
     * @returns {Array} [success, errorNorm, scale]
     */
    step(t, dt) {
        // By default, no error estimate
        return [true, 0.0, 1.0];
    }

    /**
     * Convert block to dictionary for serialization
     * @returns {Object} Dictionary representation
     */
    toDict() {
        return {
            type: this.constructor.name,
            id: this.__id.description || String(this.__id),
            inputs: this.inputs.toArray(),
            outputs: this.outputs.toArray()
        };
    }
}
