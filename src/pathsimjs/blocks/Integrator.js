/**
 * INTEGRATOR BLOCK
 * (blocks/Integrator.js)
 *
 * Integrates the input signal using a numerical integration engine.
 */

import { Block } from './Block.js';

/**
 * Integrator block - integrates input signal
 *
 * Implements:
 *   dx/dt = u(t)
 *   y(t) = x(t)
 *
 * Or in integral form:
 *   y(t) = ∫u(τ)dτ from 0 to t
 */
export class Integrator extends Block {
    /**
     * Create a new Integrator block
     * @param {number|Array} [initialValue=0.0] - Initial value of integrator
     */
    constructor(initialValue = 0.0) {
        super();
        this.initialValue = initialValue;
    }

    /**
     * Get the length of the algebraic path
     * @returns {number} Always 0 (no passthrough)
     */
    get length() {
        return 0;
    }

    /**
     * Set the internal numerical integrator
     * @param {Function} Solver - Numerical integration solver class
     * @param {Object} parent - Solver instance to use as parent
     * @param {Object} solverArgs - Parameters for solver initialization
     */
    setSolver(Solver, parent, solverArgs = {}) {
        if (!this.engine) {
            // Initialize the integration engine
            this.engine = new Solver(this.initialValue, parent, solverArgs);
        } else {
            // Change solver if already initialized
            // Simplified: just create new solver
            this.engine = new Solver(this.initialValue, parent, solverArgs);
        }
    }

    /**
     * Update system equation (no passthrough, output from engine)
     * @param {number} t - Evaluation time
     */
    update(t) {
        const x = this.engine ? this.engine.get() : this.initialValue;
        this.outputs.updateFromArray(Array.isArray(x) ? x : [x]);
    }

    /**
     * Solve implicit update equation
     * @param {number} t - Evaluation time
     * @param {number} dt - Integration timestep
     * @returns {number} Solver residual norm
     */
    solve(t, dt) {
        if (!this.engine) return 0.0;

        const f = this.inputs.toArray();
        return this.engine.solve(f, null, dt);
    }

    /**
     * Compute timestep update with integration engine
     * @param {number} t - Evaluation time
     * @param {number} dt - Integration timestep
     * @returns {Array} [success, errorNorm, scale]
     */
    step(t, dt) {
        if (!this.engine) return [true, 0.0, 1.0];

        const f = this.inputs.toArray();
        return this.engine.step(f, dt);
    }

    /**
     * Convert to dictionary for serialization
     * @returns {Object} Dictionary representation
     */
    toDict() {
        return {
            ...super.toDict(),
            params: { initialValue: this.initialValue }
        };
    }
}
