/**
 * BASE SOLVER CLASS
 * (solvers/Solver.js)
 *
 * Defines the base skeleton class for numerical integrators.
 */

import {
    SOL_TOLERANCE_LTE_ABS,
    SOL_TOLERANCE_LTE_REL
} from '../constants.js';

/**
 * Base Solver class for numerical integration
 *
 * Defines the basic solver methods and metadata.
 * Specific solvers need to implement some of the base class methods.
 */
export class Solver {
    /**
     * Create a new Solver
     * @param {number|Array} [initialValue=0] - Initial condition
     * @param {Solver} [parent=null] - Parent solver instance
     * @param {Object} [options={}] - Solver options
     * @param {number} [options.toleranceLteAbs] - Absolute LTE tolerance
     * @param {number} [options.toleranceLteRel] - Relative LTE tolerance
     */
    constructor(initialValue = 0, parent = null, {
        toleranceLteAbs = SOL_TOLERANCE_LTE_ABS,
        toleranceLteRel = SOL_TOLERANCE_LTE_REL,
        ...options
    } = {}) {
        // Set state and initial condition
        this.initialValue = initialValue;
        this.x = Array.isArray(initialValue)
            ? [...initialValue]
            : [initialValue];

        // Track if initial value was scalar
        this._scalarInitial = !Array.isArray(initialValue);

        // Tolerances for local truncation error
        this.toleranceLteAbs = toleranceLteAbs;
        this.toleranceLteRel = toleranceLteRel;

        // Parent solver instance
        this.parent = parent;

        // Flag to identify adaptive/fixed timestep solvers
        this.isAdaptive = false;

        // Flag to identify explicit/implicit solvers
        this.isExplicit = true;

        // History of past solutions
        this.history = [];
        this._historyMaxLen = 1;

        // Order of the integration scheme
        this.n = 1;

        // Number of stages
        this.s = 1;

        // Current evaluation stage for multistage solvers
        this._stage = 0;

        // Intermediate evaluation times as ratios between [t, t+dt]
        this.evalStages = [0.0];
    }

    /**
     * String representation of solver
     * @returns {string} Solver class name
     */
    toString() {
        return this.constructor.name;
    }

    /**
     * Get size of internal state
     * @returns {number} State size
     */
    get length() {
        return this.x.length;
    }

    /**
     * Get current stage
     * @returns {number} Current stage
     */
    get stage() {
        return this.parent ? this.parent.stage : this._stage;
    }

    /**
     * Set current stage
     * @param {number} val - Stage value
     */
    set stage(val) {
        if (this.parent) {
            this.parent.stage = val;
        } else {
            this._stage = val;
        }
    }

    /**
     * Check if this is the first stage
     * @returns {boolean} True if first stage
     */
    isFirstStage() {
        return this.stage === 0;
    }

    /**
     * Check if this is the last stage
     * @returns {boolean} True if last stage
     */
    isLastStage() {
        return this.stage === this.s - 1;
    }

    /**
     * Generator for solver stages
     * @param {number} t - Current time
     * @param {number} dt - Timestep
     * @yields {number} Evaluation time for each stage
     */
    *stages(t, dt) {
        for (let i = 0; i < this.evalStages.length; i++) {
            this.stage = i;
            yield t + this.evalStages[i] * dt;
        }
    }

    /**
     * Get current internal state
     * @returns {number|Array} Current state
     */
    get() {
        return this._scalarInitial ? this.x[0] : [...this.x];
    }

    /**
     * Set internal state
     * @param {number|Array} x - New state
     */
    set(x) {
        this.x = Array.isArray(x) ? [...x] : [x];
    }

    /**
     * Reset solver to initial state
     */
    reset() {
        this.x = Array.isArray(this.initialValue)
            ? [...this.initialValue]
            : [this.initialValue];
        this.history = [];
        this._stage = 0;
    }

    /**
     * Buffer current state
     * @param {number} dt - Timestep
     */
    buffer(dt) {
        this.history.push([...this.x]);
        if (this.history.length > this._historyMaxLen) {
            this.history.shift();
        }
    }

    /**
     * Revert to previous state
     */
    revert() {
        if (this.history.length > 0) {
            this.x = [...this.history[this.history.length - 1]];
        }
    }

    /**
     * Solve implicit update equation
     * @param {Array} f - Right-hand side function values
     * @param {*} jacobian - Jacobian (if available)
     * @param {number} dt - Timestep
     * @returns {number} Residual norm
     */
    solve(f, jacobian, dt) {
        // Override in implicit solvers
        return 0.0;
    }

    /**
     * Perform one integration step
     * @param {Array} f - Right-hand side function values
     * @param {number} dt - Timestep
     * @returns {Array} [success, errorNorm, scale]
     */
    step(f, dt) {
        // Override in specific solvers
        return [true, 0.0, 1.0];
    }
}
