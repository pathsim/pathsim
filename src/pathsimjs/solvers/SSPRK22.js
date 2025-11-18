/**
 * SSPRK22 SOLVER
 * (solvers/SSPRK22.js)
 *
 * Second-order Strong Stability Preserving Runge-Kutta method (SSPRK22).
 * This is a simple, robust explicit solver suitable for general use.
 */

import { Solver } from './Solver.js';

/**
 * SSPRK22 - 2nd order Strong Stability Preserving Runge-Kutta
 *
 * This is a two-stage explicit Runge-Kutta method with strong stability
 * preserving properties, making it suitable for a wide range of problems.
 *
 * Butcher tableau:
 *   0   |
 *   1   | 1
 *   ----+-------
 *       | 1/2  1/2
 */
export class SSPRK22 extends Solver {
    /**
     * Create a new SSPRK22 solver
     * @param {number|Array} [initialValue=0] - Initial condition
     * @param {Solver} [parent=null] - Parent solver instance
     * @param {Object} [options={}] - Solver options
     */
    constructor(initialValue = 0, parent = null, options = {}) {
        super(initialValue, parent, options);

        // Order of the method
        this.n = 2;

        // Number of stages
        this.s = 2;

        // Evaluation stage times
        this.evalStages = [0.0, 1.0];

        // Explicit method
        this.isExplicit = true;

        // Not adaptive
        this.isAdaptive = false;

        // Storage for intermediate stages
        this.k = [null, null];
    }

    /**
     * Perform one SSPRK22 integration step
     * @param {number|Array} f - Right-hand side function values
     * @param {number} dt - Timestep
     * @returns {Array} [success, errorNorm, scale]
     */
    step(f, dt) {
        // Ensure f is array
        const fArr = Array.isArray(f) ? f : [f];

        // Get current stage
        const stage = this.stage;

        if (stage === 0) {
            // First stage: k1 = f(t, x)
            // Store k1
            this.k[0] = [...fArr];

            // No state update yet
        } else if (stage === 1) {
            // Second stage: k2 = f(t + dt, x + dt*k1)
            // Store k2
            this.k[1] = [...fArr];

            // Update state: x_new = x + dt/2 * (k1 + k2)
            for (let i = 0; i < this.x.length; i++) {
                this.x[i] = this.history[this.history.length - 1][i] +
                           (dt / 2.0) * (this.k[0][i] + this.k[1][i]);
            }
        }

        // No error estimate for fixed-step method
        return [true, 0.0, 1.0];
    }

    /**
     * Buffer current state before timestep
     * @param {number} dt - Timestep
     */
    buffer(dt) {
        super.buffer(dt);
        // Reset intermediate stages
        this.k = [null, null];
    }
}
