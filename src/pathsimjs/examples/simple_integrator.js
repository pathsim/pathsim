/**
 * SIMPLE INTEGRATOR EXAMPLE
 * (examples/simple_integrator.js)
 *
 * This is the simplest possible example using PathSimJS.
 * It demonstrates integrating a constant input signal.
 *
 * System equation:
 *   dx/dt = u = 1.0
 *   y = x
 *
 * Result: A ramp function (linear increase from 0)
 */

import { Simulation, Connection } from '../index.js';
import { Integrator, Constant, Scope } from '../blocks/index.js';
import { SSPRK22 } from '../solvers/index.js';

console.log('PathSimJS - Simple Integrator Example');
console.log('======================================');
console.log();

// Create blocks
const constantSource = new Constant(1.0);  // Constant input of 1.0
const integrator = new Integrator(0.0);    // Integrator with initial value 0
const scope = new Scope({ labels: ['Output'], numInputs: 1 });

// Create simulation
const sim = new Simulation({
    blocks: [constantSource, integrator, scope],
    connections: [
        new Connection(constantSource.getItem(0), integrator.getItem(0)),
        new Connection(integrator.getItem(0), scope.getItem(0))
    ],
    dt: 0.1,
    Solver: SSPRK22,
    log: false  // Disable verbose logging for this simple example
});

console.log('Running simulation...');
sim.run(10, true);  // Run for 10 time units

console.log();
console.log('Results:');
scope.plot();

console.log();
console.log('Expected: Linear ramp from 0 to ~10');
console.log('(since we integrate a constant 1.0 for 10 time units)');
