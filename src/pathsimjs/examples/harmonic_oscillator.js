/**
 * HARMONIC OSCILLATOR EXAMPLE
 * (examples/harmonic_oscillator.js)
 *
 * This example demonstrates a simple harmonic oscillator (spring-mass-damper system)
 * using PathSimJS.
 *
 * System equations:
 *   dx/dt = v          (velocity)
 *   dv/dt = -k*x - c*v (acceleration with spring force and damping)
 *
 * where:
 *   x = position
 *   v = velocity
 *   k = spring constant
 *   c = damping coefficient
 */

import { Simulation, Connection } from '../index.js';
import { Integrator, Amplifier, Adder, Scope } from '../blocks/index.js';
import { SSPRK22 } from '../solvers/index.js';

console.log('='.repeat(60));
console.log('PathSimJS - Harmonic Oscillator Example');
console.log('='.repeat(60));
console.log();

// System parameters
const initialPosition = 1.0;
const initialVelocity = 0.0;
const springConstant = -1.0;  // Spring stiffness (negative for restoring force)
const dampingCoeff = -0.1;     // Damping coefficient

console.log('System parameters:');
console.log(`  Initial position: ${initialPosition}`);
console.log(`  Initial velocity: ${initialVelocity}`);
console.log(`  Spring constant (k): ${-springConstant}`);
console.log(`  Damping coefficient (c): ${-dampingCoeff}`);
console.log();

// Create blocks
console.log('Creating blocks...');

// Integrators for position and velocity
const posIntegrator = new Integrator(initialPosition);
const velIntegrator = new Integrator(initialVelocity);

// Spring force: F_spring = -k * x
const springForce = new Amplifier(springConstant);

// Damping force: F_damping = -c * v
const dampingForce = new Amplifier(dampingCoeff);

// Sum forces: F_total = F_spring + F_damping
const forceAdder = new Adder('++');

// Scope to record position and velocity
const scope = new Scope({
    labels: ['Position', 'Velocity'],
    numInputs: 2
});

console.log(`  Created ${6} blocks`);
console.log();

// Create connections
console.log('Creating connections...');

const connections = [
    // Position integrator output -> spring force input
    new Connection(posIntegrator.getItem(0), springForce.getItem(0)),

    // Position integrator output -> scope input 0
    new Connection(posIntegrator.getItem(0), scope.getItem(0)),

    // Velocity integrator output -> position integrator input
    new Connection(velIntegrator.getItem(0), posIntegrator.getItem(0)),

    // Velocity integrator output -> damping force input
    new Connection(velIntegrator.getItem(0), dampingForce.getItem(0)),

    // Velocity integrator output -> scope input 1
    new Connection(velIntegrator.getItem(0), scope.getItem(1)),

    // Spring force -> force adder input 0
    new Connection(springForce.getItem(0), forceAdder.getItem(0)),

    // Damping force -> force adder input 1
    new Connection(dampingForce.getItem(0), forceAdder.getItem(1)),

    // Force adder output -> velocity integrator input
    new Connection(forceAdder.getItem(0), velIntegrator.getItem(0))
];

console.log(`  Created ${connections.length} connections`);
console.log();

// Create simulation
console.log('Initializing simulation...');

const sim = new Simulation({
    blocks: [posIntegrator, velIntegrator, springForce, dampingForce, forceAdder, scope],
    connections: connections,
    dt: 0.01,           // Timestep
    Solver: SSPRK22,    // 2nd order Runge-Kutta solver
    log: true           // Enable logging
});

console.log();

// Run simulation
const duration = 20;  // Simulate for 20 time units
console.log(`Running simulation for ${duration} time units...`);
console.log();

const startTime = performance.now();
const stats = sim.run(duration, true);  // true = reset before starting
const endTime = performance.now();

console.log();
console.log('Simulation complete!');
console.log(`  Steps: ${stats.steps}`);
console.log(`  Final time: ${stats.finalTime.toFixed(3)}`);
console.log(`  Runtime: ${((endTime - startTime) / 1000).toFixed(3)}s`);
console.log();

// Display results
console.log('Results:');
console.log('-'.repeat(60));
scope.plot();
console.log('-'.repeat(60));
console.log();

// Export data to CSV
console.log('Exporting data to CSV format:');
console.log('-'.repeat(60));
const csvData = scope.toCSV();
const lines = csvData.split('\n');
console.log(lines.slice(0, 5).join('\n'));  // Show first 5 lines
console.log('...');
console.log(lines.slice(-3).join('\n'));     // Show last 3 lines
console.log('-'.repeat(60));
console.log();

console.log('Example complete!');
console.log('='.repeat(60));
