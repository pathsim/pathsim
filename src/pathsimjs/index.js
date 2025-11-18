/**
 * PATHSIMJS - MAIN MODULE
 * (index.js)
 *
 * PathSimJS - A flexible block-based time-domain system simulation framework for JavaScript
 *
 * This is a JavaScript port of the Python PathSim library.
 * It provides tools for modeling and simulating complex interconnected dynamical systems
 * using the block diagram paradigm.
 */

// Core classes
export { Simulation } from './Simulation.js';
export { Connection, Duplex } from './Connection.js';

// Constants
export * from './constants.js';

// Blocks
export * from './blocks/index.js';

// Solvers
export * from './solvers/index.js';

// Utilities
export * from './utils/index.js';

// Version
export const VERSION = '0.1.0';
