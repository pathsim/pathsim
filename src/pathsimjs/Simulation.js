/**
 * MAIN SIMULATION ENGINE
 * (Simulation.js)
 *
 * This module contains the Simulation class that manages the blocks,
 * connections, events, and specific simulation methods.
 */

import {
    SIM_TIMESTEP,
    SIM_TIMESTEP_MIN,
    SIM_TIMESTEP_MAX,
    SIM_TOLERANCE_FPI,
    SIM_ITERATIONS_MAX,
    LOG_ENABLE
} from './constants.js';
import { Graph } from './utils/graph.js';

/**
 * Simulation class for transient analysis of dynamical systems
 *
 * This class performs transient analysis of dynamical systems defined by blocks
 * and connections. It manages all blocks, connections, and the timestep update.
 *
 * The global system equation is evaluated by fixed-point iteration, distributing
 * information throughout the system and making it available to all blocks.
 */
export class Simulation {
    /**
     * Create a new Simulation
     * @param {Object} config - Configuration object
     * @param {Array} [config.blocks=[]] - Blocks that define the system
     * @param {Array} [config.connections=[]] - Connections between blocks
     * @param {Array} [config.events=[]] - Event trackers
     * @param {number} [config.dt=SIM_TIMESTEP] - Simulation timestep
     * @param {number} [config.dtMin=SIM_TIMESTEP_MIN] - Minimum timestep
     * @param {number} [config.dtMax=SIM_TIMESTEP_MAX] - Maximum timestep
     * @param {Function} [config.Solver] - ODE solver class
     * @param {number} [config.toleranceFPI=SIM_TOLERANCE_FPI] - Fixed-point iteration tolerance
     * @param {number} [config.iterationsMax=SIM_ITERATIONS_MAX] - Maximum iterations
     * @param {boolean} [config.log=LOG_ENABLE] - Enable logging
     * @param {Object} [config.solverKwargs={}] - Additional solver parameters
     */
    constructor({
        blocks = [],
        connections = [],
        events = [],
        dt = SIM_TIMESTEP,
        dtMin = SIM_TIMESTEP_MIN,
        dtMax = SIM_TIMESTEP_MAX,
        Solver = null,
        toleranceFPI = SIM_TOLERANCE_FPI,
        iterationsMax = SIM_ITERATIONS_MAX,
        log = LOG_ENABLE,
        ...solverKwargs
    } = {}) {
        // System definition
        this.blocks = new Set();
        this.connections = new Set();
        this.events = new Set();

        // Simulation timestep and bounds
        this.dt = dt;
        this.dtMin = dtMin;
        this.dtMax = dtMax;

        // Numerical integrator to be used (class definition)
        this.Solver = Solver;

        // Numerical integrator instance
        this.engine = Solver ? new Solver() : null;

        // Internal system graph (initialized later)
        this.graph = null;

        // Internal algebraic loop solvers (initialized later)
        this.boosters = null;

        // Error tolerance for fixed-point loop and implicit solver
        this.toleranceFPI = toleranceFPI;

        // Additional solver parameters
        this.solverKwargs = solverKwargs;

        // Iterations for fixed-point loop
        this.iterationsMax = iterationsMax;

        // Enable logging flag
        this.log = log;

        // Initial simulation time
        this.time = 0.0;

        // Collection of blocks with internal ODE solvers
        this._blocksDyn = new Set();

        // Collection of blocks with internal events
        this._blocksEvt = new Set();

        // Flag for setting the simulation active
        this._active = true;

        // Initialize logging
        this._initializeLogger();

        // Prepare and add blocks
        for (const block of blocks) {
            this.addBlock(block, true);
        }

        // Check and add connections
        for (const connection of connections) {
            this.addConnection(connection, true);
        }

        // Check and add events
        for (const event of events) {
            this.addEvent(event);
        }

        // Check if blocks from connections are in simulation
        this._checkBlocksAreManaged();

        // Assemble the system graph for simulation
        this._assembleGraph();
    }

    /**
     * String representation of the simulation
     * @returns {string} JSON representation
     */
    toString() {
        return JSON.stringify(this.toDict(), null, 2);
    }

    /**
     * Check if block/connection/event is part of simulation
     * @param {Block|Connection|Event} other - Object to check
     * @returns {boolean} Whether object is in simulation
     */
    contains(other) {
        return this.blocks.has(other) ||
               this.connections.has(other) ||
               this.events.has(other);
    }

    /**
     * Boolean evaluation of simulation
     * @returns {boolean} Whether simulation is active
     */
    get active() {
        return this._active;
    }

    /**
     * Get size information (number of blocks and states)
     * @returns {Array} [totalBlocks, totalStates]
     */
    get size() {
        let totalN = 0;
        let totalNx = 0;
        for (const block of this.blocks) {
            const [n, nx] = block.size;
            totalN += n;
            totalNx += nx;
        }
        return [totalN, totalNx];
    }

    /**
     * Initialize logger
     * @private
     */
    _initializeLogger() {
        // Simplified logging for JavaScript
        this.logger = {
            info: (...args) => {
                if (this.log) console.log('[INFO]', ...args);
            },
            warning: (...args) => {
                if (this.log) console.warn('[WARNING]', ...args);
            },
            error: (...args) => {
                if (this.log) console.error('[ERROR]', ...args);
            }
        };

        this.logger.info(`LOGGING (log: ${this.log})`);
    }

    /**
     * Plot all blocks with visualization capabilities
     * @param  {...any} args - Arguments for plotting
     */
    plot(...args) {
        for (const block of this.blocks) {
            if (block.active) {
                block.plot(...args);
            }
        }
    }

    /**
     * Convert simulation to dictionary
     * @param {Object} metadata - Additional metadata
     * @returns {Object} Dictionary representation
     */
    toDict(metadata = {}) {
        const blocks = Array.from(this.blocks).map(b => b.toDict());
        const events = Array.from(this.events).map(e => e.toDict());
        const connections = Array.from(this.connections).map(c => c.toDict());

        return {
            type: 'Simulation',
            metadata,
            structure: {
                blocks,
                events,
                connections
            },
            params: {
                dt: this.dt,
                dtMin: this.dtMin,
                dtMax: this.dtMax,
                Solver: this.Solver ? this.Solver.name : null,
                toleranceFPI: this.toleranceFPI,
                iterationsMax: this.iterationsMax,
                ...this.solverKwargs
            }
        };
    }

    /**
     * Add a block to the simulation
     * @param {Block} block - Block to add
     * @param {boolean} [_deferGraph=false] - Defer graph construction
     */
    addBlock(block, _deferGraph = false) {
        if (this.blocks.has(block)) {
            const msg = `Block ${block} already part of simulation`;
            this.logger.error(msg);
            throw new Error(msg);
        }

        // Initialize numerical integrator of block with parent
        if (this.Solver) {
            block.setSolver(this.Solver, this.engine, this.solverKwargs);
        }

        // Add to dynamic list if solver was initialized
        if (block.engine && !this._blocksDyn.has(block)) {
            this._blocksDyn.add(block);
        }

        // Add to eventful list if internal events
        if (block.events && block.events.length > 0) {
            this._blocksEvt.add(block);
        }

        // Add block to global block list
        this.blocks.add(block);

        // If graph already exists, it needs to be rebuilt
        if (!_deferGraph && this.graph) {
            this._assembleGraph();
        }
    }

    /**
     * Add a connection to the simulation
     * @param {Connection} connection - Connection to add
     * @param {boolean} [_deferGraph=false] - Defer graph construction
     */
    addConnection(connection, _deferGraph = false) {
        if (this.connections.has(connection)) {
            const msg = `${connection} already part of simulation`;
            this.logger.error(msg);
            throw new Error(msg);
        }

        // Add connection to global connection list
        this.connections.add(connection);

        // If graph already exists, it needs to be rebuilt
        if (!_deferGraph && this.graph) {
            this._assembleGraph();
        }
    }

    /**
     * Add an event to the simulation
     * @param {Event} event - Event to add
     */
    addEvent(event) {
        if (this.events.has(event)) {
            const msg = `${event} already part of simulation`;
            this.logger.error(msg);
            throw new Error(msg);
        }

        // Add event to global event list
        this.events.add(event);
    }

    /**
     * Build the internal graph representation
     * @private
     */
    _assembleGraph() {
        const startTime = performance.now();

        // Build graph
        this.graph = new Graph(this.blocks, this.connections);

        // Create boosters for loop closing connections
        if (this.graph.hasLoops) {
            // Simplified - no boosters for now
            this.boosters = [];
        }

        const runtime = ((performance.now() - startTime) / 1000).toFixed(3);

        // Log block summary
        const numDynamic = this._blocksDyn.size;
        const numStatic = this.blocks.size - numDynamic;
        const numEventful = this._blocksEvt.size;

        this.logger.info(
            `BLOCKS (total: ${this.blocks.size}, dynamic: ${numDynamic}, ` +
            `static: ${numStatic}, eventful: ${numEventful})`
        );

        // Log graph info
        const [nodes, edges] = this.graph.size;
        const [algDepth, loopDepth] = this.graph.depth;
        this.logger.info(
            `GRAPH (nodes: ${nodes}, edges: ${edges}, alg. depth: ${algDepth}, ` +
            `loop depth: ${loopDepth}, runtime: ${runtime}s)`
        );
    }

    /**
     * Check whether blocks in connections are managed by simulation
     * @private
     */
    _checkBlocksAreManaged() {
        const connBlocks = new Set();
        for (const conn of this.connections) {
            for (const block of conn.getBlocks()) {
                connBlocks.add(block);
            }
        }

        // Check if all connection blocks are in simulation blocks
        for (const blk of connBlocks) {
            if (!this.blocks.has(blk)) {
                this.logger.warning(
                    `${blk} in 'connections' but not in 'blocks'!`
                );
            }
        }
    }

    /**
     * Reset the simulation to initial state
     * @param {number} [time=0.0] - Reset time
     */
    reset(time = 0.0) {
        this.logger.info(`RESET (time: ${time})`);

        // Set active again
        this._active = true;

        // Reset simulation time
        this.time = time;

        // Reset integration engine
        if (this.engine) {
            this.engine.reset();
        }

        // Reset all blocks to initial state
        for (const block of this.blocks) {
            block.reset();
        }

        // Reset all event managers
        for (const event of this.events) {
            event.reset();
        }

        // Evaluate system function
        this._update(this.time);
    }

    /**
     * Evaluate system equation by fixed-point iteration
     * @param {number} t - Evaluation time
     * @private
     */
    _update(t) {
        // Evaluate DAG
        this._dag(t);

        // Algebraic loops - solve them
        if (this.graph.hasLoops) {
            this._loops(t);
        }
    }

    /**
     * Update the directed acyclic graph components
     * @param {number} t - Evaluation time
     * @private
     */
    _dag(t) {
        // Perform Gauss-Seidel iterations without error checking
        for (const [_, blocksDag, connectionsDag] of this.graph.dag()) {
            // Update blocks at algebraic depth
            for (const block of blocksDag) {
                if (block.active) {
                    block.update(t);
                }
            }

            // Update connections at algebraic depth
            for (const connection of connectionsDag) {
                if (connection.active) {
                    connection.update();
                }
            }
        }
    }

    /**
     * Perform algebraic loop solve using fixed-point iterations
     * @param {number} t - Evaluation time
     * @private
     */
    _loops(t) {
        // Reset accelerators of loop closing connections
        if (this.boosters) {
            for (const conBooster of this.boosters) {
                conBooster.reset();
            }
        }

        // Perform solver iterations on algebraic loops
        for (let iteration = 1; iteration < this.iterationsMax; iteration++) {
            // Iterate DAG depths of broken loops
            for (const [_, blocksLoop, connectionsLoop] of this.graph.loop()) {
                // Update blocks at algebraic depth
                for (const block of blocksLoop) {
                    if (block.active) {
                        block.update(t);
                    }
                }

                // Update connections at algebraic depth
                for (const connection of connectionsLoop) {
                    if (connection.active) {
                        connection.update();
                    }
                }
            }

            // Step boosters of loop closing connections
            let maxErr = 0.0;
            if (this.boosters) {
                for (const conBooster of this.boosters) {
                    const err = conBooster.update();
                    if (err > maxErr) {
                        maxErr = err;
                    }
                }
            }

            // Check convergence
            if (maxErr <= this.toleranceFPI) {
                return;
            }
        }

        // Not converged - error
        const msg = `Algebraic loop not converged (iters: ${this.iterationsMax})`;
        this.logger.error(msg);
        throw new Error(msg);
    }

    /**
     * Sample data from blocks
     * @param {number} t - Sampling time
     * @param {number} dt - Timestep
     * @private
     */
    _sample(t, dt) {
        for (const block of this.blocks) {
            if (block.active) {
                block.sample(t, dt);
            }
        }
    }

    /**
     * Buffer states before timestep
     * @param {number} t - Current time
     * @param {number} dt - Timestep
     * @private
     */
    _buffer(t, dt) {
        // Buffer events
        for (const event of this.events) {
            if (event.active) {
                event.buffer(t);
            }
        }

        // Buffer the dummy engine
        if (this.engine) {
            this.engine.buffer(dt);
        }

        // Buffer internal states of stateful blocks
        for (const block of this._blocksDyn) {
            if (block.active) {
                block.buffer(dt);
            }
        }
    }

    /**
     * Advance simulation by one timestep
     * @param {number} [dt] - Timestep (uses this.dt if not specified)
     * @returns {Array} [success, errorNorm, scale, totalEvals, totalSolverIts]
     */
    timestep(dt = null) {
        if (dt === null) {
            dt = this.dt;
        }

        // Simplified timestep for explicit fixed-step solver
        let totalEvals = 0;

        // Buffer events and dynamic blocks
        this._buffer(this.time, dt);

        // If no dynamic blocks, skip the solver step
        if (this._blocksDyn.size > 0 && this.engine) {
            // Iterate explicit solver stages
            const stages = this.engine.stages(this.time, dt);
            for (const timeStage of stages) {
                // Evaluate system equation
                this._update(timeStage);
                totalEvals += 1;

                // Timestep for dynamical blocks
                for (const block of this._blocksDyn) {
                    if (block.active) {
                        block.step(timeStage, dt);
                    }
                }
            }
        }

        // System time after timestep
        const timeDt = this.time + dt;

        // Evaluate system equation before sampling
        this._update(timeDt);
        totalEvals += 1;

        // Sample data after successful timestep
        this._sample(timeDt, dt);

        // Increment global time
        this.time = timeDt;

        return [true, 0.0, 1.0, totalEvals, 0];
    }

    /**
     * Stop the simulation
     */
    stop() {
        this._active = false;
    }

    /**
     * Run simulation for a given duration
     * @param {number} [duration=10] - Simulation duration
     * @param {boolean} [reset=false] - Reset before running
     * @returns {Object} Simulation statistics
     */
    run(duration = 10, reset = false) {
        // Set simulation active
        this._active = true;

        // Reset the simulation before running it
        if (reset) {
            this.reset();
        }

        // Simulation start and end time
        const startTime = this.time;
        const endTime = this.time + duration;

        // Effective timestep for duration
        let _dt = this.dt;

        // Initial system function evaluation
        this._update(this.time);

        // Sampling states at starting time
        this._sample(this.time, this.dt);

        this.logger.info(`TRANSIENT starting (duration: ${duration})`);

        let steps = 0;
        // Timestep loop
        while (this.time < endTime && this._active) {
            // Advance the simulation by one timestep
            this.timestep(_dt);
            steps++;

            // Compute simulation progress
            const progress = Math.min((this.time - startTime) / duration, 1.0);

            // Log progress periodically
            if (steps % 100 === 0) {
                this.logger.info(`Progress: ${(progress * 100).toFixed(1)}%`);
            }
        }

        this.logger.info(`TRANSIENT finished (steps: ${steps})`);

        return {
            steps,
            finalTime: this.time
        };
    }
}
