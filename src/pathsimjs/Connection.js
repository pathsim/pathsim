/**
 * CONNECTION CLASS
 * (Connection.js)
 *
 * This module implements the Connection class that transfers data between
 * blocks and their input/output channels.
 */

import { PortReference } from './utils/portreference.js';

/**
 * Connection class for managing block interconnections
 *
 * Handles input-output relations of blocks by connecting them in a directed graph
 * and transferring data from source outputs to target inputs.
 */
export class Connection {
    /**
     * Create a new Connection
     * @param {Block|PortReference} source - Source block or port reference
     * @param {...(Block|PortReference)} targets - Target block(s) or port reference(s)
     */
    constructor(source, ...targets) {
        // Assign source block and port
        this.source = source instanceof PortReference
            ? source
            : new PortReference(source);

        // Assign target blocks and ports
        this.targets = targets.map(trg =>
            trg instanceof PortReference ? trg : new PortReference(trg)
        );

        // Flag to set connection active
        this._active = true;

        // Validate port aliases
        this._validatePorts();

        // Validate port dimensions at connection creation
        this._validateDimensions();
    }

    /**
     * String representation of the connection
     * @returns {string} JSON string representation
     */
    toString() {
        return JSON.stringify(this.toDict(), null, 2);
    }

    /**
     * Get the number of ports in the connection
     * @returns {number} Number of ports
     */
    get length() {
        return this.source.length;
    }

    /**
     * Boolean evaluation of connection
     * @returns {boolean} Whether connection is active
     */
    get active() {
        return this._active;
    }

    /**
     * Check if block is part of connection
     * @param {Block} other - Block to check
     * @returns {boolean} Whether block is in connection
     */
    contains(other) {
        return this.getBlocks().includes(other);
    }

    /**
     * Validate port dimensions match between source and targets
     * @throws {Error} If dimensions don't match
     * @private
     */
    _validateDimensions() {
        const nSrc = this.source.length;
        for (const trg of this.targets) {
            if (trg.length !== nSrc) {
                throw new Error('Source and target have different number of ports!');
            }
        }
    }

    /**
     * Validate that ports exist on blocks
     * @throws {Error} If ports don't exist
     * @private
     */
    _validatePorts() {
        this.source._validateOutputPorts();
        for (const trg of this.targets) {
            trg._validateInputPorts();
        }
    }

    /**
     * Get all unique blocks in the connection
     * @returns {Array} Array of blocks
     */
    getBlocks() {
        const blocks = [this.source.block];
        for (const trg of this.targets) {
            if (!blocks.includes(trg.block)) {
                blocks.push(trg.block);
            }
        }
        return blocks;
    }

    /**
     * Activate the connection
     */
    on() {
        this._active = true;
    }

    /**
     * Deactivate the connection
     */
    off() {
        this._active = false;
    }

    /**
     * Convert connection to dictionary for serialization
     * @returns {Object} Dictionary representation
     */
    toDict() {
        return {
            id: Symbol('connId').description,
            source: this.source.toDict(),
            targets: this.targets.map(trg => trg.toDict())
        };
    }

    /**
     * Transfer data from source to target(s)
     */
    update() {
        for (const trg of this.targets) {
            this.source.to(trg);
        }
    }
}

/**
 * Duplex connection class for bidirectional connections
 * @deprecated Will be removed in future versions
 */
export class Duplex extends Connection {
    /**
     * Create a new Duplex connection
     * @param {Block|PortReference} source - Source block or port reference
     * @param {Block|PortReference} target - Target block or port reference
     */
    constructor(source, target) {
        // Initialize source
        const src = source instanceof PortReference
            ? source
            : new PortReference(source);

        const trg = target instanceof PortReference
            ? target
            : new PortReference(target);

        // Call parent constructor
        super(src, trg);

        // Store target separately for bidirectional transfer
        this.target = trg;

        // This is required for path length estimation
        this.targets = [this.target, this.source];

        console.warn("'Duplex' will be deprecated in next release!");
    }

    /**
     * Convert duplex to dictionary for serialization
     * @returns {Object} Dictionary representation
     */
    toDict() {
        return {
            id: Symbol('duplexId').description,
            source: this.source.toDict(),
            target: this.target.toDict()
        };
    }

    /**
     * Transfer data bidirectionally between blocks
     */
    update() {
        // Bidirectional data transfer
        this.target.to(this.source);
        this.source.to(this.target);
    }
}
