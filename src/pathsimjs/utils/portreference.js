/**
 * PORT REFERENCE CLASS
 * (utils/portreference.js)
 *
 * This class wraps a block reference with port indices/labels to enable
 * convenient connection syntax like Block[0], Block['out'], Block[0:2], etc.
 */

/**
 * PortReference class for wrapping blocks with port specifications
 */
export class PortReference {
    /**
     * Create a new PortReference
     * @param {Block} block - The block being referenced
     * @param {Array} [ports=[0]] - Port indices or labels
     */
    constructor(block, ports = [0]) {
        this.block = block;
        this.ports = Array.isArray(ports) ? ports : [ports];
    }

    /**
     * Get the number of ports in this reference
     * @returns {number} Number of ports
     */
    get length() {
        return this.ports.length;
    }

    /**
     * Validate that output ports exist on the block
     * @throws {Error} If ports don't exist
     * @private
     */
    _validateOutputPorts() {
        const nOutputs = this.block.outputs.length;
        for (const port of this.ports) {
            const idx = typeof port === 'string'
                ? this.block.outputs._map(port)
                : port;

            if (typeof idx !== 'number' || idx < 0 || idx >= nOutputs) {
                throw new Error(
                    `Output port ${port} does not exist on block ${this.block.constructor.name}`
                );
            }
        }
    }

    /**
     * Validate that input ports exist on the block
     * @throws {Error} If ports don't exist
     * @private
     */
    _validateInputPorts() {
        const nInputs = this.block.inputs.length;
        for (const port of this.ports) {
            const idx = typeof port === 'string'
                ? this.block.inputs._map(port)
                : port;

            if (typeof idx !== 'number' || idx < 0 || idx >= nInputs) {
                throw new Error(
                    `Input port ${port} does not exist on block ${this.block.constructor.name}`
                );
            }
        }
    }

    /**
     * Transfer data from this port reference to target port reference
     * @param {PortReference} target - Target port reference
     */
    to(target) {
        // Get source output values
        const sourceValues = this.ports.map(port => {
            const idx = typeof port === 'string'
                ? this.block.outputs._map(port)
                : port;
            return this.block.outputs.get(idx);
        });

        // Set target input values
        for (let i = 0; i < this.ports.length; i++) {
            const targetPort = target.ports[i];
            const targetIdx = typeof targetPort === 'string'
                ? target.block.inputs._map(targetPort)
                : targetPort;
            target.block.inputs.set(targetIdx, sourceValues[i]);
        }
    }

    /**
     * Convert to dictionary representation for serialization
     * @returns {Object} Dictionary representation
     */
    toDict() {
        return {
            block: this.block.__id || this.block.constructor.name,
            ports: this.ports
        };
    }
}
