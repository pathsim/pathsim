/**
 * SCOPE BLOCK
 * (blocks/Scope.js)
 *
 * Records time-domain signals for visualization and analysis.
 */

import { Block } from './Block.js';

/**
 * Scope block - records input signals over time
 *
 * Records multiple input signals and provides plotting capabilities.
 */
export class Scope extends Block {
    /**
     * Create a new Scope block
     * @param {Object} [options={}] - Configuration options
     * @param {Array<string>} [options.labels=[]] - Labels for input channels
     * @param {number} [options.numInputs=1] - Number of input channels
     */
    constructor({ labels = [], numInputs = 1 } = {}) {
        super();

        // Determine number of inputs
        this.numInputs = Math.max(labels.length, numInputs);

        // Resize inputs register
        this.inputs.resize(this.numInputs);

        // Labels for each input
        this.labels = labels.length > 0
            ? labels
            : Array.from({ length: this.numInputs }, (_, i) => `Signal ${i}`);

        // Recorded data
        this.timeData = [];
        this.signalData = Array.from({ length: this.numInputs }, () => []);
    }

    /**
     * Sample data at current time
     * @param {number} t - Current time
     * @param {number} dt - Timestep
     */
    sample(t, dt) {
        // Record time
        this.timeData.push(t);

        // Record each input signal
        for (let i = 0; i < this.numInputs; i++) {
            const value = this.inputs.get(i);
            this.signalData[i].push(value);
        }
    }

    /**
     * Reset recorded data
     */
    reset() {
        super.reset();
        this.timeData = [];
        this.signalData = Array.from({ length: this.numInputs }, () => []);
    }

    /**
     * Get recorded data
     * @returns {Object} {time, signals, labels}
     */
    getData() {
        return {
            time: this.timeData,
            signals: this.signalData,
            labels: this.labels
        };
    }

    /**
     * Plot recorded data (placeholder for visualization)
     * This would integrate with a plotting library in a real implementation
     */
    plot() {
        console.log('Scope data:');
        console.log(`  Time points: ${this.timeData.length}`);
        console.log(`  Signals: ${this.numInputs}`);

        // Display data summary
        for (let i = 0; i < this.numInputs; i++) {
            const data = this.signalData[i];
            if (data.length > 0) {
                const min = Math.min(...data);
                const max = Math.max(...data);
                const avg = data.reduce((a, b) => a + b, 0) / data.length;
                console.log(`  ${this.labels[i]}: min=${min.toFixed(4)}, max=${max.toFixed(4)}, avg=${avg.toFixed(4)}`);
            }
        }
    }

    /**
     * Export data to CSV format
     * @returns {string} CSV formatted data
     */
    toCSV() {
        const headers = ['Time', ...this.labels];
        const rows = [headers.join(',')];

        for (let i = 0; i < this.timeData.length; i++) {
            const row = [this.timeData[i]];
            for (let j = 0; j < this.numInputs; j++) {
                row.push(this.signalData[j][i]);
            }
            rows.push(row.join(','));
        }

        return rows.join('\n');
    }

    /**
     * Convert to dictionary for serialization
     * @returns {Object} Dictionary representation
     */
    toDict() {
        return {
            ...super.toDict(),
            params: {
                labels: this.labels,
                numInputs: this.numInputs
            }
        };
    }
}
