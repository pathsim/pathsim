/**
 * REGISTER CLASS
 * (utils/register.js)
 *
 * This class is intended to be used for the inputs and outputs of blocks.
 * Its basic functionality is similar to a Map but with some additional methods
 * and implemented as an array for fast data transfer.
 */

/**
 * Register class for block inputs and outputs
 *
 * This class provides dict-like functionality but is backed by an array for
 * fast data transfer. It supports dynamic resizing and string aliases for ports.
 */
export class Register {
    /**
     * Create a new Register
     * @param {number} [size=1] - Initial size of the register
     * @param {Object} [mapping={}] - String aliases for integer ports
     */
    constructor(size = 1, mapping = {}) {
        this._data = new Array(size).fill(0.0);
        this._mapping = { ...mapping };
    }

    /**
     * Map string keys to integers defined in _mapping
     * @param {number|string} key - Port key to map to index
     * @returns {number} Port index
     * @private
     */
    _map(key) {
        return this._mapping[key] !== undefined ? this._mapping[key] : key;
    }

    /**
     * Identify max index from different key types
     * @param {number|Array} key - Key to analyze
     * @returns {number} Maximum index
     * @private
     */
    _getMaxIndex(key) {
        if (typeof key === 'number') {
            return key;
        } else if (Array.isArray(key)) {
            return key.length > 0 ? Math.max(...key) : -1;
        }
        return -1;
    }

    /**
     * Get the length of the register
     * @returns {number} Length of the register
     */
    get length() {
        return this._data.length;
    }

    /**
     * Get value at key index
     * @param {number|string|Array} key - Port key
     * @returns {number|Array} Value(s) at port
     */
    get(key) {
        if (typeof key === 'string') {
            key = this._map(key);
            if (typeof key !== 'number') {
                return 0.0;
            }
        }

        if (typeof key === 'number') {
            if (key < 0 || key >= this._data.length) {
                return 0.0;
            }
            return this._data[key];
        }

        if (Array.isArray(key)) {
            return key.map(k => this.get(k));
        }

        return this._data[key];
    }

    /**
     * Set value at key index
     * @param {number|string|Array} key - Port key
     * @param {number|Array} value - Value to set
     */
    set(key, value) {
        if (typeof key === 'string' && this._mapping[key] !== undefined) {
            key = this._mapping[key];
        }

        const maxIdx = this._getMaxIndex(key);
        this.resize(maxIdx + 1);

        // Convert single-element arrays to scalars
        if (Array.isArray(value) && value.length === 1) {
            value = value[0];
        }

        this._data[key] = value;
    }

    /**
     * Resize the register if needed
     * @param {number} size - New size
     */
    resize(size) {
        if (size > this._data.length) {
            const oldLength = this._data.length;
            this._data.length = size;
            this._data.fill(0.0, oldLength);
        }
    }

    /**
     * Reset all stored values to zero
     */
    reset() {
        this._data.fill(0.0);
    }

    /**
     * Returns a copy of the internal array
     * @returns {Array} Copy of register as array
     */
    toArray() {
        return [...this._data];
    }

    /**
     * Update the register values from an array in place
     * @param {number|Array} arr - Array or scalar to update register
     */
    updateFromArray(arr) {
        if (typeof arr === 'number') {
            this._data[0] = arr;
            return;
        }

        if (!Array.isArray(arr)) {
            arr = Array.isArray(arr) ? arr : [arr];
        }

        const nArr = arr.length;
        this.resize(nArr);

        for (let i = 0; i < nArr; i++) {
            this._data[i] = arr[i];
        }
    }

    /**
     * Check if a key is in mapping or is valid integer index
     * @param {number|string} key - Key to check
     * @returns {boolean} Whether key is valid
     */
    contains(key) {
        return key in this._mapping || typeof key === 'number';
    }

    /**
     * Get iterator for the register data
     * @returns {Iterator} Iterator over register data
     */
    [Symbol.iterator]() {
        return this._data[Symbol.iterator]();
    }
}
