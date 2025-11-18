/**
 * GRAPH CLASS
 * (utils/graph.js)
 *
 * Directed graph representation for analyzing block connections,
 * detecting algebraic loops, and determining evaluation order.
 */

/**
 * Graph class for managing block interconnections
 */
export class Graph {
    /**
     * Create a new Graph
     * @param {Set} blocks - Set of blocks in the system
     * @param {Set} connections - Set of connections between blocks
     */
    constructor(blocks, connections) {
        this.blocks = blocks;
        this.connections = connections;
        this.hasLoops = false;
        this.size = [0, 0];  // [nodes, edges]
        this.depth = [0, 0]; // [algebraic depth, loop depth]

        // Build the graph structure
        this._build();
    }

    /**
     * Build the internal graph structure
     * @private
     */
    _build() {
        // Count nodes and edges
        this.size = [this.blocks.size, this.connections.size];

        // Build adjacency list
        this._adjacency = new Map();
        for (const block of this.blocks) {
            this._adjacency.set(block, []);
        }

        // Populate adjacency list
        for (const conn of this.connections) {
            const sourceBlock = conn.source.block;
            for (const target of conn.targets) {
                const targetBlock = target.block;
                if (this._adjacency.has(sourceBlock)) {
                    this._adjacency.get(sourceBlock).push(targetBlock);
                }
            }
        }

        // Detect loops and calculate depths
        this._detectLoops();
        this._calculateDepth();
    }

    /**
     * Detect algebraic loops in the graph
     * @private
     */
    _detectLoops() {
        const visited = new Set();
        const recStack = new Set();

        const dfs = (block) => {
            visited.add(block);
            recStack.add(block);

            const neighbors = this._adjacency.get(block) || [];
            for (const neighbor of neighbors) {
                if (!visited.has(neighbor)) {
                    if (dfs(neighbor)) {
                        return true;
                    }
                } else if (recStack.has(neighbor)) {
                    // Loop detected
                    this.hasLoops = true;
                    return true;
                }
            }

            recStack.delete(block);
            return false;
        };

        for (const block of this.blocks) {
            if (!visited.has(block)) {
                dfs(block);
            }
        }
    }

    /**
     * Calculate algebraic depth of the graph
     * @private
     */
    _calculateDepth() {
        // Simplified depth calculation
        let maxDepth = 0;
        const depths = new Map();

        // Initialize depths
        for (const block of this.blocks) {
            depths.set(block, 0);
        }

        // Calculate depths using topological ordering
        const calculateBlockDepth = (block, visited = new Set()) => {
            if (visited.has(block)) {
                return depths.get(block) || 0;
            }
            visited.add(block);

            let maxPredDepth = 0;
            for (const conn of this.connections) {
                for (const target of conn.targets) {
                    if (target.block === block) {
                        const sourceBlock = conn.source.block;
                        const predDepth = calculateBlockDepth(sourceBlock, visited);
                        maxPredDepth = Math.max(maxPredDepth, predDepth + (sourceBlock.length || 0));
                    }
                }
            }

            depths.set(block, maxPredDepth);
            maxDepth = Math.max(maxDepth, maxPredDepth);
            return maxPredDepth;
        };

        for (const block of this.blocks) {
            calculateBlockDepth(block);
        }

        this.depth = [maxDepth, this.hasLoops ? 1 : 0];
    }

    /**
     * Get blocks organized by directed acyclic graph (DAG) levels
     * @returns {Generator} Generator yielding [depth, blocks, connections]
     */
    *dag() {
        // Group blocks by depth
        const depthGroups = new Map();
        const depths = new Map();

        // Calculate depths for each block
        for (const block of this.blocks) {
            const depth = this._getBlockDepth(block);
            depths.set(block, depth);

            if (!depthGroups.has(depth)) {
                depthGroups.set(depth, []);
            }
            depthGroups.get(depth).push(block);
        }

        // Yield blocks and connections at each depth level
        const sortedDepths = Array.from(depthGroups.keys()).sort((a, b) => a - b);
        for (const depth of sortedDepths) {
            const blocks = depthGroups.get(depth);
            const conns = this._getConnectionsAtDepth(blocks);
            yield [depth, blocks, conns];
        }
    }

    /**
     * Get depth of a block
     * @param {Block} block - Block to get depth for
     * @returns {number} Depth of block
     * @private
     */
    _getBlockDepth(block) {
        // Simplified depth calculation
        return 0;
    }

    /**
     * Get connections at a specific depth
     * @param {Array} blocks - Blocks at this depth
     * @returns {Array} Connections
     * @private
     */
    _getConnectionsAtDepth(blocks) {
        const blockSet = new Set(blocks);
        const conns = [];

        for (const conn of this.connections) {
            for (const target of conn.targets) {
                if (blockSet.has(target.block)) {
                    conns.push(conn);
                    break;
                }
            }
        }

        return conns;
    }

    /**
     * Get loop components (placeholder for loop detection)
     * @returns {Generator} Generator yielding loop components
     */
    *loop() {
        // Simplified loop iteration - just yields all blocks if loops exist
        if (this.hasLoops) {
            yield [0, Array.from(this.blocks), Array.from(this.connections)];
        }
    }

    /**
     * Get loop-closing connections
     * @returns {Array} Array of connections that close loops
     */
    loopClosingConnections() {
        // Simplified - returns empty array for now
        // Full implementation would identify specific connections that close loops
        return [];
    }
}
