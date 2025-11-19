# PathSimJS

> **Note:** PathSimJS is a JavaScript port of the original [PathSim](https://github.com/pathsim/pathsim) Python library. This is a conversion/fork created to bring PathSim's powerful simulation capabilities to JavaScript/Node.js environments. All credit for the original design and architecture goes to the PathSim contributors.

**PathSimJS** is a flexible block-based time-domain system simulation framework for JavaScript, ported from the Python [PathSim](https://github.com/pathsim/pathsim) library.

## Overview

PathSimJS provides tools for modeling and simulating complex interconnected dynamical systems using the block diagram paradigm. It's designed for:

- Control systems simulation
- Circuit simulation
- Physical system modeling
- Signal processing
- Dynamical systems research

## Features

- **Block-Based Modeling**: Connect discrete blocks representing system components (integrators, amplifiers, adders, etc.)
- **Continuous-Time Simulation**: Multiple numerical integrators for solving ODEs
- **Event Handling**: Support for zero-crossing detection and discrete events
- **Hierarchical Modeling**: Nested subsystems for modular design
- **Minimal Dependencies**: Pure JavaScript with no external dependencies
- **Modern ES6+**: Uses modern JavaScript features and module syntax

## Installation

Since this is a JavaScript port located in the PathSim repository, you can use it directly by importing the modules:

```javascript
import { Simulation, Connection } from './pathsimjs/index.js';
import { Integrator, Amplifier, Scope, Source } from './pathsimjs/blocks/index.js';
import { SSPRK22 } from './pathsimjs/solvers/index.js';
```

## Quick Start

Here's a simple example simulating a harmonic oscillator:

```javascript
import { Simulation, Connection } from './pathsimjs/index.js';
import { Integrator, Amplifier, Adder, Scope } from './pathsimjs/blocks/index.js';
import { SSPRK22 } from './pathsimjs/solvers/index.js';

// Create blocks
const pos = new Integrator(1.0);      // Position integrator (initial = 1.0)
const vel = new Integrator(0.0);      // Velocity integrator (initial = 0.0)
const damping = new Amplifier(-0.1);  // Damping coefficient
const spring = new Amplifier(-1.0);   // Spring stiffness
const adder = new Adder('+-');        // Sum with subtraction
const scope = new Scope({ labels: ['Position', 'Velocity'], numInputs: 2 });

// Create simulation
const sim = new Simulation({
    blocks: [pos, vel, damping, spring, adder, scope],
    connections: [
        new Connection(pos.getItem(0), vel.getItem(0)),  // Position -> Velocity integrator input
        new Connection(vel.getItem(0), damping.getItem(0), scope.getItem(1)),  // Velocity -> Damping & Scope
        new Connection(pos.getItem(0), spring.getItem(0), scope.getItem(0)),   // Position -> Spring & Scope
        new Connection(damping.getItem(0), adder.getItem(0)),  // Damping -> Adder
        new Connection(spring.getItem(0), adder.getItem(1)),   // Spring -> Adder
        new Connection(adder.getItem(0), vel.getItem(0))       // Adder -> Velocity integrator
    ],
    dt: 0.01,
    Solver: SSPRK22
});

// Run simulation
sim.run(10, true);  // Run for 10 time units, reset before starting

// Display results
scope.plot();

// Export data
console.log(scope.toCSV());
```

## Architecture

### Core Components

1. **Simulation**: Main simulation engine that manages blocks, connections, and time-stepping
2. **Block**: Base class for all system components (integrators, amplifiers, etc.)
3. **Connection**: Manages data flow between blocks
4. **Solver**: Numerical integration engines (SSPRK22, RK4, etc.)

### Available Blocks

- **Integrator**: Integrates input signal
- **Amplifier**: Multiplies input by constant gain
- **Adder**: Sums multiple inputs with optional sign operations
- **Source**: Time-dependent signal generator
- **Constant**: Constant value output
- **Scope**: Records and visualizes signals

### Available Solvers

- **SSPRK22**: 2nd order Strong Stability Preserving Runge-Kutta (default)

## Project Structure

```
pathsimjs/
├── index.js              # Main entry point
├── constants.js          # Global constants and defaults
├── Simulation.js         # Main simulation engine
├── Connection.js         # Connection management
├── blocks/               # Block implementations
│   ├── Block.js         # Base block class
│   ├── Integrator.js    # Integration block
│   ├── Amplifier.js     # Amplification block
│   ├── Adder.js         # Summation block
│   ├── Source.js        # Signal source blocks
│   ├── Scope.js         # Data recording block
│   └── index.js         # Block exports
├── solvers/              # Numerical integrators
│   ├── Solver.js        # Base solver class
│   ├── SSPRK22.js       # SSPRK22 solver
│   └── index.js         # Solver exports
├── utils/                # Utility classes
│   ├── register.js      # I/O register
│   ├── portreference.js # Port reference wrapper
│   ├── graph.js         # Graph analysis
│   └── index.js         # Utils exports
├── package.json          # Package configuration
├── README.md            # This file
└── examples/            # Example simulations
```

## Key Differences from Python PathSim

1. **Array Handling**: JavaScript arrays instead of NumPy arrays
2. **Module System**: ES6 modules instead of Python imports
3. **No External Dependencies**: Pure JavaScript implementation
4. **Simplified Features**: Some advanced features from the Python version are not yet implemented

## Current Limitations

This is an initial port of PathSim to JavaScript. Some features from the Python version are not yet available:

- Advanced solvers (implicit methods, adaptive stepping)
- Event system (zero-crossing detection)
- Subsystems and hierarchical modeling
- Linearization capabilities
- Serialization/deserialization
- Most block types (only basic blocks are implemented)

## Contributing

This is a port of the Python PathSim library. For issues or contributions specific to the JavaScript version, please open an issue in the main PathSim repository.

## License

MIT License - Same as the original PathSim project

## Original Project

**PathSim (Python)** - The original implementation
🔗 Repository: [https://github.com/pathsim/pathsim](https://github.com/pathsim/pathsim)
📖 Documentation: [https://pathsim.readthedocs.io/](https://pathsim.readthedocs.io/)
📦 PyPI: [https://pypi.org/project/pathsim/](https://pypi.org/project/pathsim/)

## Acknowledgments

This JavaScript port is a conversion/fork of the original Python PathSim library. All credit for the design, architecture, and concepts goes to the original PathSim contributors. This port aims to make the powerful simulation capabilities of PathSim available in JavaScript/Node.js environments while maintaining compatibility with the original API design.

### Differences from Original Python Version

This JavaScript port includes:
- ✅ Core simulation engine with fixed-point iteration
- ✅ Basic blocks (Integrator, Amplifier, Adder, Source, Constant, Scope)
- ✅ SSPRK22 solver (2nd order Runge-Kutta)
- ✅ Block interconnection system
- ✅ Data recording and export

Not yet implemented:
- ⏳ Advanced solvers (adaptive, implicit methods)
- ⏳ Event system (zero-crossing detection)
- ⏳ Subsystems and hierarchical modeling
- ⏳ Linearization capabilities
- ⏳ Most block types from the Python version
- ⏳ Serialization/deserialization

## Contributing

Since this is a port of PathSim, we aim to maintain API compatibility with the original Python version where possible. For issues or contributions:

- **PathSimJS-specific issues**: Open an issue in this repository
- **General PathSim questions/features**: Refer to the [original PathSim repository](https://github.com/pathsim/pathsim)
