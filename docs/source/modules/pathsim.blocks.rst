Block Library
=============

PathSim provides a comprehensive library of simulation blocks for building complex dynamical systems.

----

Signal Sources & Generators
----------------------------

Blocks for generating input signals and noise.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Sources
      :link: pathsim.blocks.sources
      :link-type: doc

      Signal generators including constant, sine, square, ramp, and pulse sources.

   .. grid-item-card:: RNG
      :link: pathsim.blocks.rng
      :link-type: doc

      Random number generators with various distributions and seeding options.

   .. grid-item-card:: Noise
      :link: pathsim.blocks.noise
      :link-type: doc

      White, pink, and colored noise sources for stochastic simulations.

----

Basic Operations
----------------

Elementary mathematical operations and transformations.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Adder
      :link: pathsim.blocks.adder
      :link-type: doc

      Multi-input addition and subtraction with configurable signs.

   .. grid-item-card:: Multiplier
      :link: pathsim.blocks.multiplier
      :link-type: doc

      Multi-input multiplication and division operations.

   .. grid-item-card:: Amplifier
      :link: pathsim.blocks.amplifier
      :link-type: doc

      Gain blocks for signal amplification and attenuation.

   .. grid-item-card:: Math
      :link: pathsim.blocks.math
      :link-type: doc

      Mathematical functions including abs, sqrt, exp, log, and trigonometric operations.

   .. grid-item-card:: Function
      :link: pathsim.blocks.function
      :link-type: doc

      Custom user-defined functions for arbitrary signal transformations.

   .. grid-item-card:: Table
      :link: pathsim.blocks.table
      :link-type: doc

      Lookup tables for nonlinear mappings and data interpolation.

----

Signal Processing
-----------------

Filters and signal conditioning blocks.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Filters
      :link: pathsim.blocks.filters
      :link-type: doc

      Butterworth lowpass, highpass, bandpass, and bandstop filters.

   .. grid-item-card:: FIR
      :link: pathsim.blocks.fir
      :link-type: doc

      Finite impulse response filters with arbitrary coefficients.

   .. grid-item-card:: Converters
      :link: pathsim.blocks.converters
      :link-type: doc

      Signal converters for unit transformations and scaling.

   .. grid-item-card:: RF
      :link: pathsim.blocks.rf
      :link-type: doc

      Radio frequency components for wireless system simulation.

----

Control & Estimation
--------------------

Controllers and state estimation algorithms.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Control
      :link: pathsim.blocks.ctrl
      :link-type: doc

      PID controllers and control algorithms for feedback systems.

   .. grid-item-card:: Kalman
      :link: pathsim.blocks.kalman
      :link-type: doc

      Kalman filter for optimal state estimation from noisy measurements.

   .. grid-item-card:: Comparator
      :link: pathsim.blocks.comparator
      :link-type: doc

      Signal comparison and threshold detection for event triggering.

   .. grid-item-card:: Relay
      :link: pathsim.blocks.relay
      :link-type: doc

      Relay with hysteresis (Schmitt trigger).

----

Dynamic Systems
---------------

Blocks for modeling continuous and discrete dynamical systems.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: LTI
      :link: pathsim.blocks.lti
      :link-type: doc

      Linear time-invariant systems with state-space and transfer function representations.

   .. grid-item-card:: ODE
      :link: pathsim.blocks.ode
      :link-type: doc

      Custom ordinary differential equations with user-defined dynamics.

   .. grid-item-card:: Dynamical System
      :link: pathsim.blocks.dynsys
      :link-type: doc

      Nonlinear dynamical systems with state and output equations.

   .. grid-item-card:: Integrator
      :link: pathsim.blocks.integrator
      :link-type: doc

      Signal integration with optional initial conditions and limits.

   .. grid-item-card:: Differentiator
      :link: pathsim.blocks.differentiator
      :link-type: doc

      Signal differentiation using numerical approximation methods.

----

Time & Sampling
---------------

Blocks for time-based operations and discrete sampling.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Delay
      :link: pathsim.blocks.delay
      :link-type: doc

      Time delays for modeling transport lags and communication delays.

   .. grid-item-card:: Sample & Hold
      :link: pathsim.blocks.samplehold
      :link-type: doc

      Sample and hold circuits for discrete-time signal processing.

   .. grid-item-card:: Switch
      :link: pathsim.blocks.switch
      :link-type: doc

      Conditional signal routing and switching based on control inputs.

   .. grid-item-card:: Counter
      :link: pathsim.blocks.counter
      :link-type: doc

      Event counters for discrete event tracking and digital logic.

----

External Models
---------------

Integration with external simulation tools and custom code.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: FMU
      :link: pathsim.blocks.fmu
      :link-type: doc

      Functional Mock-up Unit (FMU) co-simulation for FMI 2.0 and 3.0 models.

   .. grid-item-card:: Wrapper
      :link: pathsim.blocks.wrapper
      :link-type: doc

      Wrapper for external code and discrete-time implementations.

----

Analysis & Monitoring
---------------------

Tools for recording and analyzing simulation results.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Scope
      :link: pathsim.blocks.scope
      :link-type: doc

      Signal recording and visualization for time-domain analysis.

   .. grid-item-card:: Spectrum
      :link: pathsim.blocks.spectrum
      :link-type: doc

      Signal recording and visualization for frequency-domain analysis.

----

Base Classes
------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Block Base
      :link: pathsim.blocks._block
      :link-type: doc

      Base class for all simulation blocks with core functionality.

----

.. toctree::
   :hidden:
   :maxdepth: 1

   pathsim.blocks._block
   pathsim.blocks.sources
   pathsim.blocks.rng
   pathsim.blocks.noise
   pathsim.blocks.adder
   pathsim.blocks.multiplier
   pathsim.blocks.amplifier
   pathsim.blocks.math
   pathsim.blocks.function
   pathsim.blocks.filters
   pathsim.blocks.fir
   pathsim.blocks.spectrum
   pathsim.blocks.converters
   pathsim.blocks.rf
   pathsim.blocks.ctrl
   pathsim.blocks.kalman
   pathsim.blocks.comparator
   pathsim.blocks.lti
   pathsim.blocks.ode
   pathsim.blocks.dynsys
   pathsim.blocks.integrator
   pathsim.blocks.differentiator
   pathsim.blocks.delay
   pathsim.blocks.samplehold
   pathsim.blocks.switch
   pathsim.blocks.counter
   pathsim.blocks.fmu
   pathsim.blocks.wrapper
   pathsim.blocks.scope
   pathsim.blocks.table
