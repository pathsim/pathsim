.. _ref-examples:

Examples
========

Here we show a range of examples utilizing `PathSim` to simulate different dynamical systems and how to implement them step by step, starting from the system definition.

There is an even more comprehensive collection of example dynamical system simulations availabe in the `GitHub repository <https://github.com/milanofthe/pathsim/blob/master/examples>`_.


.. note::
   Examples are available as interactive Jupyter notebooks that can be downloaded and executed directly.


----

Fundamental Systems
-------------------

Basic examples demonstrating core PathSim concepts with linear and nonlinear systems.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📐 Linear Feedback
      :link: examples/linear_feedback
      :link-type: doc

      First-order linear feedback system demonstrating basic block connections and simulation setup.

   .. grid-item-card:: 🌊 Harmonic Oscillator
      :link: examples/harmonic_oscillator
      :link-type: doc

      Damped spring-mass-damper system with second-order dynamics and exponential decay.

   .. grid-item-card:: 🔗 Coupled Oscillators
      :link: examples/coupled_oscillators
      :link-type: doc

      Two spring-coupled spring-mass-damper systems with second-order dynamics.

   .. grid-item-card:: ⚙️ Pendulum
      :link: examples/pendulum
      :link-type: doc

      Nonlinear mathematical pendulum demonstrating the sine nonlinearity and oscillatory behavior.

   .. grid-item-card:: 🌀 Van der Pol Oscillator
      :link: examples/vanderpol
      :link-type: doc

      Self-oscillating system with nonlinear damping, demonstrating limit cycle behavior.

   .. grid-item-card:: 🦋 Lorenz Attractor
      :link: examples/lorenz_attractor
      :link-type: doc

      Chaotic system demonstrating sensitive dependence on initial conditions and strange attractors.

.. toctree::
   :hidden:
   :maxdepth: 1

   examples/linear_feedback.ipynb
   examples/harmonic_oscillator.ipynb
   examples/coupled_oscillators.ipynb
   examples/pendulum.ipynb
   examples/vanderpol.ipynb
   examples/lorenz_attractor.ipynb

----

Event-Driven Systems
--------------------

Hybrid dynamical systems with discrete events and zero-crossing detection.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ⚽ Bouncing Ball
      :link: examples/bouncing_ball
      :link-type: doc

      Classic hybrid system with zero-crossing events for bounce detection and velocity reversal.

   .. grid-item-card:: 🎯 Bouncing Pendulum
      :link: examples/bouncing_pendulum
      :link-type: doc

      Nonlinear pendulum with ground collisions, featuring automatic differentiation through events.

   .. grid-item-card:: 🔀 Switched Bouncing Ball
      :link: examples/switched_bouncing_ball
      :link-type: doc

      Advanced event handling with multiple events, conditional logic, and dynamic event switching.

   .. grid-item-card:: 🌡️ Thermostat
      :link: examples/thermostat
      :link-type: doc

      Temperature control system with hysteresis and on-off switching events.

   .. grid-item-card:: 🔧 Stick-Slip Friction
      :link: examples/stick_slip
      :link-type: doc

      Friction model with stick-slip transitions demonstrating state-dependent switching.

.. toctree::
   :hidden:
   :maxdepth: 1

   examples/bouncing_ball.ipynb
   examples/bouncing_pendulum.ipynb
   examples/switched_bouncing_ball.ipynb
   examples/thermostat.ipynb
   examples/stick_slip.ipynb

----

Control Systems
---------------

Feedback control examples including PID controllers, multi-domain systems, and automotive control.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🎛️ PID Controller
      :link: examples/pid_controller
      :link-type: doc

      Classical PID feedback control of a linear plant.

   .. grid-item-card:: 🔗 Cascade Controller
      :link: examples/cascade_controller
      :link-type: doc

      Two-loop cascade control architecture with nested PID controllers and subsystems.

   .. grid-item-card:: ⚡ DC Motor Control
      :link: examples/dcmotor_control
      :link-type: doc

      Multi-domain DC motor modeling with anti-windup PID speed control and load rejection.

   .. grid-item-card:: 🚗 ABS Braking
      :link: examples/abs_braking
      :link-type: doc

      Anti-lock braking system with Pacejka tire model and slip ratio control.

.. toctree::
   :hidden:
   :maxdepth: 1

   examples/pid_controller.ipynb
   examples/cascade_controller.ipynb
   examples/dcmotor_control.ipynb
   examples/abs_braking.ipynb

----

Signal Processing & Communications
----------------------------------

Examples demonstrating frequency domain analysis, filters, and signal processing systems.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📡 FMCW Radar
      :link: examples/fmcw_radar
      :link-type: doc

      Frequency-modulated continuous-wave radar system with mixing and frequency analysis.

   .. grid-item-card:: 📊 Spectrum Analysis
      :link: examples/spectrum_analysis
      :link-type: doc

      Frequency domain analysis using the Spectrum block to recover filter frequency responses.

   .. grid-item-card:: 🔄 Transfer Function
      :link: examples/transfer_function
      :link-type: doc

      Linear system representation using poles and residues with complex conjugate dynamics.

   .. grid-item-card:: 📢 Noisy Amplifier
      :link: examples/noisy_amplifier
      :link-type: doc

      Nonlinar noisy amplifier model as a subsystem with spectral sensitivities

   .. grid-item-card:: 🎯 Kalman Filter
      :link: examples/kalman_filter
      :link-type: doc

      Optimal state estimation from noisy measurements using the Kalman filter algorithm

.. toctree::
   :hidden:
   :maxdepth: 1

   examples/fmcw_radar.ipynb
   examples/spectrum_analysis.ipynb
   examples/transfer_function.ipynb
   examples/noisy_amplifier.ipynb
   examples/kalman_filter.ipynb

----

Electronics & Circuit Systems
------------------------------

Analog and mixed-signal circuit simulations including ADCs, nonlinear components and RF networks.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 💡 Diode Circuit
      :link: examples/diode_circuit
      :link-type: doc

      Nonlinear diode characteristics with implicit solver for stiff circuit dynamics.

   .. grid-item-card:: 📈 Delta-Sigma ADC
      :link: examples/delta_sigma_adc
      :link-type: doc

      Oversampling analog-to-digital converter with noise shaping and quantization.

   .. grid-item-card:: 🔢 SAR ADC
      :link: examples/sar_adc
      :link-type: doc

      Successive approximation register ADC with binary search and comparator logic.

   .. grid-item-card:: 📡 RF Network
      :link: examples/rf_network_oneport
      :link-type: doc

      RF network with spectrum analysis. Enabled by **Scikit-rf** integration.

.. toctree::
   :hidden:
   :maxdepth: 1

   examples/diode_circuit.ipynb
   examples/delta_sigma_adc.ipynb
   examples/sar_adc.ipynb
   examples/rf_network_oneport.ipynb

----

Advanced Topics
---------------

Complex systems featuring algebraic loops, subsystems, chemical processes, automatic differentiation, and FMU co-simulation.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔁 Algebraic Loop
      :link: examples/algebraic_loop
      :link-type: doc

      Implicit system with algebraic constraints requiring iterative solvers.

   .. grid-item-card:: 🧪 Chemical Reactor
      :link: examples/chemical_reactor
      :link-type: doc

      Chemical reaction kinetics with temperature-dependent rates and nonlinear dynamics.

   .. grid-item-card:: 📦 Nested Subsystems
      :link: examples/nested_subsystems
      :link-type: doc

      Hierarchical modeling with nested subsystems for modular system design.

      Sensitivity analysis and uncertainty quantification using forward-mode automatic differentiation.

   .. grid-item-card:: 🔌 FMU Co-Simulation
      :link: examples/fmu_cosimulation
      :link-type: doc

      Integration of Functional Mock-up Units (FMU) as PathSim blocks using FMI co-simulation.

   .. grid-item-card:: ⚽ FMU ME: Bouncing Ball
      :link: examples/fmu_model_exchange_bouncing_ball
      :link-type: doc

      Model Exchange FMU with hybrid dynamics and state event detection.

   .. grid-item-card:: 🌀 FMU ME: Van der Pol
      :link: examples/fmu_model_exchange_vanderpol
      :link-type: doc

      Model Exchange FMU with nonlinear oscillations and stiff dynamics.

   .. grid-item-card:: 💫 Lorenz Poincaré Maps
      :link: examples/poincare_maps
      :link-type: doc

      Using PathSim's event system to create Poincaré maps of the chaotic Lorenz attractor.

.. toctree::
   :hidden:
   :maxdepth: 1

   examples/algebraic_loop.ipynb
   examples/chemical_reactor.ipynb
   examples/nested_subsystems.ipynb
   examples/fmu_cosimulation.ipynb
   examples/fmu_model_exchange_bouncing_ball.ipynb
   examples/fmu_model_exchange_vanderpol.ipynb
   examples/poincare_maps.ipynb
