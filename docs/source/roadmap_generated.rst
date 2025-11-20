.. raw:: html

   <div class='github-issues-container'>
   <p class='issues-updated'>Last updated: 2025-11-20 23:01 UTC</p>

   <div class='github-issue-card'>
      <div class='issue-header'>
         <span class='issue-number'>#149</span>
         <h3 class='issue-title'>Simulation class refactoring</h3>
      </div>
      <div class='issue-labels'>
         <span class='issue-label'>roadmap</span>
         <span class='issue-label'>refactor</span>
      </div>
      <div class='issue-body'>
         <p>The `Simulation` class is big (1.8k LOC) and has many responsibilities:<br><br>- managing all the blocks, connections and events and their states<br>- serialization and deserialization<br>- linearization and delinearization<br>- timestepping including system evaluation, event handling, ...<br>- 4 different timestepping strategies and 1 steady state solve method<br><br>This can probably be broken down to be more modular a...</p>
      </div>
      <div class='issue-footer'>
         <span class='issue-date'>Created: Nov 07, 2025</span>
         <a href='https://github.com/pathsim/pathsim/issues/149' class='issue-link' target='_blank'>View on GitHub →</a>
      </div>
   </div>

   <div class='github-issue-card'>
      <div class='issue-header'>
         <span class='issue-number'>#146</span>
         <h3 class='issue-title'>FSAL for ode solvers</h3>
      </div>
      <div class='issue-labels'>
         <span class='issue-label'>numerics</span>
         <span class='issue-label'>roadmap</span>
      </div>
      <div class='issue-body'>
         <p>Some runge kutta solvers such as `RKBS32` and `RKDP54` have the first-same-as-last (FSAL) property, where the last slope of the previous stage is the first slope of the next one. This saves one function evaluation per successful step and therefore increases solver efficiency. <br><br>Currently this is not implemented in pathsim, but it would give some additional 5% to 20% speedup for some solvers.</p>
      </div>
      <div class='issue-footer'>
         <span class='issue-date'>Created: Nov 04, 2025</span>
         <a href='https://github.com/pathsim/pathsim/issues/146' class='issue-link' target='_blank'>View on GitHub →</a>
      </div>
   </div>

   <div class='github-issue-card'>
      <div class='issue-header'>
         <span class='issue-number'>#143</span>
         <h3 class='issue-title'>real-time simulation</h3>
      </div>
      <div class='issue-labels'>
         <span class='issue-label'>enhancement</span>
         <span class='issue-label'>roadmap</span>
      </div>
      <div class='issue-body'>
         <p>- implement real-time mode for the simulation with a tick-rate<br>- warnings when realtime is not reached, reducing tick-rate</p>
      </div>
      <div class='issue-footer'>
         <span class='issue-date'>Created: Nov 01, 2025</span>
         <a href='https://github.com/pathsim/pathsim/issues/143' class='issue-link' target='_blank'>View on GitHub →</a>
      </div>
   </div>

   <div class='github-issue-card'>
      <div class='issue-header'>
         <span class='issue-number'>#133</span>
         <h3 class='issue-title'>More Examples</h3>
      </div>
      <div class='issue-labels'>
         <span class='issue-label'>documentation</span>
         <span class='issue-label'>good first issue</span>
         <span class='issue-label'>roadmap</span>
      </div>
      <div class='issue-body'>
         <p>You can never have enough examples. <br><br>Here is a list of ideas for example systems that want to be implemented and documented:<br>- Kalman Filter<br>- Chemical process<br>- DC motor control<br>- Inverted pendulum with controler<br>- Battery charging<br>- Phase-locked loop</p>
      </div>
      <div class='issue-footer'>
         <span class='issue-date'>Created: Okt 20, 2025</span>
         <a href='https://github.com/pathsim/pathsim/issues/133' class='issue-link' target='_blank'>View on GitHub →</a>
      </div>
   </div>

   <div class='github-issue-card'>
      <div class='issue-header'>
         <span class='issue-number'>#124</span>
         <h3 class='issue-title'>Type Hints</h3>
      </div>
      <div class='issue-labels'>
         <span class='issue-label'>documentation</span>
         <span class='issue-label'>roadmap</span>
      </div>
      <div class='issue-body'>
         <p>Currently, PathSim is not explicitly typed. Adding type hints makes sense.</p>
      </div>
      <div class='issue-footer'>
         <span class='issue-date'>Created: Okt 15, 2025</span>
         <a href='https://github.com/pathsim/pathsim/issues/124' class='issue-link' target='_blank'>View on GitHub →</a>
      </div>
   </div>

   <div class='github-issue-card'>
      <div class='issue-header'>
         <span class='issue-number'>#109</span>
         <h3 class='issue-title'>Checkpoints</h3>
      </div>
      <div class='issue-labels'>
         <span class='issue-label'>enhancement</span>
         <span class='issue-label'>roadmap</span>
      </div>
      <div class='issue-body'>
         <p>Saving simulation state as checkpoint and loading simulation state from checkpoints.</p>
      </div>
      <div class='issue-footer'>
         <span class='issue-date'>Created: Okt 09, 2025</span>
         <a href='https://github.com/pathsim/pathsim/issues/109' class='issue-link' target='_blank'>View on GitHub →</a>
      </div>
   </div>

   <div class='github-issue-card'>
      <div class='issue-header'>
         <span class='issue-number'>#105</span>
         <h3 class='issue-title'>pseudo steady state mode for dynamical blocks</h3>
      </div>
      <div class='issue-labels'>
         <span class='issue-label'>enhancement</span>
         <span class='issue-label'>numerics</span>
         <span class='issue-label'>roadmap</span>
      </div>
      <div class='issue-body'>
         <p>Sometimes in big complex systems, individual components have wildly different timescales for their physics. In some cases it makes sense to approximate components with very fast dynamics as being in a steady state at each timestep, such that the component becomes purely algebraic.<br><br>To achieve this, the time derivative of the block ode <br><br>```math<br>\dot{x} = f(x, u, t) <br>```<br><br>will be forced to zero (tr...</p>
      </div>
      <div class='issue-footer'>
         <span class='issue-date'>Created: Okt 08, 2025</span>
         <a href='https://github.com/pathsim/pathsim/issues/105' class='issue-link' target='_blank'>View on GitHub →</a>
      </div>
   </div>

   <div class='github-issue-card'>
      <div class='issue-header'>
         <span class='issue-number'>#104</span>
         <h3 class='issue-title'>Runtime System Modifications</h3>
      </div>
      <div class='issue-labels'>
         <span class='issue-label'>enhancement</span>
         <span class='issue-label'>roadmap</span>
      </div>
      <div class='issue-body'>
         <p>PathSim already supports adding and activating/deactivating blocks and connections at simulation runtime. For example through events. Whats missing is the capability to cleanly replace and remove blocks in a similar fashion.<br><br>**What this will enable:**<br>Imagine you are running a big system simulation with many (maybe hundreds) of blocks that might be small or large individual models themself. Some ...</p>
      </div>
      <div class='issue-footer'>
         <span class='issue-date'>Created: Okt 08, 2025</span>
         <a href='https://github.com/pathsim/pathsim/issues/104' class='issue-link' target='_blank'>View on GitHub →</a>
      </div>
   </div>

   <div class='github-issue-card'>
      <div class='issue-header'>
         <span class='issue-number'>#91</span>
         <h3 class='issue-title'>Asynchronous and parallel block updates</h3>
      </div>
      <div class='issue-labels'>
         <span class='issue-label'>enhancement</span>
         <span class='issue-label'>numerics</span>
         <span class='issue-label'>roadmap</span>
      </div>
      <div class='issue-body'>
         <p>PathSim has a decentralized architecture for the blocks which lends itself to parallelism and asynchronizity. Expensive blocks should compute asynchronously and not make the other blocks wait. With free-threading from Python 3.13, parallelization of the block updates is possible and has been verified with multiprocessing (slow but validation of the concept) for an earlier build.<br><br>Near linear scali...</p>
      </div>
      <div class='issue-footer'>
         <span class='issue-date'>Created: Sep 25, 2025</span>
         <a href='https://github.com/pathsim/pathsim/issues/91' class='issue-link' target='_blank'>View on GitHub →</a>
      </div>
   </div>

   <div class='github-issue-card'>
      <div class='issue-header'>
         <span class='issue-number'>#84</span>
         <h3 class='issue-title'>Copy blocks, subsystems and simulation</h3>
      </div>
      <div class='issue-labels'>
         <span class='issue-label'>enhancement</span>
         <span class='issue-label'>roadmap</span>
      </div>
      <div class='issue-body'>
         <p>Implement a `copy` method for the blocks, the `Subsystem` class, and the `Simulation`. <br><br>This should enable convenient copying of standard blocks for defining a system.</p>
      </div>
      <div class='issue-footer'>
         <span class='issue-date'>Created: Sep 15, 2025</span>
         <a href='https://github.com/pathsim/pathsim/issues/84' class='issue-link' target='_blank'>View on GitHub →</a>
      </div>
   </div>

   <div class='github-issue-card'>
      <div class='issue-header'>
         <span class='issue-number'>#82</span>
         <h3 class='issue-title'>IMEX integrators</h3>
      </div>
      <div class='issue-labels'>
         <span class='issue-label'>enhancement</span>
         <span class='issue-label'>numerics</span>
         <span class='issue-label'>roadmap</span>
      </div>
      <div class='issue-body'>
         <p>Implementing implicit-explicit ode solvers. <br><br>Some blocks in large systems may exhibit local stiffness while the coupling to external blocks is non-stiff. In these cases it would be nice to use more stable implicit ode solvers for these blocks while using explicit solvers for the other, non-stiff blocks. <br><br>The global solver would remain explicit, while locally, blocks can be flagged as stiff and t...</p>
      </div>
      <div class='issue-footer'>
         <span class='issue-date'>Created: Sep 12, 2025</span>
         <a href='https://github.com/pathsim/pathsim/issues/82' class='issue-link' target='_blank'>View on GitHub →</a>
      </div>
   </div>

   <div class='github-issue-card'>
      <div class='issue-header'>
         <span class='issue-number'>#81</span>
         <h3 class='issue-title'>exponential integrators</h3>
      </div>
      <div class='issue-labels'>
         <span class='issue-label'>enhancement</span>
         <span class='issue-label'>numerics</span>
         <span class='issue-label'>roadmap</span>
      </div>
      <div class='issue-body'>
         <p>Using exponential integrators is a way to eliminate stiffness from linear dynamical systems. Many pathsim blocks are pure linear odes such as the `StateSpace` blocks and its derivates, as well as the `Differentiator` and the `PID`. <br><br>They are more or less of the following form:<br><br>```math<br>\dot{\vec{x}} = \mathbf{A} \vec{x} + \mathbf{B} \vec{u} <br>```<br><br>Stiffness occurs when the eigenvalues of A are on ...</p>
      </div>
      <div class='issue-footer'>
         <span class='issue-date'>Created: Sep 12, 2025</span>
         <a href='https://github.com/pathsim/pathsim/issues/81' class='issue-link' target='_blank'>View on GitHub →</a>
      </div>
   </div>

   </div>

