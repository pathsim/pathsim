.. pathsim documentation master file

========================================
PathSim
========================================

.. raw:: html

   <div style="text-align: center; margin-bottom: 2rem;">
      <p style="font-size: 1.3rem; color: var(--color-foreground-secondary); max-width: 800px; margin: 0 auto;">
         A scalable block-based time-domain system simulation framework in Python with hierarchical modeling and event handling!
      </p>
   </div>

**PathSim** provides a variety of classes that enable modeling and simulating complex interconnected dynamical systems through intuitive Python scripting.

Minimal dependencies: only ``numpy``, ``scipy`` and ``matplotlib``!

.. raw:: html

   <div style="text-align: center; margin: 2rem 0;">
      <a href="https://pypi.org/project/pathsim/" style="display: inline-block; padding: 0.75rem 2rem; background: var(--color-background-secondary); color: var(--color-foreground-primary); text-decoration: none; border-radius: 0.5rem; font-weight: 600; margin: 0.5rem; border: 2px solid var(--color-background-border);">
         Install via pip
      </a>
      <a href="https://github.com/milanofthe/pathsim" style="display: inline-block; padding: 0.75rem 2rem; background: var(--color-background-secondary); color: var(--color-foreground-primary); text-decoration: none; border-radius: 0.5rem; font-weight: 600; margin: 0.5rem; border: 2px solid var(--color-background-border);">
         View on GitHub
      </a>
      <a href="https://github.com/sponsors/milanofthe" style="display: inline-block; padding: 0.75rem 2rem; background: var(--color-background-secondary); color: var(--color-foreground-primary); text-decoration: none; border-radius: 0.5rem; font-weight: 600; margin: 0.5rem; border: 2px solid var(--color-background-border);">
         ❤️ Sponsor
      </a>
   </div>

----

Key Features
============

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔄 Hot-Swappable

      Switch blocks and solvers **during simulation** for flexible experimentation and analysis.

   .. grid-item-card:: 🎯 MIMO Capable

      Blocks are inherently **Multiple Input, Multiple Output** capable for complex systems.

   .. grid-item-card:: 🔢 Numerical Integrators

      Wide range of solvers: implicit, explicit, high-order, and adaptive time-stepping.

   .. grid-item-card:: 🏗️ Modular & Hierarchical

      Build complex systems with **nested subsystems** (:class:`.Subsystem`, `example <examples/nested_subsystems.ipynb>`_).

   .. grid-item-card:: ⚡ Event Handling

      Detect and resolve discrete events with **zero-crossing detection** (`example <examples/bouncing_ball.ipynb>`_).

   .. grid-item-card:: 🔧 Extensible

      Subclass the base ``Block`` class and implement just a handful of methods.

   .. grid-item-card:: 📖 Open Source

      MIT licensed on `GitHub <https://github.com/milanofthe/pathsim>`_ - star to support development!


Quickstart
==========

Get started with PathSim in three simple steps:

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Install

      Install PathSim with pip:

      .. code-block:: bash

         pip install pathsim

      or conda:

      .. code-block:: bash

         conda install conda-forge::pathsim

   .. grid-item-card:: Build

      Create a system using blocks and connections

   .. grid-item-card:: Run

      Execute the simulation and visualize results

----

Example: Integrating a Cosine
------------------------------

Here's a simple interactive example that demonstrates PathSim basics. Click to view the full notebook:

.. grid:: 1

   .. grid-item-card:: 🚀 Quickstart Example
      :link: quickstart
      :link-type: doc
      :text-align: center

      PathSim usage - integrating a cosine function to produce a sine wave.

      .. image:: figures/sin_cos_blockdiagram_g.png
         :width: 400
         :align: center

.. toctree::
   :hidden:

   quickstart

----

Explore the Documentation
=========================

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📚 Examples
      :link: examples
      :link-type: doc

      Explore practical examples demonstrating PathSim's capabilities, from simple oscillators to complex hybrid systems with event handling and hierarchical modeling.

   .. grid-item-card:: 📖 API Reference
      :link: api
      :link-type: doc

      Complete API documentation for all PathSim classes, methods, and modules.

   .. grid-item-card:: 🛣️ Roadmap
      :link: roadmap
      :link-type: doc

      See what's planned for future releases and contribute your ideas.

   .. grid-item-card:: 🤝 Contributing
      :link: contributing
      :link-type: doc

      Learn how to contribute to PathSim development and join the community.

   .. grid-item-card:: 🔍 Index
      :link: genindex
      :link-type: ref

      Alphabetical index of all functions, classes, and terms.

.. toctree::
   :hidden:
   :maxdepth: 2

   examples
   roadmap
   contributing
   api