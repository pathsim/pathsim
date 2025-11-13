#!/usr/bin/env python3
"""
Create a simple test FMU for Model Exchange testing.

This creates a minimal exponential decay FMU: dx/dt = -k*x, x(0) = 1
"""

import os
import tempfile
import shutil
from pathlib import Path

TEST_DIR = Path(__file__).parent

# Simple Model Exchange FMU model description (FMI 2.0)
MODEL_DESCRIPTION = """<?xml version="1.0" encoding="UTF-8"?>
<fmiModelDescription
  fmiVersion="2.0"
  modelName="SimpleExponentialDecay"
  guid="{12345678-1234-1234-1234-123456789012}"
  description="Simple exponential decay test model: dx/dt = -x"
  generationTool="PathSim Test Generator"
  numberOfEventIndicators="0">

  <ModelExchange
    modelIdentifier="SimpleExponentialDecay"
    canGetAndSetFMUstate="true"
    canSerializeFMUstate="false"/>

  <ModelVariables>
    <ScalarVariable name="x" valueReference="0" causality="output" variability="continuous" initial="exact">
      <Real start="1.0"/>
    </ScalarVariable>
    <ScalarVariable name="der(x)" valueReference="1" causality="local" variability="continuous">
      <Real derivative="1"/>
    </ScalarVariable>
  </ModelVariables>

  <ModelStructure>
    <Outputs>
      <Unknown index="1"/>
    </Outputs>
    <Derivatives>
      <Unknown index="2"/>
    </Derivatives>
  </ModelStructure>

</fmiModelDescription>
"""

# Note: Creating a fully functional FMU requires compiled C code
# This would create the structure, but without binaries it won't work with FMPy
# For actual testing, we need to download pre-compiled FMUs

def create_fmu_structure():
    """Create FMU directory structure (without binaries)"""
    print("Note: Creating FMU structure requires compiled binaries.")
    print("For testing, please use pre-compiled reference FMUs.")
    print()
    print("You can download working Model Exchange FMUs from:")
    print("1. Install fmpy and run: fmpy create <model>")
    print("2. Download from https://github.com/modelica/Reference-FMUs/releases")
    print("3. Use FMUs from your own FMU-exporting tool (Modelica, Simulink, etc.)")


if __name__ == "__main__":
    create_fmu_structure()
