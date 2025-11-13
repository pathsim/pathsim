# Model Exchange FMU Tests

This directory contains tests for PathSim's Model Exchange FMU support.

## Test FMUs Required

The Model Exchange FMU tests require actual FMU files to run. These are **not** included in the repository due to:
- Platform-specific binary dependencies (Windows/Linux/macOS)
- Large file sizes
- Licensing considerations

## How to Obtain Test FMUs

### Option 1: Download Reference FMUs (Recommended)

Download pre-compiled reference FMUs from the official Modelica Reference-FMUs repository:

```bash
cd tests/evals

# BouncingBall - Simple model with state events (ball bouncing)
curl -L -O "https://github.com/modelica/Reference-FMUs/releases/latest/download/BouncingBall.fmu"

# Dahlquist - Simple ODE test equation
curl -L -O "https://github.com/modelica/Reference-FMUs/releases/latest/download/Dahlquist.fmu"

# Van der Pol - Stiff ODE oscillator
curl -L -O "https://github.com/modelica/Reference-FMUs/releases/latest/download/VanDerPol.fmu"
```

**Note**: Check that downloaded FMUs support Model Exchange:
```python
from fmpy import read_model_description
md = read_model_description('BouncingBall.fmu')
print(f"Model Exchange supported: {md.modelExchange is not None}")
```

### Option 2: Generate from Modelica/Simulink

If you have access to FMU-exporting tools:

1. **Dymola/OpenModelica**: Export any Modelica model as FMI 2.0/3.0 Model Exchange
2. **Simulink**: Use FMU Export (requires Simulink Coder)
3. **Other tools**: Any tool supporting FMI Model Exchange standard

### Option 3: Use Your Own FMUs

Place any Model Exchange FMU files in this directory. Tests will automatically detect and use them.

## Running the Tests

### Without FMUs (Unit Tests Only)
```bash
python3 tests/pathsim/blocks/test_fmu.py
```
These tests don't require actual FMU files.

### With FMUs (Full Integration Tests)
```bash
# Validation tests
python3 tests/evals/test_me_fmu_validation.py

# System integration tests
python3 tests/evals/test_me_fmu_system.py
```

Tests will automatically skip if FMU files are not found.

## Test FMU Requirements

For comprehensive testing, FMUs should have:

| Feature | Why Needed |
|---------|------------|
| **Continuous states** | Test ODE integration |
| **Event indicators** | Test state event (zero-crossing) detection |
| **Time events** | Test scheduled event handling |
| **Inputs/outputs** | Test signal connections |

Good test candidates:
- **BouncingBall**: Has state events (impacts)
- **Dahlquist**: Simple, no events, good for basic integration
- **VanDerPol**: Stiff ODE, tests solver robustness

## Troubleshooting

### "FMU does not support Model Exchange"
Your FMU only supports Co-Simulation. Look for FMUs explicitly marked as Model Exchange (ME).

### Platform Compatibility
FMUs contain compiled binaries for specific platforms. Ensure your FMU matches your OS:
- `darwin64` for macOS
- `linux64` for Linux
- `win64` for Windows

### FMI Version
PathSim supports FMI 2.0 and 3.0. Older FMI 1.0 FMUs are not supported.

## CI/CD Note

FMU tests are skipped in CI (`os.getenv("CI") == "true"`) because they require platform-specific binaries that may not be available in all environments.

## Additional Resources

- [FMI Standard](https://fmi-standard.org/)
- [Reference FMUs](https://github.com/modelica/Reference-FMUs)
- [FMPy Documentation](https://github.com/CATIA-Systems/FMPy)
- [FMI Cross-Check](https://github.com/modelica/fmi-cross-check)
