#!/usr/bin/env python3
"""
Validation script to compare PathSim ModelExchangeFMU against FMPy reference.

This script simulates the same FMU with both:
1. PathSim's ModelExchangeFMU block
2. FMPy's simulate_fmu function

Then compares the results to ensure PathSim produces correct outputs.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TEST_DIR = Path(__file__).parent

def find_me_fmu():
    """Find a Model Exchange FMU in the test directory"""
    try:
        from fmpy import read_model_description

        # Look for any FMU that supports Model Exchange
        for fmu_file in TEST_DIR.glob("*.fmu"):
            try:
                md = read_model_description(str(fmu_file))
                if md.modelExchange is not None:
                    print(f"Found Model Exchange FMU: {fmu_file.name}")
                    print(f"  Model: {md.modelName}")
                    print(f"  States: {md.numberOfContinuousStates}")
                    print(f"  Event Indicators: {md.numberOfEventIndicators}")
                    return str(fmu_file)
            except Exception as e:
                continue

        print("No Model Exchange FMU found in test directory.")
        print(f"Please place an ME FMU in: {TEST_DIR}")
        return None

    except ImportError:
        print("FMPy not installed. Install with: pip install fmpy")
        return None


def simulate_with_fmpy(fmu_path, stop_time=10.0, step_size=0.01):
    """Simulate FMU using FMPy's built-in simulator"""
    try:
        from fmpy import simulate_fmu

        print(f"\n{'='*60}")
        print("Simulating with FMPy (Reference)")
        print('='*60)

        result = simulate_fmu(
            fmu_path,
            stop_time=stop_time,
            step_size=step_size,
            solver='CVode',  # Use same solver family as PathSim
            output_interval=step_size
        )

        print(f"✓ FMPy simulation completed")
        print(f"  Time points: {len(result['time'])}")
        print(f"  Variables: {list(result.dtype.names)}")

        return result

    except Exception as e:
        print(f"✗ FMPy simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def simulate_with_pathsim(fmu_path, stop_time=10.0, step_size=0.01):
    """Simulate FMU using PathSim's ModelExchangeFMU"""
    try:
        from pathsim import Simulation
        from pathsim.blocks import ModelExchangeFMU, Scope
        from pathsim.solvers import RKDP54

        print(f"\n{'='*60}")
        print("Simulating with PathSim ModelExchangeFMU")
        print('='*60)

        # Create FMU block
        fmu = ModelExchangeFMU(fmu_path, verbose=True)
        sco = Scope()

        print(f"  FMU: {fmu.model_name}")
        print(f"  States: {fmu.n_states}")
        print(f"  Inputs: {fmu.inputs.n}")
        print(f"  Outputs: {fmu.outputs.n}")
        print(f"  Event Indicators: {fmu.n_event_indicators}")

        # Create simulation
        sim = Simulation(
            blocks=[fmu, sco],
            connections=[],
            dt=step_size,
            Solver=RKDP54,
            log=True
        )

        # Run simulation
        sim.run(stop_time)

        # Get results
        time, outputs = sco.read()

        print(f"✓ PathSim simulation completed")
        print(f"  Time points: {len(time)}")
        print(f"  Output channels: {len(outputs)}")

        return time, outputs, fmu

    except Exception as e:
        print(f"✗ PathSim simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def compare_results(fmpy_result, pathsim_time, pathsim_outputs, fmu):
    """Compare FMPy and PathSim results"""

    print(f"\n{'='*60}")
    print("Comparing Results")
    print('='*60)

    if fmpy_result is None or pathsim_time is None:
        print("✗ Cannot compare - one simulation failed")
        return False

    # Get FMPy time array
    fmpy_time = fmpy_result['time']

    # Find common time range
    t_max = min(fmpy_time[-1], pathsim_time[-1])

    print(f"Time range: 0 to {t_max}")
    print(f"FMPy points: {len(fmpy_time)}")
    print(f"PathSim points: {len(pathsim_time)}")

    # Get output variable names from FMPy result
    output_vars = [name for name in fmpy_result.dtype.names if name != 'time']

    print(f"\nComparing {len(output_vars)} output variables:")

    max_error = 0.0
    all_match = True

    for i, var_name in enumerate(output_vars):
        if i >= len(pathsim_outputs):
            print(f"  {var_name}: ✗ Missing in PathSim output")
            all_match = False
            continue

        # Interpolate both signals to common time base
        t_common = np.linspace(0, t_max, 1000)

        fmpy_interp = np.interp(t_common, fmpy_time, fmpy_result[var_name])
        pathsim_interp = np.interp(t_common, pathsim_time, pathsim_outputs[i])

        # Calculate errors
        abs_error = np.abs(fmpy_interp - pathsim_interp)
        max_abs_error = np.max(abs_error)
        mean_abs_error = np.mean(abs_error)

        # Relative error (avoid division by zero)
        denom = np.maximum(np.abs(fmpy_interp), 1e-10)
        rel_error = abs_error / denom
        max_rel_error = np.max(rel_error)

        max_error = max(max_error, max_abs_error)

        # Check if error is acceptable
        if max_abs_error < 1e-3 and max_rel_error < 0.01:
            status = "✓"
        else:
            status = "⚠"
            all_match = False

        print(f"  {var_name}:")
        print(f"    Max abs error: {max_abs_error:.2e} {status}")
        print(f"    Mean abs error: {mean_abs_error:.2e}")
        print(f"    Max rel error: {max_rel_error:.2%}")

    print(f"\n{'='*60}")
    if all_match:
        print("✓ VALIDATION PASSED - Results match within tolerance")
    else:
        print("⚠ VALIDATION FAILED - Significant differences detected")
    print('='*60)

    return all_match, fmpy_result, pathsim_time, pathsim_outputs


def plot_comparison(fmpy_result, pathsim_time, pathsim_outputs):
    """Plot comparison of results"""

    try:
        output_vars = [name for name in fmpy_result.dtype.names if name != 'time']
        n_vars = min(len(output_vars), len(pathsim_outputs))

        if n_vars == 0:
            return

        fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3*n_vars))
        if n_vars == 1:
            axes = [axes]

        for i, var_name in enumerate(output_vars[:n_vars]):
            ax = axes[i]

            # Plot both results
            ax.plot(fmpy_result['time'], fmpy_result[var_name],
                   'b-', label='FMPy', linewidth=2)
            ax.plot(pathsim_time, pathsim_outputs[i],
                   'r--', label='PathSim', linewidth=1.5)

            ax.set_xlabel('Time [s]')
            ax.set_ylabel(var_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = TEST_DIR / "validation_comparison.png"
        plt.savefig(plot_file, dpi=150)
        print(f"\n✓ Comparison plot saved to: {plot_file}")

        # Show plot
        plt.show()

    except Exception as e:
        print(f"Warning: Could not create plot: {e}")


def main():
    """Main validation function"""

    print("="*60)
    print("PathSim ModelExchangeFMU Validation")
    print("="*60)

    # Find FMU
    fmu_path = find_me_fmu()
    if fmu_path is None:
        print("\nTo run this validation:")
        print("1. Download a Model Exchange FMU (e.g., BouncingBall.fmu)")
        print("2. Place it in tests/evals/")
        print("3. Run this script again")
        return 1

    # Simulation parameters
    stop_time = 3.0
    step_size = 0.01

    # Simulate with FMPy
    fmpy_result = simulate_with_fmpy(fmu_path, stop_time, step_size)

    # Simulate with PathSim
    pathsim_time, pathsim_outputs, fmu = simulate_with_pathsim(fmu_path, stop_time, step_size)

    # Compare
    if fmpy_result is not None and pathsim_time is not None:
        match, fmpy_result, pathsim_time, pathsim_outputs = compare_results(
            fmpy_result, pathsim_time, pathsim_outputs, fmu
        )

        # Plot
        plot_comparison(fmpy_result, pathsim_time, pathsim_outputs)

        return 0 if match else 1
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
