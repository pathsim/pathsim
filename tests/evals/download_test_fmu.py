#!/usr/bin/env python3
"""
Helper script to download reference FMUs for testing.

This script downloads standard reference FMUs from the FMI cross-check repository
for use in PathSim's Model Exchange FMU tests.
"""

import urllib.request
import os
from pathlib import Path

# Test directory
TEST_DIR = Path(__file__).parent

# Reference FMU URLs from FMI cross-check or FMPy test-FMUs repository
REFERENCE_FMUS = {
    "BouncingBall_ME.fmu": {
        "url": "https://github.com/modelica/fmi-cross-check/raw/master/fmus/2.0/me/linux64/Test-FMUs/0.0.2/BouncingBall/BouncingBall.fmu",
        "description": "Simple bouncing ball with state events (impacts)",
        "states": 2,
        "inputs": 0,
        "outputs": 2,
        "events": True
    },
    "Dahlquist_ME.fmu": {
        "url": "https://github.com/modelica/fmi-cross-check/raw/master/fmus/2.0/me/linux64/Test-FMUs/0.0.2/Dahlquist/Dahlquist.fmu",
        "description": "Simple test equation: dx/dt = -k*x",
        "states": 1,
        "inputs": 0,
        "outputs": 1,
        "events": False
    },
    "VanDerPol_ME.fmu": {
        "url": "https://github.com/modelica/fmi-cross-check/raw/master/fmus/2.0/me/linux64/Test-FMUs/0.0.2/VanDerPol/VanDerPol.fmu",
        "description": "Van der Pol oscillator (stiff ODE)",
        "states": 2,
        "inputs": 0,
        "outputs": 2,
        "events": False
    }
}


def download_fmu(name, info):
    """Download a single FMU file"""
    target_path = TEST_DIR / name

    if target_path.exists():
        print(f"✓ {name} already exists")
        return True

    try:
        print(f"Downloading {name}...")
        print(f"  Description: {info['description']}")
        print(f"  URL: {info['url']}")

        urllib.request.urlretrieve(info['url'], target_path)

        if target_path.exists():
            print(f"✓ {name} downloaded successfully")
            return True
        else:
            print(f"✗ {name} download failed")
            return False

    except Exception as e:
        print(f"✗ Error downloading {name}: {e}")
        return False


def main():
    """Download all reference FMUs"""
    print("=" * 80)
    print("PathSim Model Exchange FMU Test Downloader")
    print("=" * 80)
    print()

    success_count = 0
    total_count = len(REFERENCE_FMUS)

    for name, info in REFERENCE_FMUS.items():
        if download_fmu(name, info):
            success_count += 1
        print()

    print("=" * 80)
    print(f"Downloaded {success_count}/{total_count} FMUs successfully")
    print("=" * 80)

    if success_count > 0:
        print("\nYou can now run the Model Exchange FMU tests:")
        print("  python3 -m pytest tests/evals/test_me_fmu_system.py -v")
    else:
        print("\nNote: If downloads fail, you may need to manually download FMUs from:")
        print("  https://github.com/modelica/fmi-cross-check")
        print("  or")
        print("  https://github.com/modelica/Reference-FMUs/releases")


if __name__ == "__main__":
    main()
