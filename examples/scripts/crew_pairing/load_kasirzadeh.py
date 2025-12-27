"""
Example: Loading a Kasirzadeh crew scheduling instance.

This example demonstrates how to use the OpenCG library to load
a Kasirzadeh benchmark instance and inspect its structure.

Usage:
    python examples/crew_pairing/load_kasirzadeh.py /path/to/instance1

Prerequisites:
    - Unzip one of the Kasirzadeh instances (e.g., instance1.zip)
    - The directory should contain listOfBases.csv and day_*.csv files
"""

import sys
from pathlib import Path

# Add parent directory to path (for running without installation)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from opencg.parsers import KasirzadehParser
from opencg.parsers.base import ParserConfig


def get_data_path() -> Path:
    """
    Get the data directory path.

    Priority:
    1. OPENCG_DATA_PATH environment variable
    2. Default: <project_root>/data
    """
    import os
    env_path = os.environ.get('OPENCG_DATA_PATH')
    if env_path:
        return Path(env_path)
    # Default to data/ folder in project root
    return Path(__file__).parent.parent.parent / "data"


def main():
    """Load and inspect a Kasirzadeh instance."""
    # Get instance path from command line or use default
    if len(sys.argv) > 1:
        instance_path = Path(sys.argv[1])
    else:
        # Default to data/kasirzadeh/instance1
        default_path = get_data_path() / "kasirzadeh" / "instance1"
        if not default_path.exists():
            print("Usage: python load_kasirzadeh.py /path/to/instance")
            print(f"Default path not found: {default_path}")
            print("\nYou can set OPENCG_DATA_PATH environment variable to your data folder.")
            return
        instance_path = default_path

    print(f"Loading instance from: {instance_path}")
    print("=" * 60)

    # Create parser with verbose output
    config = ParserConfig(verbose=True, validate=True)
    parser = KasirzadehParser(config)

    # Check if parser can handle this format
    if not parser.can_parse(instance_path):
        print(f"Error: Cannot parse {instance_path}")
        print("Make sure the directory contains listOfBases.csv and day_*.csv files")
        return

    # Parse the instance
    try:
        problem = parser.parse(instance_path)
    except Exception as e:
        print(f"Error parsing instance: {e}")
        raise

    # Print problem summary
    print("\n" + "=" * 60)
    print("PROBLEM SUMMARY")
    print("=" * 60)
    print(problem.summary())

    # Print network details
    print("\n" + "-" * 60)
    print("NETWORK DETAILS")
    print("-" * 60)
    print(problem.network.summary())

    # Print some sample nodes
    print("\n" + "-" * 60)
    print("SAMPLE NODES (first 10)")
    print("-" * 60)
    for node in problem.network.nodes[:10]:
        print(f"  {node}")

    # Print some sample arcs
    print("\n" + "-" * 60)
    print("SAMPLE FLIGHT ARCS (first 10)")
    print("-" * 60)
    from opencg.core.arc import ArcType
    flight_arcs = list(problem.network.arcs_of_type(ArcType.FLIGHT))[:10]
    for arc in flight_arcs:
        print(f"  {arc}")
        print(f"      Resources: {arc.resource_consumption}")

    # Print resources
    print("\n" + "-" * 60)
    print("RESOURCES")
    print("-" * 60)
    for resource in problem.resources:
        print(f"  {resource}")

    # Print cover constraints summary
    print("\n" + "-" * 60)
    print("COVER CONSTRAINTS")
    print("-" * 60)
    print(f"  Total: {problem.num_cover_constraints}")
    print(f"  Type: {problem.cover_type.name}")
    if problem.cover_constraints:
        print("  First 5:")
        for constraint in problem.cover_constraints[:5]:
            print(f"    - {constraint.name} (item_id={constraint.item_id})")

    # Validate the problem
    print("\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    errors = problem.validate()
    if errors:
        print("  Validation errors:")
        for error in errors:
            print(f"    - {error}")
    else:
        print("  Problem is valid!")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
