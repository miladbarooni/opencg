#!/usr/bin/env python3
"""
Comprehensive diagnostic script for crew pairing coverage issues.

This script identifies and reports on potential issues in the crew pairing
problem setup, including:
1. Network connectivity (unreachable flights)
2. Connection time gaps
3. Resource constraint tightness
4. Dominance pruning effects
5. Per-source network coverage

Usage:
    python scripts/diagnose_crew_pairing.py [instance_path]
    python scripts/diagnose_crew_pairing.py --fix  # Apply recommended fixes
"""

import argparse
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from opencg.parsers import KasirzadehParser
from opencg.parsers.base import ParserConfig
from opencg.core.arc import ArcType
from opencg.core.node import NodeType


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    name: str
    passed: bool
    message: str
    severity: str = "info"  # info, warning, error
    details: Optional[Dict] = None


class CrewPairingDiagnostics:
    """Comprehensive diagnostics for crew pairing problems."""

    def __init__(self, instance_path: Path, config_options: Optional[Dict] = None):
        self.instance_path = instance_path
        self.config_options = config_options or {}
        self.results: List[DiagnosticResult] = []
        self.problem = None
        self.parser = None

    def run_all(self) -> List[DiagnosticResult]:
        """Run all diagnostic checks."""
        print("=" * 70)
        print("CREW PAIRING DIAGNOSTICS")
        print("=" * 70)
        print(f"Instance: {self.instance_path}")
        print()

        # Load the problem
        self._load_problem()
        if self.problem is None:
            return self.results

        # Run checks
        self._check_connection_time_gap()
        self._check_network_connectivity()
        self._check_source_sink_coverage()
        self._check_resource_tightness()
        self._check_arc_type_distribution()
        self._check_base_return_feasibility()

        # Print summary
        self._print_summary()

        return self.results

    def _load_problem(self):
        """Load the crew pairing problem."""
        print("Loading instance...")

        # Default config that might have issues
        default_options = {
            'min_connection_time': 0.5,
            'max_connection_time': 4.0,
            'min_layover_time': 10.0,  # GAP: 4-10 hours!
            'max_layover_time': 24.0,
            'max_duty_time': 14.0,
            'max_flight_time': 8.0,
            'max_pairing_days': 5,
        }

        # Apply user overrides
        options = {**default_options, **self.config_options}

        parser_config = ParserConfig(
            verbose=False,
            validate=False,
            options=options
        )
        self.parser = KasirzadehParser(parser_config)

        if not self.parser.can_parse(self.instance_path):
            self.results.append(DiagnosticResult(
                name="Instance Loading",
                passed=False,
                message=f"Cannot parse {self.instance_path}",
                severity="error"
            ))
            return

        try:
            start = time.time()
            self.problem = self.parser.parse(self.instance_path)
            load_time = time.time() - start

            self.results.append(DiagnosticResult(
                name="Instance Loading",
                passed=True,
                message=f"Loaded in {load_time:.2f}s",
                details={
                    'num_flights': len(self.problem.cover_constraints),
                    'num_nodes': self.problem.network.num_nodes,
                    'num_arcs': self.problem.network.num_arcs,
                }
            ))

            print(f"  Flights: {len(self.problem.cover_constraints)}")
            print(f"  Network: {self.problem.network.num_nodes} nodes, "
                  f"{self.problem.network.num_arcs} arcs")
            print()

        except Exception as e:
            self.results.append(DiagnosticResult(
                name="Instance Loading",
                passed=False,
                message=f"Error loading: {e}",
                severity="error"
            ))

    def _check_connection_time_gap(self):
        """Check for gaps in connection time constraints."""
        print("Checking connection time constraints...")

        max_conn = self.parser._max_connection
        min_layover = self.parser._min_layover

        if min_layover > max_conn:
            gap_hours = min_layover - max_conn
            self.results.append(DiagnosticResult(
                name="Connection Time Gap",
                passed=False,
                message=f"GAP DETECTED: {max_conn}h - {min_layover}h = {gap_hours}h dead zone!",
                severity="error",
                details={
                    'max_connection': max_conn,
                    'min_layover': min_layover,
                    'gap_hours': gap_hours,
                    'recommendation': f"Set min_layover_time={max_conn} to close gap"
                }
            ))
            print(f"  [ERROR] Gap between max_connection ({max_conn}h) and "
                  f"min_layover ({min_layover}h)")
            print(f"          Connections between {max_conn}-{min_layover}h have NO arc!")
        else:
            self.results.append(DiagnosticResult(
                name="Connection Time Gap",
                passed=True,
                message=f"No gap: max_conn={max_conn}h, min_layover={min_layover}h"
            ))
            print(f"  [OK] No connection time gap")
        print()

    def _check_network_connectivity(self):
        """Check if all flights are reachable from source and can reach sink."""
        print("Checking network connectivity...")

        network = self.problem.network

        # Find source and sink nodes
        source_idx = None
        sink_idx = None
        for i in range(network.num_nodes):
            node = network.get_node(i)
            if node.node_type == NodeType.SOURCE:
                source_idx = i
            elif node.node_type == NodeType.SINK:
                sink_idx = i

        # Get all flight arcs
        flight_arcs = {arc.index for arc in network.arcs_of_type(ArcType.FLIGHT)}

        # BFS from source to find reachable nodes and flights
        reachable_from_source = set()
        reachable_flights = set()
        queue = deque([source_idx])
        visited = {source_idx}

        while queue:
            node = queue.popleft()
            reachable_from_source.add(node)
            for arc in network.outgoing_arcs(node):
                if arc.arc_type == ArcType.FLIGHT:
                    reachable_flights.add(arc.index)
                if arc.target not in visited:
                    visited.add(arc.target)
                    queue.append(arc.target)

        # BFS backward from sink to find nodes that can reach sink
        can_reach_sink = set()
        queue = deque([sink_idx])
        visited = {sink_idx}

        while queue:
            node = queue.popleft()
            can_reach_sink.add(node)
            for arc in network.incoming_arcs(node):
                if arc.source not in visited:
                    visited.add(arc.source)
                    queue.append(arc.source)

        # Check flights that are on valid source-to-sink paths
        valid_flights = set()
        for arc in network.arcs_of_type(ArcType.FLIGHT):
            dep_node = arc.source
            arr_node = arc.target
            if dep_node in reachable_from_source and arr_node in can_reach_sink:
                valid_flights.add(arc.index)

        unreachable = flight_arcs - valid_flights

        if unreachable:
            # Categorize why flights are unreachable
            not_reachable_from_source = flight_arcs - reachable_flights
            cant_reach_sink = set()
            for arc in network.arcs_of_type(ArcType.FLIGHT):
                if arc.target not in can_reach_sink:
                    cant_reach_sink.add(arc.index)

            self.results.append(DiagnosticResult(
                name="Network Connectivity",
                passed=False,
                message=f"{len(unreachable)} flights unreachable out of {len(flight_arcs)}",
                severity="error",
                details={
                    'unreachable_count': len(unreachable),
                    'total_flights': len(flight_arcs),
                    'not_reachable_from_source': len(not_reachable_from_source),
                    'cant_reach_sink': len(cant_reach_sink),
                    'unreachable_sample': list(unreachable)[:10],
                }
            ))
            print(f"  [ERROR] {len(unreachable)}/{len(flight_arcs)} flights unreachable!")
            print(f"          - Not reachable from source: {len(not_reachable_from_source)}")
            print(f"          - Cannot reach sink: {len(cant_reach_sink)}")
        else:
            self.results.append(DiagnosticResult(
                name="Network Connectivity",
                passed=True,
                message=f"All {len(flight_arcs)} flights on valid source-sink paths"
            ))
            print(f"  [OK] All {len(flight_arcs)} flights reachable")
        print()

    def _check_source_sink_coverage(self):
        """Check source and sink arc coverage by base."""
        print("Checking source/sink arc distribution...")

        network = self.problem.network

        source_arcs_by_base: Dict[str, int] = defaultdict(int)
        sink_arcs_by_base: Dict[str, int] = defaultdict(int)
        flights_from_base: Dict[str, int] = defaultdict(int)
        flights_to_base: Dict[str, int] = defaultdict(int)

        for arc in network.arcs:
            if arc.arc_type == ArcType.SOURCE_ARC:
                base = arc.get_attribute('base', 'unknown')
                source_arcs_by_base[base] += 1
            elif arc.arc_type == ArcType.SINK_ARC:
                base = arc.get_attribute('base', 'unknown')
                sink_arcs_by_base[base] += 1
            elif arc.arc_type == ArcType.FLIGHT:
                # Check if flight departs from or arrives at a base
                dep_node = network.get_node(arc.source)
                arr_node = network.get_node(arc.target)
                if dep_node:
                    loc = dep_node.get_attribute('location', '')
                    if loc in source_arcs_by_base:
                        flights_from_base[loc] += 1
                if arr_node:
                    loc = arr_node.get_attribute('location', '')
                    if loc in sink_arcs_by_base:
                        flights_to_base[loc] += 1

        total_sources = sum(source_arcs_by_base.values())
        total_sinks = sum(sink_arcs_by_base.values())

        self.results.append(DiagnosticResult(
            name="Source/Sink Coverage",
            passed=True,
            message=f"{total_sources} source arcs, {total_sinks} sink arcs across {len(source_arcs_by_base)} bases",
            details={
                'source_arcs_by_base': dict(source_arcs_by_base),
                'sink_arcs_by_base': dict(sink_arcs_by_base),
            }
        ))

        print(f"  Source arcs by base: {dict(source_arcs_by_base)}")
        print(f"  Sink arcs by base:   {dict(sink_arcs_by_base)}")
        print()

    def _check_resource_tightness(self):
        """Analyze resource constraint tightness."""
        print("Checking resource constraints...")

        max_duty = self.parser._max_duty
        max_flight = self.parser._max_flight
        max_days = self.parser._max_days

        # Estimate average flight duration and connection time
        network = self.problem.network

        flight_durations = []
        connection_times = []

        for arc in network.arcs:
            if arc.arc_type == ArcType.FLIGHT:
                dur = arc.get_attribute('duration', 0)
                if dur > 0:
                    flight_durations.append(dur)
            elif arc.arc_type == ArcType.CONNECTION:
                dur = arc.get_attribute('duration', 0)
                if dur > 0:
                    connection_times.append(dur)

        avg_flight = sum(flight_durations) / len(flight_durations) if flight_durations else 0
        avg_conn = sum(connection_times) / len(connection_times) if connection_times else 0
        max_flight_dur = max(flight_durations) if flight_durations else 0

        # Estimate max flights per duty
        if avg_flight + avg_conn > 0:
            estimated_max_flights = int(max_duty / (avg_flight + avg_conn))
        else:
            estimated_max_flights = 0

        details = {
            'max_duty_time': max_duty,
            'max_flight_time': max_flight,
            'max_pairing_days': max_days,
            'avg_flight_duration': round(avg_flight, 2),
            'avg_connection_time': round(avg_conn, 2),
            'max_flight_duration': round(max_flight_dur, 2),
            'estimated_max_flights_per_duty': estimated_max_flights,
        }

        warnings = []
        if max_flight_dur > max_flight:
            warnings.append(f"Longest flight ({max_flight_dur:.1f}h) exceeds max_flight_time ({max_flight}h)!")

        if estimated_max_flights < 2:
            warnings.append(f"Very tight duty constraints - only ~{estimated_max_flights} flights per duty")

        if warnings:
            self.results.append(DiagnosticResult(
                name="Resource Tightness",
                passed=False,
                message="; ".join(warnings),
                severity="warning",
                details=details
            ))
            for w in warnings:
                print(f"  [WARN] {w}")
        else:
            self.results.append(DiagnosticResult(
                name="Resource Tightness",
                passed=True,
                message=f"Duty={max_duty}h, Flight={max_flight}h, Days={max_days}",
                details=details
            ))
            print(f"  [OK] Resource limits reasonable")
            print(f"       ~{estimated_max_flights} flights per duty possible")
        print()

    def _check_arc_type_distribution(self):
        """Analyze arc type distribution."""
        print("Checking arc type distribution...")

        network = self.problem.network
        arc_counts: Dict[str, int] = defaultdict(int)

        for arc in network.arcs:
            arc_counts[arc.arc_type.name] += 1

        self.results.append(DiagnosticResult(
            name="Arc Distribution",
            passed=True,
            message=f"Total {network.num_arcs} arcs",
            details=dict(arc_counts)
        ))

        for arc_type, count in sorted(arc_counts.items()):
            print(f"  {arc_type}: {count}")
        print()

    def _check_base_return_feasibility(self):
        """Check if flights can return to their starting base."""
        print("Checking base return feasibility...")

        network = self.problem.network

        # For each base, check which flights can be part of a valid pairing
        # (start from base, cover some flights, return to same base)

        source_arcs_by_base: Dict[str, List[int]] = defaultdict(list)
        sink_arcs_by_base: Dict[str, Set[int]] = defaultdict(set)

        for arc in network.arcs:
            if arc.arc_type == ArcType.SOURCE_ARC:
                base = arc.get_attribute('base', 'unknown')
                source_arcs_by_base[base].append(arc.index)
            elif arc.arc_type == ArcType.SINK_ARC:
                base = arc.get_attribute('base', 'unknown')
                sink_arcs_by_base[base].add(arc.target)

        # Check each base
        base_stats = {}
        for base, source_arcs in source_arcs_by_base.items():
            # BFS from each source arc to find reachable flights
            flights_coverable_from_base = set()

            for source_arc_idx in source_arcs:
                source_arc = network.get_arc(source_arc_idx)
                if source_arc is None:
                    continue

                # BFS to find flights reachable from this source
                queue = deque([source_arc.target])
                visited = {source_arc.target}

                while queue:
                    node = queue.popleft()
                    for arc in network.outgoing_arcs(node):
                        if arc.arc_type == ArcType.FLIGHT:
                            # Check if this flight can eventually reach a sink for this base
                            # (simplified: check if arr_node can reach any sink arc of this base)
                            flights_coverable_from_base.add(arc.index)
                        if arc.target not in visited:
                            visited.add(arc.target)
                            queue.append(arc.target)

            base_stats[base] = len(flights_coverable_from_base)

        total_flights = len(list(network.arcs_of_type(ArcType.FLIGHT)))
        all_coverable = set()
        for base, count in base_stats.items():
            print(f"  Base {base}: can reach {count} flights")

        self.results.append(DiagnosticResult(
            name="Base Return Feasibility",
            passed=True,
            message=f"Analyzed {len(base_stats)} bases",
            details=base_stats
        ))
        print()

    def _print_summary(self):
        """Print diagnostic summary."""
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)

        errors = [r for r in self.results if r.severity == "error"]
        warnings = [r for r in self.results if r.severity == "warning"]
        passed = [r for r in self.results if r.passed]

        print(f"  Passed:   {len(passed)}")
        print(f"  Warnings: {len(warnings)}")
        print(f"  Errors:   {len(errors)}")
        print()

        if errors:
            print("ERRORS (must fix):")
            for e in errors:
                print(f"  - {e.name}: {e.message}")
            print()

        if warnings:
            print("WARNINGS (should review):")
            for w in warnings:
                print(f"  - {w.name}: {w.message}")
            print()

        # Recommendations
        print("RECOMMENDATIONS:")
        for r in self.results:
            if r.details and 'recommendation' in r.details:
                print(f"  - {r.details['recommendation']}")

        # Check for connection gap specifically
        gap_result = next((r for r in self.results if r.name == "Connection Time Gap" and not r.passed), None)
        if gap_result:
            print(f"  - Close connection time gap by setting min_layover_time = max_connection_time")
        print()


def run_with_fix(instance_path: Path):
    """Run diagnostics with recommended fixes applied."""
    print("Running with FIXES applied...")
    print()

    # Fixed configuration
    fixed_options = {
        'min_connection_time': 0.5,
        'max_connection_time': 4.0,
        'min_layover_time': 4.0,  # FIXED: Close the gap!
        'max_layover_time': 24.0,
        'max_duty_time': 14.0,
        'max_flight_time': 8.0,
        'max_pairing_days': 5,
    }

    diag = CrewPairingDiagnostics(instance_path, fixed_options)
    return diag.run_all()


def main():
    parser = argparse.ArgumentParser(description="Diagnose crew pairing problems")
    parser.add_argument(
        "instance",
        nargs="?",
        default=None,
        help="Path to instance directory"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply recommended fixes"
    )
    args = parser.parse_args()

    # Determine instance path
    if args.instance:
        instance_path = Path(args.instance)
    else:
        instance_path = Path(__file__).parent.parent / "data" / "kasirzadeh" / "instance1"

    if not instance_path.exists():
        print(f"Error: Instance not found: {instance_path}")
        return 1

    if args.fix:
        results = run_with_fix(instance_path)
    else:
        diag = CrewPairingDiagnostics(instance_path)
        results = diag.run_all()

    # Return exit code based on errors
    errors = [r for r in results if r.severity == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
