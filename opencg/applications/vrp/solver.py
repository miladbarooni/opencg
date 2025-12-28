"""
VRP solvers using column generation.

This module provides:
- solve_cvrp: Solve Capacitated VRP
- solve_vrptw: Solve VRP with Time Windows
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

from opencg.applications.vrp.instance import CVRPInstance, VRPTWInstance
from opencg.applications.vrp.network_builder import (
    build_vrp_network,
    build_vrptw_network,
)
from opencg.applications.vrp.resources import CapacityResource, TimeResource
from opencg.core.column import Column
from opencg.core.problem import CoverConstraint, CoverType, ObjectiveSense, Problem


@dataclass
class CVRPConfig:
    """Configuration for CVRP solver."""
    max_iterations: int = 100
    max_time: float = 600.0  # seconds
    pricing_max_columns: int = 100
    pricing_max_labels_per_node: int = 50
    solver: str = "highs"
    verbose: bool = False


@dataclass
class CVRPSolution:
    """Solution to a CVRP instance."""
    total_distance: float
    total_distance_ip: Optional[float]
    num_vehicles: int
    routes: list[list[int]]  # List of routes, each route is list of customer indices
    solve_time: float
    iterations: int
    num_columns: int
    lower_bound: float  # Capacity-based lower bound on vehicles


def _extract_route_from_column(
    column: Column,
    network,
    customer_node_map: dict[int, int]
) -> list[int]:
    """
    Extract customer sequence from a column.

    Args:
        column: The column representing a route
        network: The network
        customer_node_map: Mapping from customer idx to node idx

    Returns:
        List of customer indices in visit order
    """
    # Reverse the customer_node_map
    {v: k for k, v in customer_node_map.items()}

    # Get arc sequence from column
    route = []
    for arc_idx in column.arc_indices:
        arc = network.arcs[arc_idx]
        customer_id = arc.get_attribute('customer_id', None)
        if customer_id is not None:
            route.append(customer_id)

    return route


def _generate_greedy_routes(instance: CVRPInstance) -> list[list[int]]:
    """
    Generate initial routes using a greedy nearest-neighbor heuristic.

    Returns:
        List of routes, each route is a list of customer indices
    """
    remaining = set(range(instance.num_customers))
    routes = []

    while remaining:
        route = []
        load = 0.0
        current = 0  # Start at depot

        while remaining:
            # Find nearest feasible customer
            best_dist = float('inf')
            best_cust = None

            for cust in remaining:
                if load + instance.demands[cust] <= instance.vehicle_capacity:
                    dist = instance.distance(current, cust + 1)
                    if dist < best_dist:
                        best_dist = dist
                        best_cust = cust

            if best_cust is None:
                break  # No feasible customer - end this route

            route.append(best_cust)
            load += instance.demands[best_cust]
            remaining.remove(best_cust)
            current = best_cust + 1

        if route:
            routes.append(route)

    return routes


def _route_cost(instance: CVRPInstance, route: list[int]) -> float:
    """Compute total distance of a route."""
    if not route:
        return 0.0

    # Depot to first customer
    cost = instance.distance(0, route[0] + 1)

    # Customer to customer
    for i in range(len(route) - 1):
        cost += instance.distance(route[i] + 1, route[i + 1] + 1)

    # Last customer to depot
    cost += instance.distance(route[-1] + 1, 0)

    return cost


def _create_column_from_route(
    route: list[int],
    instance: CVRPInstance,
    network,
    customer_node_map: dict[int, int],
) -> Column:
    """
    Create a Column from a route.

    Args:
        route: List of customer indices
        instance: The CVRP instance
        network: The network
        customer_node_map: Mapping from customer idx to node idx

    Returns:
        Column representing this route
    """
    cost = _route_cost(instance, route)
    covered = frozenset(route)

    # For now, we don't track arc_indices (would need to find arcs in network)
    # This is fine for the master problem which only needs covered_items

    return Column(
        arc_indices=(),  # We'll skip arc tracking for simplicity
        cost=cost,
        covered_items=covered,
        attributes={'route': route.copy()},
    )


def solve_cvrp(
    instance: CVRPInstance,
    config: Optional[CVRPConfig] = None,
) -> CVRPSolution:
    """
    Solve a CVRP instance using column generation.

    Args:
        instance: The CVRP instance
        config: Solver configuration

    Returns:
        CVRPSolution with results
    """
    from opencg.master import HiGHSMasterProblem
    from opencg.pricing import (
        AcceleratedLabelingAlgorithm,
        PricingConfig,
    )

    if config is None:
        config = CVRPConfig()

    start_time = time.time()

    # Compute lower bound
    lower_bound = math.ceil(instance.total_demand / instance.vehicle_capacity)
    if config.verbose:
        print(f"Lower bound (capacity): {lower_bound} vehicles")

    # Build network
    network, customer_node_map = build_vrp_network(instance)
    if config.verbose:
        print(f"Network: {network.num_nodes} nodes, {network.num_arcs} arcs")

    # Create problem
    cover_constraints = []
    for i in range(instance.num_customers):
        cover_constraints.append(CoverConstraint(
            item_id=i,
            name=f"customer_{i}",
            rhs=1.0,
            is_equality=True,  # Each customer visited exactly once
        ))

    problem = Problem(
        name="CVRP",
        network=network,
        resources=[CapacityResource(instance.vehicle_capacity)],
        cover_constraints=cover_constraints,
        cover_type=CoverType.SET_PARTITIONING,
        objective_sense=ObjectiveSense.MINIMIZE,
    )

    # Create master problem
    master = HiGHSMasterProblem(problem, verbosity=1 if config.verbose else 0)

    # Generate initial columns using greedy heuristic
    greedy_routes = _generate_greedy_routes(instance)
    if config.verbose:
        greedy_cost = sum(_route_cost(instance, r) for r in greedy_routes)
        print(f"Greedy heuristic: {len(greedy_routes)} routes, cost={greedy_cost:.2f}")

    next_col_id = 0
    for route in greedy_routes:
        col = _create_column_from_route(route, instance, network, customer_node_map)
        col = col.with_id(next_col_id)
        next_col_id += 1
        master.add_column(col)

    # Add artificial columns for feasibility
    big_m = 1e6
    for i in range(instance.num_customers):
        art_col = Column(
            arc_indices=(),
            cost=big_m,
            covered_items=frozenset([i]),
            column_id=next_col_id,
            attributes={'artificial': True, 'route': [i]},
        )
        master.add_column(art_col)
        next_col_id += 1

    # Create pricing problem
    pricing_config = PricingConfig(
        max_columns=config.pricing_max_columns,
        reduced_cost_threshold=-1e-6,
        check_elementarity=True,
        use_dominance=True,
    )

    # Use C++ accelerated labeling when available
    pricing = AcceleratedLabelingAlgorithm(
        problem,
        config=pricing_config,
    )
    if config.verbose:
        print(f"Using C++ backend: {pricing.uses_cpp_backend}")

    # Column generation loop
    if config.verbose:
        print("\nColumn Generation:")
        print(f"{'Iter':>5} {'Objective':>12} {'Columns':>8} {'New':>5}")
        print("-" * 35)

    lp_sol = None
    iterations = 0

    for iteration in range(config.max_iterations):
        iterations = iteration + 1

        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= config.max_time:
            if config.verbose:
                print(f"Time limit reached ({elapsed:.1f}s)")
            break

        # Solve LP
        lp_sol = master.solve_lp()
        if lp_sol.status.name != 'OPTIMAL':
            if config.verbose:
                print(f"LP not optimal: {lp_sol.status}")
            break

        if config.verbose and (iteration % 5 == 0 or iteration < 5):
            print(f"{iteration:>5} {lp_sol.objective_value:>12.2f} "
                  f"{master.num_columns:>8} ", end="")

        # Get duals and run pricing
        duals = master.get_dual_values()
        pricing.set_dual_values(duals)
        pricing_sol = pricing.solve()

        if config.verbose and (iteration % 5 == 0 or iteration < 5):
            print(f"{len(pricing_sol.columns):>5}")

        # Check convergence
        if not pricing_sol.columns:
            if config.verbose:
                print(f"Converged at iteration {iteration}")
            break

        # Add new columns
        for col in pricing_sol.columns:
            # Extract route from column
            route = []
            for arc_idx in col.arc_indices:
                arc = network.arcs[arc_idx]
                cust_id = arc.get_attribute('customer_id', None)
                if cust_id is not None:
                    route.append(cust_id)

            # Create column with route attribute
            new_col = Column(
                arc_indices=col.arc_indices,
                cost=col.cost,
                covered_items=col.covered_items,
                column_id=next_col_id,
                attributes={'route': route},
            )
            next_col_id += 1
            master.add_column(new_col)

    solve_time = time.time() - start_time

    # Extract solution
    lp_objective = lp_sol.objective_value if lp_sol else 0

    # Solve IP
    ip_sol = master.solve_ip()
    ip_objective = None
    routes = []

    if ip_sol.status.name == 'OPTIMAL':
        ip_objective = ip_sol.objective_value

        for col_id, value in ip_sol.column_values.items():
            if value > 0.5:  # Integer solution
                col = master.get_column(col_id)
                if col and not col.attributes.get('artificial'):
                    route = col.attributes.get('route', [])
                    if route:
                        routes.append(route)

    num_vehicles = len(routes)

    return CVRPSolution(
        total_distance=lp_objective,
        total_distance_ip=ip_objective,
        num_vehicles=num_vehicles,
        routes=routes,
        solve_time=solve_time,
        iterations=iterations,
        num_columns=master.num_columns,
        lower_bound=lower_bound,
    )


# =============================================================================
# VRPTW (VRP with Time Windows)
# =============================================================================

@dataclass
class VRPTWConfig:
    """Configuration for VRPTW solver."""
    max_iterations: int = 100
    max_time: float = 600.0  # seconds
    pricing_max_columns: int = 100
    pricing_max_labels_per_node: int = 50
    solver: str = "highs"
    verbose: bool = False


@dataclass
class VRPTWSolution:
    """Solution to a VRPTW instance."""
    total_distance: float
    total_distance_ip: Optional[float]
    num_vehicles: int
    routes: list[list[int]]  # List of routes, each route is list of customer indices
    solve_time: float
    iterations: int
    num_columns: int
    lower_bound: float  # Capacity-based lower bound on vehicles


def _generate_greedy_routes_vrptw(instance: VRPTWInstance) -> list[list[int]]:
    """
    Generate initial routes for VRPTW using a time-aware greedy heuristic.

    Prioritizes customers by earliest deadline (due date).
    """
    # Sort customers by due date (latest time)
    customers_by_deadline = sorted(
        range(instance.num_customers),
        key=lambda i: instance.time_windows[i][1]
    )

    remaining = set(customers_by_deadline)
    routes = []

    while remaining:
        route = []
        load = 0.0
        current_time = 0.0
        current_loc = 0  # Depot

        while remaining:
            # Find feasible customer with earliest deadline
            best_cust = None
            best_deadline = float('inf')

            for cust in remaining:
                # Check capacity
                if load + instance.demands[cust] > instance.vehicle_capacity:
                    continue

                # Check time window
                travel_time = instance.travel_time(current_loc, cust + 1)
                arrival_time = current_time + travel_time
                earliest, latest = instance.time_windows[cust]

                if arrival_time > latest:
                    continue  # Too late

                # Check if can return to depot in time
                service_time = instance.service_times[cust]
                departure_time = max(arrival_time, earliest) + service_time
                return_time = departure_time + instance.travel_time(cust + 1, 0)
                _, depot_latest = instance.depot_time_window

                if return_time > depot_latest:
                    continue  # Can't return in time

                # Prefer earlier deadlines
                if latest < best_deadline:
                    best_deadline = latest
                    best_cust = cust

            if best_cust is None:
                break  # No feasible customer

            # Add customer to route
            travel_time = instance.travel_time(current_loc, best_cust + 1)
            arrival_time = current_time + travel_time
            earliest, _ = instance.time_windows[best_cust]
            service_time = instance.service_times[best_cust]

            route.append(best_cust)
            load += instance.demands[best_cust]
            current_time = max(arrival_time, earliest) + service_time
            current_loc = best_cust + 1
            remaining.remove(best_cust)

        if route:
            routes.append(route)

    return routes


def _route_cost_vrptw(instance: VRPTWInstance, route: list[int]) -> float:
    """Compute total distance of a VRPTW route."""
    if not route:
        return 0.0

    # Depot to first customer
    cost = instance.distance(0, route[0] + 1)

    # Customer to customer
    for i in range(len(route) - 1):
        cost += instance.distance(route[i] + 1, route[i + 1] + 1)

    # Last customer to depot
    cost += instance.distance(route[-1] + 1, 0)

    return cost


def _create_column_from_route_vrptw(
    route: list[int],
    instance: VRPTWInstance,
    network,
    customer_node_map: dict[int, int],
) -> Column:
    """Create a Column from a VRPTW route."""
    cost = _route_cost_vrptw(instance, route)
    covered = frozenset(route)

    return Column(
        arc_indices=(),
        cost=cost,
        covered_items=covered,
        attributes={'route': route.copy()},
    )


def solve_vrptw(
    instance: VRPTWInstance,
    config: Optional[VRPTWConfig] = None,
) -> VRPTWSolution:
    """
    Solve a VRPTW instance using column generation.

    Args:
        instance: The VRPTW instance
        config: Solver configuration

    Returns:
        VRPTWSolution with results
    """
    from opencg.master import HiGHSMasterProblem
    from opencg.pricing import (
        AcceleratedLabelingAlgorithm,
        PricingConfig,
    )

    if config is None:
        config = VRPTWConfig()

    start_time = time.time()

    # Compute lower bound (capacity-based)
    lower_bound = math.ceil(instance.total_demand / instance.vehicle_capacity)
    if config.verbose:
        print(f"Lower bound (capacity): {lower_bound} vehicles")

    # Build network with time window information
    network, customer_node_map = build_vrptw_network(instance)
    if config.verbose:
        print(f"Network: {network.num_nodes} nodes, {network.num_arcs} arcs")

    # Create problem with both capacity and time resources
    cover_constraints = []
    for i in range(instance.num_customers):
        cover_constraints.append(CoverConstraint(
            item_id=i,
            name=f"customer_{i}",
            rhs=1.0,
            is_equality=True,
        ))

    _, depot_latest = instance.depot_time_window
    resources = [
        CapacityResource(instance.vehicle_capacity),
        TimeResource(depot_latest=depot_latest),
    ]

    problem = Problem(
        name="VRPTW",
        network=network,
        resources=resources,
        cover_constraints=cover_constraints,
        cover_type=CoverType.SET_PARTITIONING,
        objective_sense=ObjectiveSense.MINIMIZE,
    )

    # Create master problem
    master = HiGHSMasterProblem(problem, verbosity=1 if config.verbose else 0)

    # Generate initial columns using greedy heuristic
    greedy_routes = _generate_greedy_routes_vrptw(instance)
    if config.verbose:
        greedy_cost = sum(_route_cost_vrptw(instance, r) for r in greedy_routes)
        covered = set()
        for r in greedy_routes:
            covered.update(r)
        print(f"Greedy heuristic: {len(greedy_routes)} routes, cost={greedy_cost:.2f}, "
              f"coverage={len(covered)}/{instance.num_customers}")

    next_col_id = 0
    for route in greedy_routes:
        col = _create_column_from_route_vrptw(route, instance, network, customer_node_map)
        col = col.with_id(next_col_id)
        next_col_id += 1
        master.add_column(col)

    # Add artificial columns for feasibility
    big_m = 1e6
    for i in range(instance.num_customers):
        art_col = Column(
            arc_indices=(),
            cost=big_m,
            covered_items=frozenset([i]),
            column_id=next_col_id,
            attributes={'artificial': True, 'route': [i]},
        )
        master.add_column(art_col)
        next_col_id += 1

    # Create pricing problem with time limit per call
    pricing_config = PricingConfig(
        max_columns=config.pricing_max_columns,
        reduced_cost_threshold=-1e-6,
        check_elementarity=True,
        use_dominance=True,
        max_time=10.0,  # Time limit per pricing iteration
    )

    # Use C++ accelerated labeling (with time window support)
    pricing = AcceleratedLabelingAlgorithm(
        problem,
        config=pricing_config,
    )
    if config.verbose:
        print(f"Using C++ backend: {pricing.uses_cpp_backend}")

    # Column generation loop
    if config.verbose:
        print("\nColumn Generation:")
        print(f"{'Iter':>5} {'Objective':>12} {'Columns':>8} {'New':>5}")
        print("-" * 35)

    lp_sol = None
    iterations = 0

    for iteration in range(config.max_iterations):
        iterations = iteration + 1

        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= config.max_time:
            if config.verbose:
                print(f"Time limit reached ({elapsed:.1f}s)")
            break

        # Solve LP
        lp_sol = master.solve_lp()
        if lp_sol.status.name != 'OPTIMAL':
            if config.verbose:
                print(f"LP not optimal: {lp_sol.status}")
            break

        if config.verbose and (iteration % 5 == 0 or iteration < 5):
            print(f"{iteration:>5} {lp_sol.objective_value:>12.2f} "
                  f"{master.num_columns:>8} ", end="")

        # Get duals and run pricing
        duals = master.get_dual_values()
        pricing.set_dual_values(duals)
        pricing_sol = pricing.solve()

        if config.verbose and (iteration % 5 == 0 or iteration < 5):
            print(f"{len(pricing_sol.columns):>5}")

        # Check convergence
        if not pricing_sol.columns:
            if config.verbose:
                print(f"Converged at iteration {iteration}")
            break

        # Add new columns
        for col in pricing_sol.columns:
            # Extract route from column
            route = []
            for arc_idx in col.arc_indices:
                arc = network.arcs[arc_idx]
                cust_id = arc.get_attribute('customer_id', None)
                if cust_id is not None:
                    route.append(cust_id)

            new_col = Column(
                arc_indices=col.arc_indices,
                cost=col.cost,
                covered_items=col.covered_items,
                column_id=next_col_id,
                attributes={'route': route},
            )
            next_col_id += 1
            master.add_column(new_col)

    solve_time = time.time() - start_time

    # Extract solution
    lp_objective = lp_sol.objective_value if lp_sol else 0

    # Solve IP
    ip_sol = master.solve_ip()
    ip_objective = None
    routes = []

    if ip_sol.status.name == 'OPTIMAL':
        ip_objective = ip_sol.objective_value

        for col_id, value in ip_sol.column_values.items():
            if value > 0.5:
                col = master.get_column(col_id)
                if col and not col.attributes.get('artificial'):
                    route = col.attributes.get('route', [])
                    if route:
                        routes.append(route)

    num_vehicles = len(routes)

    return VRPTWSolution(
        total_distance=lp_objective,
        total_distance_ip=ip_objective,
        num_vehicles=num_vehicles,
        routes=routes,
        solve_time=solve_time,
        iterations=iterations,
        num_columns=master.num_columns,
        lower_bound=lower_bound,
    )
