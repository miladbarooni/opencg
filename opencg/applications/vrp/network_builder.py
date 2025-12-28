"""
VRP network builder.

Constructs the time-space network for VRP column generation pricing.
"""


from opencg.applications.vrp.instance import CVRPInstance, VRPTWInstance
from opencg.core.arc import ArcType
from opencg.core.network import Network
from opencg.core.node import NodeType


def build_vrp_network(instance: CVRPInstance) -> tuple[Network, dict[int, int]]:
    """
    Build a network for CVRP pricing.

    The network has:
    - Source node (artificial start)
    - Depot node (physical depot)
    - Customer nodes (one per customer)
    - Sink node (artificial end)

    Arcs:
    - Source -> Depot (start route)
    - Depot -> each Customer (leave depot)
    - Customer -> Customer (for all pairs, if feasible)
    - Customer -> Depot (return to depot)
    - Depot -> Sink (end route)

    Args:
        instance: The CVRP instance

    Returns:
        Tuple of:
        - Network: The constructed network
        - customer_node_map: Mapping from customer index (0-based) to node index
    """
    network = Network()

    # Add source and sink
    source_idx = network.add_source()
    sink_idx = network.add_sink()

    # Add depot node
    depot_idx = network.add_node(
        name="depot",
        node_type=NodeType.BASE,
        x=instance.depot[0],
        y=instance.depot[1],
        is_depot=True,
    )

    # Add customer nodes
    customer_node_map = {}  # customer_idx -> node_idx
    for i in range(instance.num_customers):
        x, y = instance.customers[i]
        demand = instance.demands[i]

        node_idx = network.add_node(
            name=f"customer_{i}",
            node_type=NodeType.GENERIC,
            x=x,
            y=y,
            demand=demand,
            customer_id=i,
        )
        customer_node_map[i] = node_idx

    # Add arcs
    # Note: We set both resource_consumption (for C++ backend) and attributes (for Python)

    # Source -> Depot (start a route)
    network.add_arc(
        source=source_idx,
        target=depot_idx,
        cost=0.0,
        resource_consumption={"capacity": 0.0},
        arc_type=ArcType.SOURCE_ARC,
    )

    # Depot -> each Customer
    for i in range(instance.num_customers):
        cust_node = customer_node_map[i]
        dist = instance.distance(0, i + 1)  # 0 is depot, i+1 is customer
        demand = instance.demands[i]

        network.add_arc(
            source=depot_idx,
            target=cust_node,
            cost=dist,
            resource_consumption={"capacity": demand},  # For C++ backend
            arc_type=ArcType.GENERIC,
            demand=demand,  # For Python resource (backward compat)
            customer_id=i,
            distance=dist,
        )

    # Customer -> Customer (all pairs)
    for i in range(instance.num_customers):
        for j in range(instance.num_customers):
            if i == j:
                continue

            # Check if this pair is feasible (combined demand <= capacity)
            # This is a simple pruning - actual feasibility depends on path
            if instance.demands[i] + instance.demands[j] > instance.vehicle_capacity:
                # Still add arc - capacity check happens during SPPRC
                pass

            src_node = customer_node_map[i]
            tgt_node = customer_node_map[j]
            dist = instance.distance(i + 1, j + 1)
            demand = instance.demands[j]

            network.add_arc(
                source=src_node,
                target=tgt_node,
                cost=dist,
                resource_consumption={"capacity": demand},  # For C++ backend
                arc_type=ArcType.GENERIC,
                demand=demand,  # For Python resource (backward compat)
                customer_id=j,
                distance=dist,
            )

    # Customer -> Depot (return)
    for i in range(instance.num_customers):
        cust_node = customer_node_map[i]
        dist = instance.distance(i + 1, 0)

        network.add_arc(
            source=cust_node,
            target=depot_idx,
            cost=dist,
            resource_consumption={"capacity": 0.0},  # No demand on return
            arc_type=ArcType.GENERIC,
            demand=0.0,  # For Python resource (backward compat)
            distance=dist,
        )

    # Depot -> Sink (end route)
    network.add_arc(
        source=depot_idx,
        target=sink_idx,
        cost=0.0,
        resource_consumption={"capacity": 0.0},
        arc_type=ArcType.SINK_ARC,
    )

    return network, customer_node_map


def build_vrp_network_with_depot_copies(
    instance: CVRPInstance
) -> tuple[Network, dict[int, int]]:
    """
    Build a network with separate depot copies for start/end.

    This variant uses:
    - Depot_start for leaving
    - Depot_end for returning

    This can be more efficient for some pricing algorithms.

    Args:
        instance: The CVRP instance

    Returns:
        Tuple of (Network, customer_node_map)
    """
    network = Network()

    # Add source and sink
    source_idx = network.add_source()
    sink_idx = network.add_sink()

    # Add depot start node
    depot_start_idx = network.add_node(
        name="depot_start",
        node_type=NodeType.BASE,
        x=instance.depot[0],
        y=instance.depot[1],
        is_depot=True,
        depot_type="start",
    )

    # Add depot end node
    depot_end_idx = network.add_node(
        name="depot_end",
        node_type=NodeType.BASE,
        x=instance.depot[0],
        y=instance.depot[1],
        is_depot=True,
        depot_type="end",
    )

    # Add customer nodes
    customer_node_map = {}
    for i in range(instance.num_customers):
        x, y = instance.customers[i]
        demand = instance.demands[i]

        node_idx = network.add_node(
            name=f"customer_{i}",
            node_type=NodeType.GENERIC,
            x=x,
            y=y,
            demand=demand,
            customer_id=i,
        )
        customer_node_map[i] = node_idx

    # Source -> Depot_start
    network.add_arc(
        source=source_idx,
        target=depot_start_idx,
        cost=0.0,
        arc_type=ArcType.SOURCE_ARC,
    )

    # Depot_start -> each Customer
    for i in range(instance.num_customers):
        cust_node = customer_node_map[i]
        dist = instance.distance(0, i + 1)

        network.add_arc(
            source=depot_start_idx,
            target=cust_node,
            cost=dist,
            arc_type=ArcType.GENERIC,
            demand=instance.demands[i],
            customer_id=i,
            distance=dist,
        )

    # Customer -> Customer
    for i in range(instance.num_customers):
        for j in range(instance.num_customers):
            if i == j:
                continue

            src_node = customer_node_map[i]
            tgt_node = customer_node_map[j]
            dist = instance.distance(i + 1, j + 1)

            network.add_arc(
                source=src_node,
                target=tgt_node,
                cost=dist,
                arc_type=ArcType.GENERIC,
                demand=instance.demands[j],
                customer_id=j,
                distance=dist,
            )

    # Customer -> Depot_end
    for i in range(instance.num_customers):
        cust_node = customer_node_map[i]
        dist = instance.distance(i + 1, 0)

        network.add_arc(
            source=cust_node,
            target=depot_end_idx,
            cost=dist,
            arc_type=ArcType.GENERIC,
            demand=0.0,
            distance=dist,
        )

    # Depot_end -> Sink
    network.add_arc(
        source=depot_end_idx,
        target=sink_idx,
        cost=0.0,
        arc_type=ArcType.SINK_ARC,
    )

    return network, customer_node_map


def build_vrptw_network(instance: VRPTWInstance) -> tuple[Network, dict[int, int]]:
    """
    Build a network for VRPTW (VRP with Time Windows) pricing.

    Similar to CVRP network, but arcs include time-related attributes:
    - travel_time: Time to traverse the arc
    - service_time: Service time at target
    - earliest: Earliest arrival at target
    - latest: Latest arrival at target

    Args:
        instance: The VRPTW instance

    Returns:
        Tuple of:
        - Network: The constructed network
        - customer_node_map: Mapping from customer index (0-based) to node index
    """
    network = Network()

    # Add source and sink
    source_idx = network.add_source()
    sink_idx = network.add_sink()

    # Get depot time window
    depot_earliest, depot_latest = instance.depot_time_window

    # Add depot node
    depot_idx = network.add_node(
        name="depot",
        node_type=NodeType.BASE,
        x=instance.depot[0],
        y=instance.depot[1],
        is_depot=True,
        earliest=depot_earliest,
        latest=depot_latest,
    )

    # Add customer nodes
    customer_node_map = {}
    for i in range(instance.num_customers):
        x, y = instance.customers[i]
        demand = instance.demands[i]
        earliest, latest = instance.time_windows[i]
        service_time = instance.service_times[i]

        node_idx = network.add_node(
            name=f"customer_{i}",
            node_type=NodeType.GENERIC,
            x=x,
            y=y,
            demand=demand,
            customer_id=i,
            earliest=earliest,
            latest=latest,
            service_time=service_time,
        )
        customer_node_map[i] = node_idx

    # Source -> Depot (start a route)
    network.add_arc(
        source=source_idx,
        target=depot_idx,
        cost=0.0,
        resource_consumption={"capacity": 0.0},  # For C++ backend
        arc_type=ArcType.SOURCE_ARC,
        travel_time=0.0,
        service_time=0.0,
        earliest=depot_earliest,
        latest=depot_latest,
    )

    # Depot -> each Customer
    for i in range(instance.num_customers):
        cust_node = customer_node_map[i]
        dist = instance.distance(0, i + 1)
        travel_time = instance.travel_time(0, i + 1)
        earliest, latest = instance.time_windows[i]
        service_time = instance.service_times[i]

        # Time window pruning: skip if impossible to reach in time
        if travel_time > latest:
            continue

        network.add_arc(
            source=depot_idx,
            target=cust_node,
            cost=dist,
            resource_consumption={"capacity": instance.demands[i]},  # For C++ backend
            arc_type=ArcType.GENERIC,
            demand=instance.demands[i],  # For Python resource (backward compat)
            customer_id=i,
            distance=dist,
            travel_time=travel_time,
            service_time=service_time,
            earliest=earliest,
            latest=latest,
        )

    # Customer -> Customer (all pairs)
    for i in range(instance.num_customers):
        for j in range(instance.num_customers):
            if i == j:
                continue

            src_node = customer_node_map[i]
            tgt_node = customer_node_map[j]
            dist = instance.distance(i + 1, j + 1)
            travel_time = instance.travel_time(i + 1, j + 1)

            # Time window of source and target
            src_earliest, src_latest = instance.time_windows[i]
            src_service = instance.service_times[i]
            tgt_earliest, tgt_latest = instance.time_windows[j]
            tgt_service = instance.service_times[j]

            # Time window pruning: skip if can't reach j from i in time
            # Earliest departure from i = src_earliest + src_service
            # Earliest arrival at j = src_earliest + src_service + travel_time
            earliest_arrival_at_j = src_earliest + src_service + travel_time
            if earliest_arrival_at_j > tgt_latest:
                continue

            network.add_arc(
                source=src_node,
                target=tgt_node,
                cost=dist,
                resource_consumption={"capacity": instance.demands[j]},  # For C++ backend
                arc_type=ArcType.GENERIC,
                demand=instance.demands[j],  # For Python resource (backward compat)
                customer_id=j,
                distance=dist,
                travel_time=travel_time,
                service_time=tgt_service,
                earliest=tgt_earliest,
                latest=tgt_latest,
            )

    # Customer -> Depot (return)
    for i in range(instance.num_customers):
        cust_node = customer_node_map[i]
        dist = instance.distance(i + 1, 0)
        travel_time = instance.travel_time(i + 1, 0)

        # Time window pruning for return to depot
        src_earliest, _ = instance.time_windows[i]
        src_service = instance.service_times[i]
        earliest_return = src_earliest + src_service + travel_time
        if earliest_return > depot_latest:
            continue

        network.add_arc(
            source=cust_node,
            target=depot_idx,
            cost=dist,
            resource_consumption={"capacity": 0.0},  # For C++ backend
            arc_type=ArcType.GENERIC,
            demand=0.0,  # For Python resource (backward compat)
            distance=dist,
            travel_time=travel_time,
            service_time=0.0,
            earliest=depot_earliest,
            latest=depot_latest,
        )

    # Depot -> Sink (end route)
    network.add_arc(
        source=depot_idx,
        target=sink_idx,
        cost=0.0,
        resource_consumption={"capacity": 0.0},  # For C++ backend
        arc_type=ArcType.SINK_ARC,
        travel_time=0.0,
        service_time=0.0,
    )

    return network, customer_node_map
