"""
VRP instance definitions.

This module provides instance classes for vehicle routing problems:
- CVRPInstance: Capacitated VRP
- VRPTWInstance: VRP with Time Windows
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math


@dataclass
class CVRPInstance:
    """
    A Capacitated Vehicle Routing Problem instance.

    The problem consists of:
    - A depot where all vehicles start and end
    - A set of customers, each with a location and demand
    - Vehicles with limited capacity

    Attributes:
        depot: (x, y) coordinates of the depot
        customers: List of (x, y) coordinates for each customer
        demands: Demand at each customer (index matches customers)
        vehicle_capacity: Maximum capacity of each vehicle
        num_vehicles: Optional limit on number of vehicles (None = unlimited)
        name: Optional instance name
    """
    depot: Tuple[float, float]
    customers: List[Tuple[float, float]]
    demands: List[float]
    vehicle_capacity: float
    num_vehicles: Optional[int] = None
    name: Optional[str] = None

    def __post_init__(self):
        if len(self.customers) != len(self.demands):
            raise ValueError("customers and demands must have same length")
        if any(d < 0 for d in self.demands):
            raise ValueError("demands must be non-negative")
        if any(d > self.vehicle_capacity for d in self.demands):
            raise ValueError("no customer demand can exceed vehicle capacity")

    @property
    def num_customers(self) -> int:
        """Number of customers (excluding depot)."""
        return len(self.customers)

    @property
    def total_demand(self) -> float:
        """Total demand across all customers."""
        return sum(self.demands)

    @property
    def min_vehicles(self) -> int:
        """Minimum number of vehicles needed (capacity lower bound)."""
        return math.ceil(self.total_demand / self.vehicle_capacity)

    def distance(self, i: int, j: int) -> float:
        """
        Compute Euclidean distance between two nodes.

        Node 0 is the depot, nodes 1..n are customers.

        Args:
            i: Source node (0 = depot, 1..n = customers)
            j: Target node (0 = depot, 1..n = customers)

        Returns:
            Euclidean distance
        """
        loc_i = self.depot if i == 0 else self.customers[i - 1]
        loc_j = self.depot if j == 0 else self.customers[j - 1]

        dx = loc_i[0] - loc_j[0]
        dy = loc_i[1] - loc_j[1]
        return math.sqrt(dx * dx + dy * dy)

    def distance_matrix(self) -> List[List[float]]:
        """
        Compute full distance matrix.

        Returns:
            (n+1) x (n+1) matrix where entry [i][j] is distance from i to j.
            Index 0 is depot, indices 1..n are customers.
        """
        n = self.num_customers + 1  # +1 for depot
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self.distance(i, j)

        return matrix

    @classmethod
    def from_solomon(cls, filepath: str) -> 'CVRPInstance':
        """
        Load a CVRP instance from Solomon format.

        Solomon format (used for VRPTW, we ignore time windows for CVRP):
            Line 1: Instance name
            Lines 2-4: Empty or comments
            Line 5: VEHICLE section header
            Line 6: NUMBER CAPACITY
            Line 7: num_vehicles capacity
            Lines 8-9: Empty or CUSTOMER section header
            Line 10+: CUST_ID X Y DEMAND READY_TIME DUE_DATE SERVICE_TIME

        Args:
            filepath: Path to the Solomon format file

        Returns:
            CVRPInstance
        """
        import os
        name = os.path.splitext(os.path.basename(filepath))[0]

        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # Find vehicle capacity line (after VEHICLE header)
        vehicle_idx = None
        for i, line in enumerate(lines):
            if 'VEHICLE' in line.upper():
                vehicle_idx = i
                break

        if vehicle_idx is None:
            raise ValueError("Could not find VEHICLE section")

        # Parse vehicle info (skip header line "NUMBER CAPACITY")
        vehicle_line = lines[vehicle_idx + 2].split()
        num_vehicles = int(vehicle_line[0])
        capacity = float(vehicle_line[1])

        # Find customer section
        customer_idx = None
        for i, line in enumerate(lines):
            if 'CUSTOMER' in line.upper() or 'CUST NO' in line.upper():
                customer_idx = i
                break

        if customer_idx is None:
            # Try to find data section by looking for numeric lines
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    customer_idx = i - 1  # Back up one for header
                    break

        if customer_idx is None:
            raise ValueError("Could not find CUSTOMER section")

        # Skip header lines until we find data
        data_start = customer_idx + 1
        while data_start < len(lines):
            parts = lines[data_start].split()
            if len(parts) >= 4 and parts[0].replace('.', '').isdigit():
                break
            data_start += 1

        # Parse customer data
        depot = None
        customers = []
        demands = []

        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                cust_id = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3])

                if cust_id == 0:
                    depot = (x, y)
                else:
                    customers.append((x, y))
                    demands.append(demand)
            except (ValueError, IndexError):
                continue

        if depot is None:
            raise ValueError("Could not find depot (customer 0)")

        return cls(
            depot=depot,
            customers=customers,
            demands=demands,
            vehicle_capacity=capacity,
            num_vehicles=num_vehicles,
            name=name,
        )

    @classmethod
    def from_cvrplib(cls, filepath: str) -> 'CVRPInstance':
        """
        Load a CVRP instance from CVRPLIB format.

        CVRPLIB format:
            NAME : instance_name
            COMMENT : ...
            TYPE : CVRP
            DIMENSION : n
            EDGE_WEIGHT_TYPE : EUC_2D
            CAPACITY : Q
            NODE_COORD_SECTION
            1 x1 y1
            2 x2 y2
            ...
            DEMAND_SECTION
            1 0
            2 d2
            ...
            DEPOT_SECTION
            1
            -1
            EOF

        Args:
            filepath: Path to the CVRPLIB format file

        Returns:
            CVRPInstance
        """
        import os
        name = os.path.splitext(os.path.basename(filepath))[0]

        with open(filepath, 'r') as f:
            content = f.read()

        lines = content.strip().split('\n')

        # Parse header
        capacity = None
        dimension = None

        coords = {}
        demands_dict = {}
        depot_id = None

        section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            if line.startswith('NAME'):
                name = line.split(':')[1].strip()
            elif line.startswith('CAPACITY'):
                capacity = float(line.split(':')[1].strip())
            elif line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())
            elif line == 'NODE_COORD_SECTION':
                section = 'coords'
            elif line == 'DEMAND_SECTION':
                section = 'demand'
            elif line == 'DEPOT_SECTION':
                section = 'depot'
            elif line == 'EOF':
                break
            elif section == 'coords':
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[node_id] = (x, y)
            elif section == 'demand':
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0])
                    demand = float(parts[1])
                    demands_dict[node_id] = demand
            elif section == 'depot':
                if line != '-1':
                    depot_id = int(line)

        if capacity is None:
            raise ValueError("Could not find CAPACITY")
        if depot_id is None:
            depot_id = 1  # Default to node 1 as depot

        # Build instance
        depot = coords[depot_id]
        customers = []
        demands = []

        for node_id in sorted(coords.keys()):
            if node_id != depot_id:
                customers.append(coords[node_id])
                demands.append(demands_dict.get(node_id, 0))

        return cls(
            depot=depot,
            customers=customers,
            demands=demands,
            vehicle_capacity=capacity,
            name=name,
        )

    def __repr__(self) -> str:
        return (
            f"CVRPInstance(name={self.name!r}, "
            f"customers={self.num_customers}, "
            f"capacity={self.vehicle_capacity}, "
            f"total_demand={self.total_demand})"
        )


@dataclass
class VRPTWInstance:
    """
    A Vehicle Routing Problem with Time Windows instance.

    Extends CVRP with time windows for each customer:
    - Each customer has an earliest and latest arrival time
    - Vehicles can wait if arriving early, but cannot arrive late
    - Each customer has a service time

    Attributes:
        depot: (x, y) coordinates of the depot
        customers: List of (x, y) coordinates for each customer
        demands: Demand at each customer
        time_windows: List of (earliest, latest) for each customer
        service_times: Service time at each customer
        vehicle_capacity: Maximum capacity of each vehicle
        depot_time_window: (earliest, latest) for depot (planning horizon)
        num_vehicles: Optional limit on number of vehicles
        speed: Vehicle speed for travel time calculation (default: 1.0)
        name: Optional instance name
    """
    depot: Tuple[float, float]
    customers: List[Tuple[float, float]]
    demands: List[float]
    time_windows: List[Tuple[float, float]]  # (earliest, latest) per customer
    service_times: List[float]
    vehicle_capacity: float
    depot_time_window: Tuple[float, float] = (0.0, float('inf'))
    num_vehicles: Optional[int] = None
    speed: float = 1.0
    name: Optional[str] = None

    def __post_init__(self):
        n = len(self.customers)
        if len(self.demands) != n:
            raise ValueError("demands must match customers length")
        if len(self.time_windows) != n:
            raise ValueError("time_windows must match customers length")
        if len(self.service_times) != n:
            raise ValueError("service_times must match customers length")
        if any(d < 0 for d in self.demands):
            raise ValueError("demands must be non-negative")
        if any(d > self.vehicle_capacity for d in self.demands):
            raise ValueError("no customer demand can exceed vehicle capacity")
        # Validate time windows
        for i, (e, l) in enumerate(self.time_windows):
            if e > l:
                raise ValueError(f"Invalid time window for customer {i}: [{e}, {l}]")

    @property
    def num_customers(self) -> int:
        """Number of customers (excluding depot)."""
        return len(self.customers)

    @property
    def total_demand(self) -> float:
        """Total demand across all customers."""
        return sum(self.demands)

    @property
    def min_vehicles(self) -> int:
        """Minimum number of vehicles needed (capacity lower bound)."""
        return math.ceil(self.total_demand / self.vehicle_capacity)

    def distance(self, i: int, j: int) -> float:
        """
        Compute Euclidean distance between two nodes.

        Node 0 is the depot, nodes 1..n are customers.
        """
        loc_i = self.depot if i == 0 else self.customers[i - 1]
        loc_j = self.depot if j == 0 else self.customers[j - 1]

        dx = loc_i[0] - loc_j[0]
        dy = loc_i[1] - loc_j[1]
        return math.sqrt(dx * dx + dy * dy)

    def travel_time(self, i: int, j: int) -> float:
        """
        Compute travel time between two nodes.

        Travel time = distance / speed
        """
        return self.distance(i, j) / self.speed

    def get_time_window(self, i: int) -> Tuple[float, float]:
        """
        Get time window for node i.

        Node 0 is depot, nodes 1..n are customers.
        """
        if i == 0:
            return self.depot_time_window
        return self.time_windows[i - 1]

    def get_service_time(self, i: int) -> float:
        """
        Get service time for node i.

        Node 0 is depot (service time 0), nodes 1..n are customers.
        """
        if i == 0:
            return 0.0
        return self.service_times[i - 1]

    @classmethod
    def from_solomon(cls, filepath: str) -> 'VRPTWInstance':
        """
        Load a VRPTW instance from Solomon format.

        Solomon format:
            Line 1: Instance name
            Lines 2-4: Empty or comments
            Line 5: VEHICLE section header
            Line 6: NUMBER CAPACITY
            Line 7: num_vehicles capacity
            Lines 8-9: Empty or CUSTOMER section header
            Line 10+: CUST_ID X Y DEMAND READY_TIME DUE_DATE SERVICE_TIME

        Args:
            filepath: Path to the Solomon format file

        Returns:
            VRPTWInstance
        """
        import os
        name = os.path.splitext(os.path.basename(filepath))[0]

        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # Find vehicle capacity line (after VEHICLE header)
        vehicle_idx = None
        for i, line in enumerate(lines):
            if 'VEHICLE' in line.upper():
                vehicle_idx = i
                break

        if vehicle_idx is None:
            raise ValueError("Could not find VEHICLE section")

        # Parse vehicle info (skip header line "NUMBER CAPACITY")
        vehicle_line = lines[vehicle_idx + 2].split()
        num_vehicles = int(vehicle_line[0])
        capacity = float(vehicle_line[1])

        # Find customer section
        customer_idx = None
        for i, line in enumerate(lines):
            if 'CUSTOMER' in line.upper() or 'CUST NO' in line.upper():
                customer_idx = i
                break

        if customer_idx is None:
            # Try to find data section by looking for numeric lines
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) >= 7 and parts[0].isdigit():
                    customer_idx = i - 1
                    break

        if customer_idx is None:
            raise ValueError("Could not find CUSTOMER section")

        # Skip header lines until we find data
        data_start = customer_idx + 1
        while data_start < len(lines):
            parts = lines[data_start].split()
            if len(parts) >= 7 and parts[0].replace('.', '').isdigit():
                break
            data_start += 1

        # Parse customer data
        depot = None
        depot_tw = None
        customers = []
        demands = []
        time_windows = []
        service_times = []

        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) < 7:
                continue

            try:
                cust_id = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3])
                ready_time = float(parts[4])
                due_date = float(parts[5])
                service_time = float(parts[6])

                if cust_id == 0:
                    depot = (x, y)
                    depot_tw = (ready_time, due_date)
                else:
                    customers.append((x, y))
                    demands.append(demand)
                    time_windows.append((ready_time, due_date))
                    service_times.append(service_time)
            except (ValueError, IndexError):
                continue

        if depot is None:
            raise ValueError("Could not find depot (customer 0)")

        return cls(
            depot=depot,
            customers=customers,
            demands=demands,
            time_windows=time_windows,
            service_times=service_times,
            vehicle_capacity=capacity,
            depot_time_window=depot_tw or (0.0, float('inf')),
            num_vehicles=num_vehicles,
            name=name,
        )

    def to_cvrp(self) -> CVRPInstance:
        """Convert to CVRP instance (ignore time windows)."""
        return CVRPInstance(
            depot=self.depot,
            customers=self.customers,
            demands=self.demands,
            vehicle_capacity=self.vehicle_capacity,
            num_vehicles=self.num_vehicles,
            name=self.name,
        )

    def __repr__(self) -> str:
        return (
            f"VRPTWInstance(name={self.name!r}, "
            f"customers={self.num_customers}, "
            f"capacity={self.vehicle_capacity}, "
            f"total_demand={self.total_demand})"
        )
