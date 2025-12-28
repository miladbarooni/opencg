"""
Kasirzadeh Parser - reads Kasirzadeh crew scheduling benchmark instances.

The Kasirzadeh format is from GERAD (2014) and contains:
- listOfBases.csv: Airports and crew bases
- day_X.csv: Flight schedules for each day
- solution_0: Example crew schedules (optional)
- Various constraint files (credit, availability, etc.)

Data Structure:
--------------
listOfBases.csv:
    airport, status, nbEmployees
    BASE1, 1, 7
    AIR1, 0, 0
    (status 1 = base, 0 = regular airport)

day_X.csv:
    #leg_nb, airport_dep, date_dep, hour_dep, airport_arr, date_arr, hour_arr
    LEG_01_0, BASE1, 2000-01-01, 12:00, AIR1, 2000-01-01, 13:13

solution_0:
    schedule N EMPXXX (BASE): TASK--->TASK--->TASK;
    (TASK can be LEG_XX_Y, VACATION, TDH_*, PAL_*, POST_PAIRING, etc.)

Reference:
---------
Kasirzadeh, A., Saddoune, M., & Soumis, F. (2017).
Airline crew scheduling: models, algorithms, and data sets.
EURO Journal on Transportation and Logistics, 6(2), 111-137.
"""

import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from opencg.core.arc import ArcType
from opencg.core.network import Network
from opencg.core.node import NodeType
from opencg.core.problem import CoverType, Problem, ProblemBuilder
from opencg.core.resource import AccumulatingResource
from opencg.parsers.base import Parser, ParserConfig


@dataclass
class FlightLeg:
    """Parsed flight leg data."""
    leg_id: str
    day: int  # Day number (1-31)
    leg_number: int  # Leg number within day
    dep_airport: str
    dep_date: str
    dep_time: str
    arr_airport: str
    arr_date: str
    arr_time: str

    @property
    def dep_datetime(self) -> datetime:
        """Parse departure datetime."""
        return datetime.strptime(
            f"{self.dep_date} {self.dep_time}",
            "%Y-%m-%d %H:%M"
        )

    @property
    def arr_datetime(self) -> datetime:
        """Parse arrival datetime."""
        return datetime.strptime(
            f"{self.arr_date} {self.arr_time}",
            "%Y-%m-%d %H:%M"
        )

    @property
    def duration_hours(self) -> float:
        """Flight duration in hours."""
        delta = self.arr_datetime - self.dep_datetime
        return delta.total_seconds() / 3600.0


@dataclass
class BaseInfo:
    """Parsed base/airport information."""
    name: str
    is_base: bool
    num_employees: int


class KasirzadehParser(Parser):
    """
    Parser for Kasirzadeh crew scheduling instances.

    Reads the GERAD benchmark instances and constructs a Problem object.

    The parser creates:
    - Network with airports as nodes and flights as arcs
    - Resources for duty time, flight time
    - Cover constraints for each flight leg

    Example:
        >>> parser = KasirzadehParser()
        >>> problem = parser.parse("/path/to/instance1")
        >>> print(problem.summary())

    Configuration Options:
        - min_connection_time: Minimum connection time in hours (default 0.5)
        - max_connection_time: Maximum connection time in hours (default 4.0)
        - max_duty_time: Maximum duty time in hours (default 10.0)
        - include_deadheads: Whether to include deadhead arcs (default False)
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        """Initialize parser."""
        super().__init__(config)

        # Default options
        self._min_connection = self.config.options.get('min_connection_time', 0.5)
        self._max_connection = self.config.options.get('max_connection_time', 4.0)
        self._max_duty = self.config.options.get('max_duty_time', 10.0)
        self._max_flight = self.config.options.get('max_flight_time', 8.0)  # Max flight time per duty
        self._include_deadheads = self.config.options.get('include_deadheads', False)

        # Overnight layover options
        # If connection time exceeds max_connection, it's an overnight layover
        # that resets duty time (crew rests at hotel)
        # NOTE: min_layover should be close to max_connection to avoid "dead zone"
        # where connections are neither valid connections nor valid layovers
        self._min_layover = self.config.options.get('min_layover_time', self._max_connection)  # Default to close the gap
        self._max_layover = self.config.options.get('max_layover_time', 24.0)  # Max overnight

        # Pairing limits
        self._max_days = self.config.options.get('max_pairing_days', 4)  # Max duty days in pairing

        # Validate: check for connection time gap
        if self._min_layover > self._max_connection:
            gap = self._min_layover - self._max_connection
            warnings.warn(
                f"Connection time gap detected: connections between "
                f"{self._max_connection}h and {self._min_layover}h ({gap}h gap) "
                f"will have no arc, potentially making flights unreachable. "
                f"Set min_layover_time={self._max_connection} to close the gap.",
                UserWarning
            )

    def can_parse(self, path: Union[str, Path]) -> bool:
        """
        Check if path contains Kasirzadeh-format files.

        Args:
            path: Path to instance directory

        Returns:
            True if listOfBases.csv and day files exist
        """
        path = Path(path)
        if not path.is_dir():
            return False

        # Check for required files
        bases_file = path / "listOfBases.csv"
        day1_file = path / "day_1.csv"

        return bases_file.exists() and day1_file.exists()

    def parse(self, path: Union[str, Path]) -> Problem:
        """
        Parse a Kasirzadeh instance.

        Args:
            path: Path to instance directory

        Returns:
            Constructed Problem

        Raises:
            FileNotFoundError: If required files not found
            ValueError: If files are malformed
        """
        path = Path(path)
        self._log(f"Parsing instance from: {path}")

        # 1. Parse bases and airports
        bases = self._parse_bases(path / "listOfBases.csv")
        self._log(f"Found {len(bases)} locations, {sum(1 for b in bases if b.is_base)} bases")

        # 2. Parse all day files
        flights = self._parse_all_days(path)
        self._log(f"Found {len(flights)} flight legs")

        # 3. Build the network
        network = self._build_network(bases, flights)
        self._log(f"Built network: {network.num_nodes} nodes, {network.num_arcs} arcs")

        # 4. Define resources
        resources = self._create_resources()

        # 5. Create cover constraints (one per flight)
        # 6. Build and return Problem
        problem = self._build_problem(path.name, network, resources, flights, bases)

        if self.config.validate:
            errors = problem.validate()
            if errors:
                for error in errors:
                    self._log(f"Validation error: {error}")

        return problem

    def _parse_bases(self, filepath: Path) -> list[BaseInfo]:
        """
        Parse listOfBases.csv.

        Format:
            airport, status, nbEmployees
            BASE1, 1, 7
        """
        bases = []

        with open(filepath, encoding=self.config.encoding) as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip header
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                name = parts[0]
                is_base = parts[1].strip() == '1'
                num_employees = int(parts[2].strip())
                bases.append(BaseInfo(name, is_base, num_employees))

        return bases

    def _parse_day_file(self, filepath: Path, day: int) -> list[FlightLeg]:
        """
        Parse a single day_X.csv file.

        Format:
            #leg_nb, airport_dep, date_dep, hour_dep, airport_arr, date_arr, hour_arr
            LEG_01_0, BASE1, 2000-01-01, 12:00, AIR1, 2000-01-01, 13:13
        """
        flights = []

        with open(filepath, encoding=self.config.encoding) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 7:
                leg_id = parts[0]

                # Extract day and leg number from ID (e.g., LEG_01_0)
                match = re.match(r'LEG_(\d+)_(\d+)', leg_id)
                if match:
                    leg_day = int(match.group(1))
                    leg_num = int(match.group(2))
                else:
                    leg_day = day
                    leg_num = 0

                flight = FlightLeg(
                    leg_id=leg_id,
                    day=leg_day,
                    leg_number=leg_num,
                    dep_airport=parts[1],
                    dep_date=parts[2],
                    dep_time=parts[3],
                    arr_airport=parts[4],
                    arr_date=parts[5],
                    arr_time=parts[6]
                )
                flights.append(flight)

        return flights

    def _parse_all_days(self, path: Path) -> list[FlightLeg]:
        """Parse all day_X.csv files in the instance directory."""
        all_flights = []
        day = 1

        while True:
            day_file = path / f"day_{day}.csv"
            if not day_file.exists():
                break

            flights = self._parse_day_file(day_file, day)
            all_flights.extend(flights)
            day += 1

        return all_flights

    def _build_network(
        self,
        bases: list[BaseInfo],
        flights: list[FlightLeg]
    ) -> Network:
        """
        Build the time-space network from parsed data.

        Network structure:
        - Global source node (index 0)
        - Global sink node (index 1)
        - For each flight: departure node and arrival node
        - Arcs: flights, connections, source-to-base-departures, base-arrivals-to-sink

        The constraint that crew must return to their home base is enforced by
        the HomeBaseResource, which tracks which base a pairing started from
        and only allows ending at the same base.
        """
        network = Network()

        # Add global source and sink
        source_idx = network.add_source()
        sink_idx = network.add_sink()

        # Create location lookup
        base_names = {b.name for b in bases if b.is_base}

        # Create nodes for each flight event
        # We'll use a time-space network: (airport, time) pairs
        flight_nodes: dict[str, tuple[int, int]] = {}  # leg_id -> (dep_node, arr_node)

        for flight in flights:
            # Departure node
            dep_name = f"DEP_{flight.leg_id}"
            dep_idx = network.add_node(
                name=dep_name,
                node_type=NodeType.FLIGHT_DEP,
                location=flight.dep_airport,
                time=self._datetime_to_hours(flight.dep_datetime),
                flight_id=flight.leg_id
            )

            # Arrival node
            arr_name = f"ARR_{flight.leg_id}"
            arr_idx = network.add_node(
                name=arr_name,
                node_type=NodeType.FLIGHT_ARR,
                location=flight.arr_airport,
                time=self._datetime_to_hours(flight.arr_datetime),
                flight_id=flight.leg_id
            )

            flight_nodes[flight.leg_id] = (dep_idx, arr_idx)

            # Add flight arc
            duration = flight.duration_hours
            network.add_arc(
                source=dep_idx,
                target=arr_idx,
                cost=1.0,  # Unit cost per flight (can be customized)
                resource_consumption={
                    "duty_time": duration,
                    "flight_time": duration,
                },
                arc_type=ArcType.FLIGHT,
                flight_id=flight.leg_id,
                flight_number=flight.leg_id,
                duration=duration
            )

        # Add connection arcs between compatible flights
        # (arrival at airport A, followed by departure at airport A)
        self._add_connection_arcs(network, flights, flight_nodes)

        # Add source and sink arcs connecting bases to global source/sink
        self._add_source_sink_arcs(network, flights, flight_nodes, base_names, source_idx, sink_idx)

        return network

    def _add_source_sink_arcs(
        self,
        network: Network,
        flights: list[FlightLeg],
        flight_nodes: dict[str, tuple[int, int]],
        base_names: set,
        source_idx: int,
        sink_idx: int
    ) -> None:
        """Add source and sink arcs connecting bases to global source/sink.

        Creates:
        - SOURCE_ARC from global source to departure nodes of flights at bases
        - SINK_ARC from arrival nodes at bases to global sink

        The 'base' attribute on these arcs is used by FastPerSourcePricing
        to ensure crews return to their home base.
        """
        for flight in flights:
            dep_node, arr_node = flight_nodes[flight.leg_id]

            # If flight departs from a base, add source arc
            if flight.dep_airport in base_names:
                network.add_arc(
                    source=source_idx,
                    target=dep_node,
                    cost=0.0,
                    resource_consumption={
                        "duty_time": 0.0,
                        "flight_time": 0.0,
                        "duty_days": 0.0,
                    },
                    arc_type=ArcType.SOURCE_ARC,
                    base=flight.dep_airport
                )

            # If flight arrives at a base, add sink arc
            if flight.arr_airport in base_names:
                network.add_arc(
                    source=arr_node,
                    target=sink_idx,
                    cost=0.0,
                    resource_consumption={
                        "duty_time": 0.0,
                        "flight_time": 0.0,
                        "duty_days": 0.0,
                    },
                    arc_type=ArcType.SINK_ARC,
                    base=flight.arr_airport
                )

    def _add_connection_arcs(
        self,
        network: Network,
        flights: list[FlightLeg],
        flight_nodes: dict[str, tuple[int, int]]
    ) -> None:
        """Add connection arcs between compatible flights.

        This method creates two types of arcs:
        1. CONNECTION arcs: Short connections (e.g., 0.5-4 hours) that count as duty time
        2. OVERNIGHT arcs: Layovers (e.g., 8-24 hours) where crew rests and duty resets
        """
        # Sort flights by arrival time at each airport
        arrivals_by_airport: dict[str, list[tuple[float, str]]] = {}
        departures_by_airport: dict[str, list[tuple[float, str]]] = {}

        for flight in flights:
            arr_time = self._datetime_to_hours(flight.arr_datetime)
            dep_time = self._datetime_to_hours(flight.dep_datetime)

            if flight.arr_airport not in arrivals_by_airport:
                arrivals_by_airport[flight.arr_airport] = []
            arrivals_by_airport[flight.arr_airport].append((arr_time, flight.leg_id))

            if flight.dep_airport not in departures_by_airport:
                departures_by_airport[flight.dep_airport] = []
            departures_by_airport[flight.dep_airport].append((dep_time, flight.leg_id))

        # For each airport, connect arrivals to feasible departures
        for airport in arrivals_by_airport:
            if airport not in departures_by_airport:
                continue

            arrivals = sorted(arrivals_by_airport[airport])
            departures = sorted(departures_by_airport[airport])

            for arr_time, arr_leg_id in arrivals:
                for dep_time, dep_leg_id in departures:
                    # Check connection time constraints
                    connection_time = dep_time - arr_time

                    if self._min_connection <= connection_time <= self._max_connection:
                        # Short connection - counts as duty time
                        arr_node = flight_nodes[arr_leg_id][1]  # arrival node
                        dep_node = flight_nodes[dep_leg_id][0]  # departure node

                        network.add_arc(
                            source=arr_node,
                            target=dep_node,
                            cost=0.0,  # No cost for connections
                            resource_consumption={
                                "duty_time": connection_time,
                                "connection_time": connection_time,
                            },
                            arc_type=ArcType.CONNECTION,
                            duration=connection_time
                        )
                    elif self._min_layover <= connection_time <= self._max_layover:
                        # Overnight layover - resets duty time (crew rests)
                        arr_node = flight_nodes[arr_leg_id][1]  # arrival node
                        dep_node = flight_nodes[dep_leg_id][0]  # departure node

                        # Overnight arcs reset duty_time by consuming a large negative
                        # value (-max_duty). This brings the accumulated duty back to ~0.
                        # The AccumulatingResource will clamp this at 0 (initial value).
                        network.add_arc(
                            source=arr_node,
                            target=dep_node,
                            cost=0.0,  # No additional cost for overnight
                            resource_consumption={
                                "duty_time": -self._max_duty,  # Reset duty after rest
                                "flight_time": -self._max_flight,  # Also reset flight time
                                "duty_days": 1.0,  # Increment day counter
                                "rest_time": connection_time,  # Track rest duration
                            },
                            arc_type=ArcType.OVERNIGHT,
                            duration=connection_time,
                            is_layover=True
                        )

    def _add_base_flight_arcs(
        self,
        network: Network,
        flights: list[FlightLeg],
        flight_nodes: dict[str, tuple[int, int]],
        base_source_nodes: dict[str, int],
        base_sink_nodes: dict[str, int]
    ) -> None:
        """Add arcs connecting base intermediate nodes to flights.

        This creates:
        - Arcs from each base's intermediate source to flights departing from that base
        - Arcs from flights arriving at a base to that base's intermediate sink

        The key insight is that a pairing starting at BASE1 (via BASE1's source node)
        can only end at BASE1 (via BASE1's sink node) because the only way to reach
        the global sink from within the flight network is through a base sink node,
        and the network structure ensures crews returning to a base connect to that
        specific base's sink node.
        """
        for flight in flights:
            dep_node, arr_node = flight_nodes[flight.leg_id]

            # If departing from a base, connect from that base's intermediate source
            if flight.dep_airport in base_source_nodes:
                base_src = base_source_nodes[flight.dep_airport]
                network.add_arc(
                    source=base_src,
                    target=dep_node,
                    cost=0.0,
                    resource_consumption={},
                    arc_type=ArcType.SOURCE_ARC,
                    base=flight.dep_airport
                )

            # If arriving at a base, connect to that base's intermediate sink
            if flight.arr_airport in base_sink_nodes:
                base_sink = base_sink_nodes[flight.arr_airport]
                network.add_arc(
                    source=arr_node,
                    target=base_sink,
                    cost=0.0,
                    resource_consumption={},
                    arc_type=ArcType.SINK_ARC,
                    base=flight.arr_airport
                )

    def _create_resources(self) -> list:
        """Create resource constraints."""
        return [
            AccumulatingResource(
                name="duty_time",
                initial=0.0,
                max_value=self._max_duty
            ),
            AccumulatingResource(
                name="flight_time",
                initial=0.0,
                max_value=self._max_flight
            ),
            AccumulatingResource(
                name="duty_days",
                initial=1.0,  # Start on day 1
                max_value=float(self._max_days)  # Limit pairing length
            ),
        ]

    def _build_problem(
        self,
        name: str,
        network: Network,
        resources: list,
        flights: list[FlightLeg],
        bases: list[BaseInfo]
    ) -> Problem:
        """Build the final Problem object."""
        # Get flight arcs (these need to be covered)
        flight_arcs = list(network.arcs_of_type(ArcType.FLIGHT))

        # Build problem
        builder = ProblemBuilder(name)
        builder.with_network(network)
        builder.with_resources(resources)
        builder.with_cover_type(CoverType.SET_PARTITIONING)
        builder.minimize()

        # Add cover constraints for each flight
        for arc in flight_arcs:
            flight_id = arc.get_attribute('flight_id', f"flight_{arc.index}")
            builder.add_cover_constraint(
                item_id=arc.index,
                name=flight_id
            )

        # Add metadata
        builder.with_metadata(
            source="Kasirzadeh",
            num_flights=len(flights),
            num_bases=sum(1 for b in bases if b.is_base),
            num_airports=len(bases),
            days=max(f.day for f in flights) if flights else 0
        )

        return builder.build_unchecked()  # Don't validate here, we do it in parse()

    def _datetime_to_hours(self, dt: datetime) -> float:
        """
        Convert datetime to hours since epoch.

        We use a simple conversion for this benchmark:
        hours = day * 24 + hour + minute/60
        """
        # Use Jan 1, 2000 as epoch (matches Kasirzadeh data)
        epoch = datetime(2000, 1, 1, 0, 0)
        delta = dt - epoch
        return delta.total_seconds() / 3600.0

    def get_format_name(self) -> str:
        return "Kasirzadeh"
