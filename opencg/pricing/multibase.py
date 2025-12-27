"""
Multi-base pricing algorithm for crew pairing problems.

In crew pairing, each crew must start and end at the same base. When using a
single global pricing problem, the labeling algorithm may only explore paths
from one base before hitting time limits, leaving other bases unexplored.

This module provides MultiBasePricingAlgorithm which runs separate pricing
subproblems for each base, ensuring balanced exploration across all bases.

References:
----------
Kasirzadeh, A., Saddoune, M., & Soumis, F. (2017).
Airline crew scheduling: models, algorithms, and data sets.
"In the bidline problem for the same set of instances, a subproblem is
associated with each crew base (giving a total of 3 subproblems)"
"""

from typing import Dict, List, Optional, Set
import time

from opencg.core.arc import ArcType
from opencg.core.column import Column
from opencg.core.network import Network
from opencg.core.problem import Problem
from opencg.pricing.base import (
    PricingProblem,
    PricingConfig,
    PricingSolution,
    PricingStatus,
)
from opencg.pricing.labeling import LabelingAlgorithm
from opencg.pricing.label import Label


class MultiBasePricingAlgorithm(PricingProblem):
    """
    Multi-base pricing algorithm for crew pairing.

    This algorithm runs separate pricing subproblems for each base, then
    combines the results. This ensures that columns from all bases are
    generated, even when time limits are tight.

    The algorithm:
    1. Identifies all bases from SOURCE_ARC attributes
    2. For each base, runs a constrained pricing that only explores
       paths starting from that base
    3. Combines columns from all bases
    4. Returns the best columns overall

    Example:
        >>> pricing = MultiBasePricingAlgorithm(problem)
        >>> pricing.set_dual_values(duals)
        >>> result = pricing.solve()
        >>> print(f"Found {len(result.columns)} columns from all bases")
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None
    ):
        """
        Initialize the multi-base pricing algorithm.

        Args:
            problem: The Problem instance
            config: Optional configuration. Time/column limits are divided
                    among bases.
        """
        super().__init__(problem, config)

        # Identify bases from network
        self._bases = self._find_bases()
        self._base_source_arcs = self._find_base_source_arcs()

    def _find_bases(self) -> List[str]:
        """Find all base names from SOURCE_ARC attributes."""
        bases = set()
        for arc in self._problem.network.arcs:
            if arc.arc_type == ArcType.SOURCE_ARC:
                base = arc.get_attribute('base')
                if base:
                    bases.add(base)
        return sorted(bases)

    def _find_base_source_arcs(self) -> Dict[str, List[int]]:
        """Find source arc indices for each base."""
        base_arcs = {base: [] for base in self._bases}
        for arc in self._problem.network.arcs:
            if arc.arc_type == ArcType.SOURCE_ARC:
                base = arc.get_attribute('base')
                if base in base_arcs:
                    base_arcs[base].append(arc.index)
        return base_arcs

    def _solve_impl(self) -> PricingSolution:
        """
        Run pricing for each base and combine results.

        Returns:
            Combined PricingSolution from all bases
        """
        start_time = time.time()
        all_columns: List[Column] = []
        total_labels_created = 0
        total_labels_dominated = 0

        # Divide time/column limits among bases
        num_bases = len(self._bases)
        if num_bases == 0:
            # No bases found - fall back to regular pricing
            pricing = LabelingAlgorithm(self._problem, self._config)
            pricing.set_dual_values(self._dual_values)
            return pricing.solve()

        # Per-base config
        per_base_time = (
            self._config.max_time / num_bases
            if self._config.max_time > 0 else 0
        )
        per_base_columns = (
            max(1, self._config.max_columns // num_bases)
            if self._config.max_columns > 0 else 0
        )
        per_base_labels = (
            max(1, self._config.max_labels // num_bases)
            if self._config.max_labels > 0 else 0
        )

        per_base_config = PricingConfig(
            max_columns=per_base_columns,
            max_labels=per_base_labels,
            max_time=per_base_time,
            reduced_cost_threshold=self._config.reduced_cost_threshold,
            use_dominance=self._config.use_dominance,
            check_elementarity=self._config.check_elementarity,
        )

        # Run pricing for each base
        for base in self._bases:
            base_result = self._price_for_base(base, per_base_config)
            all_columns.extend(base_result.columns)
            total_labels_created += base_result.num_labels_created
            total_labels_dominated += base_result.num_labels_dominated

        # Sort by reduced cost and apply global limit
        all_columns.sort(key=lambda c: c.reduced_cost)
        if self._config.max_columns > 0:
            all_columns = all_columns[:self._config.max_columns]

        # Build combined solution
        solve_time = time.time() - start_time
        best_rc = min(c.reduced_cost for c in all_columns) if all_columns else None

        if all_columns:
            status = (
                PricingStatus.COLUMNS_FOUND
                if best_rc < self._config.reduced_cost_threshold
                else PricingStatus.NO_COLUMNS
            )
        else:
            status = PricingStatus.NO_COLUMNS

        return PricingSolution(
            status=status,
            columns=all_columns,
            best_reduced_cost=best_rc,
            num_labels_created=total_labels_created,
            num_labels_dominated=total_labels_dominated,
            solve_time=solve_time,
        )

    def _price_for_base(
        self,
        base_name: str,
        config: PricingConfig
    ) -> PricingSolution:
        """
        Run pricing for a single base.

        This creates a constrained pricing problem that only explores
        paths starting from the specified base.

        Args:
            base_name: Name of the base to price for
            config: Configuration for this base's pricing

        Returns:
            PricingSolution with columns from this base
        """
        # Create base-specific pricing algorithm
        pricing = BaseRestrictedLabelingAlgorithm(
            self._problem,
            config,
            allowed_base=base_name,
            base_source_arcs=set(self._base_source_arcs[base_name])
        )
        pricing.set_dual_values(self._dual_values)
        return pricing.solve()


class BaseRestrictedLabelingAlgorithm(LabelingAlgorithm):
    """
    Labeling algorithm restricted to paths starting from a specific base.

    This is a helper class used by MultiBasePricingAlgorithm to run
    per-base pricing. It only extends labels along SOURCE_ARCs that
    belong to the specified base.
    """

    def __init__(
        self,
        problem: Problem,
        config: Optional[PricingConfig] = None,
        allowed_base: str = "",
        base_source_arcs: Optional[Set[int]] = None
    ):
        """
        Initialize base-restricted labeling.

        Args:
            problem: The Problem instance
            config: Configuration
            allowed_base: Name of the base to restrict to
            base_source_arcs: Set of SOURCE_ARC indices for this base
        """
        super().__init__(problem, config)
        self._allowed_base = allowed_base
        self._base_source_arcs = base_source_arcs or set()

    def _extend_label(self, label: Label, arc) -> Optional[Label]:
        """
        Extend a label along an arc, with base restriction.

        For SOURCE_ARCs, only allows extension if the arc belongs to
        the allowed base.
        """
        # Restrict SOURCE_ARCs to the allowed base
        if arc.arc_type == ArcType.SOURCE_ARC:
            if arc.index not in self._base_source_arcs:
                return None  # Skip arcs from other bases

        # Otherwise, use standard extension
        return super()._extend_label(label, arc)
