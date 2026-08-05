"""Accepted Stage-1 controller baselines over immutable profile target sets.

The module exposes query-only direct scoring, an additive independent baseline,
a causal interaction-aware planner, static Pareto selection, and paired
bootstrap analysis. It does not load or quantize a model or establish evidence
by itself.
"""

from ._controller_common import (
    Bit,
    PrefixContextProvider,
    Profile,
    QueryFeatures,
    StructuralNormalizer,
    TrainingQuery,
    enumerate_profiles,
    profile_from_id,
    profile_id,
)
from ._controller_interaction import (
    BitDecision,
    CausalInteractionPlanner,
    InteractionPlanResult,
)
from ._controller_query import (
    DirectProfileScorer,
    IndependentGroupScorer,
    select_pareto_static_profile,
)
from ._controller_stats import (
    PairedBootstrapResult,
    paired_bootstrap_difference,
)
from .controller_artifacts import (
    Stage1ControllerArtifactAdapter,
    Stage1ProfileArtifact,
)

__all__ = [
    "Bit",
    "BitDecision",
    "CausalInteractionPlanner",
    "DirectProfileScorer",
    "IndependentGroupScorer",
    "InteractionPlanResult",
    "PairedBootstrapResult",
    "PrefixContextProvider",
    "Profile",
    "QueryFeatures",
    "Stage1ControllerArtifactAdapter",
    "Stage1ProfileArtifact",
    "StructuralNormalizer",
    "TrainingQuery",
    "enumerate_profiles",
    "paired_bootstrap_difference",
    "profile_from_id",
    "profile_id",
    "select_pareto_static_profile",
]
