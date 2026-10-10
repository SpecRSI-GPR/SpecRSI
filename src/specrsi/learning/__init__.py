"""SpecRSI learned roles. Importing the trusted engine does not require this package."""
from .heads import TinyModelConfig, TinyRoleModel, SurfaceHead, ContinuationValueHead, WholeBatchValueHead, RelationalPlanBranch
from .records import CheckerAudit, DecisionRecord, EditExample, PairwiseValueExample, SearchTrajectory, TransitionRecord, collect_independent_group, split_by_lineage
from .trainer import SpecRSITrainer, TrainingConfig

__all__ = ["TinyModelConfig", "TinyRoleModel", "SurfaceHead", "ContinuationValueHead", "WholeBatchValueHead",
           "CheckerAudit", "DecisionRecord", "EditExample", "PairwiseValueExample", "SearchTrajectory", "TransitionRecord", "RelationalPlanBranch",
           "collect_independent_group", "split_by_lineage", "SpecRSITrainer", "TrainingConfig"]
