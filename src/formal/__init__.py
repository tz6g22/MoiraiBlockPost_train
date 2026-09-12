"""Formal task-adaptive Block AttnRes utilities.

The package is deliberately small: it contains the conversion checks and
optimizer policy shared by the formal pipeline.  Discovery remains separate
and must provide a residual-only cost before this runtime is instantiated.
"""

from src.formal.runtime import (
    alpha_parameter_names,
    attnres_parameter_names,
    build_joint_optimizer,
    identity_test,
    task_routing_parameter_names,
    trainability_audit,
)
from src.formal.task_banks import TaskBank

__all__ = [
    "alpha_parameter_names",
    "attnres_parameter_names",
    "build_joint_optimizer",
    "identity_test",
    "task_routing_parameter_names",
    "trainability_audit",
    "TaskBank",
]
