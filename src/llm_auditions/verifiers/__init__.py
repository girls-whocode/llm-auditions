"""Verifier package — exports all verifier classes."""

from .base import BaseVerifier, VerifierResult
from .command_safety import CommandSafetyVerifier
from .contradiction import ContradictionVerifier
from .development import DevelopmentVerifier
from .evidence import EvidenceVerifier
from .fact_preservation import FactPreservationVerifier
from .mathematics import MathematicsVerifier
from .structure import StructureVerifier

# Registry: verifier name → class
REGISTRY: dict[str, type[BaseVerifier]] = {
    "structure": StructureVerifier,
    "mathematics": MathematicsVerifier,
    "development": DevelopmentVerifier,
    "command_safety": CommandSafetyVerifier,
    "evidence": EvidenceVerifier,
    "fact_preservation": FactPreservationVerifier,
    "contradiction": ContradictionVerifier,
}

__all__ = [
    "BaseVerifier",
    "VerifierResult",
    "StructureVerifier",
    "MathematicsVerifier",
    "DevelopmentVerifier",
    "CommandSafetyVerifier",
    "EvidenceVerifier",
    "FactPreservationVerifier",
    "ContradictionVerifier",
    "REGISTRY",
]
