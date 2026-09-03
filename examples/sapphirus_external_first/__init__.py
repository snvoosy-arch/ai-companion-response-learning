"""CPU reference slice for the Sapphirus External-First contract."""

from .external_first import (
    ActorContext,
    ActorEnvelope,
    AuthoritySnapshot,
    ExternalFirstRuntime,
    FixtureReadOnlyExecutor,
    InMemoryDeliverySink,
    MemoryCandidate,
    ScriptedActorBackend,
    ToolCall,
)

__all__ = [
    "ActorContext",
    "ActorEnvelope",
    "AuthoritySnapshot",
    "ExternalFirstRuntime",
    "FixtureReadOnlyExecutor",
    "InMemoryDeliverySink",
    "MemoryCandidate",
    "ScriptedActorBackend",
    "ToolCall",
]
