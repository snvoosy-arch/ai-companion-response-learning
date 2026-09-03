"""A dependency-free vertical slice of Sapphirus's External-First runtime.

This is a public reference implementation, not the private Discord runtime. It keeps
the important contract boundaries while replacing the model, network, database, and
Discord delivery with deterministic in-memory components.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Mapping, Protocol, Sequence


ACTOR_ACTIONS = frozenset({"reply", "silence", "use_tool"})
MEMORY_KINDS = frozenset({"profile", "ongoing", "open_loop", "episodic", "other"})
TOOL_OUTCOME_STATUSES = frozenset({"resolved", "unresolved", "failed"})
MAX_TOOL_CALLS = 1
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,63}\Z")


class ContractError(ValueError):
    """Raised when an Actor or runtime object violates the public contract."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    query: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not isinstance(self.query, str):
            raise ContractError("tool name and query must be strings")
        if not self.name.strip() or not self.query.strip():
            raise ContractError("tool name and query are required")
        if not _SAFE_IDENTIFIER.fullmatch(self.name):
            raise ContractError("tool name must be a safe identifier")


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    kind: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not isinstance(self.text, str):
            raise ContractError("memory kind and text must be strings")
        if self.kind not in MEMORY_KINDS:
            raise ContractError(f"unsupported memory kind: {self.kind}")
        if not self.text.strip():
            raise ContractError("memory candidate text is required")


@dataclass(frozen=True, slots=True)
class ActorEnvelope:
    action: str
    speech: str
    tool_calls: tuple[ToolCall, ...] = ()
    memory_candidates: tuple[MemoryCandidate, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or not isinstance(self.speech, str):
            raise ContractError("action and speech must be strings")
        if not isinstance(self.tool_calls, tuple) or not all(
            isinstance(item, ToolCall) for item in self.tool_calls
        ):
            raise ContractError("tool_calls must be a tuple of ToolCall objects")
        if not isinstance(self.memory_candidates, tuple) or not all(
            isinstance(item, MemoryCandidate) for item in self.memory_candidates
        ):
            raise ContractError(
                "memory_candidates must be a tuple of MemoryCandidate objects"
            )
        if self.action not in ACTOR_ACTIONS:
            raise ContractError(f"unsupported actor action: {self.action}")
        if self.action == "reply":
            if not self.speech.strip() or self.tool_calls:
                raise ContractError("reply requires speech and forbids tool calls")
        elif self.action == "silence":
            if self.speech or self.tool_calls:
                raise ContractError("silence requires empty speech and tool calls")
        elif self.action == "use_tool":
            if self.speech or len(self.tool_calls) != 1:
                raise ContractError(
                    "use_tool requires empty speech and exactly one call"
                )
        if len(self.memory_candidates) > 3:
            raise ContractError("at most three memory candidates are allowed")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ActorEnvelope":
        if not isinstance(payload, Mapping):
            raise ContractError("actor envelope must be an object")
        expected = {"action", "speech", "tool_calls", "memory_candidates"}
        if set(payload) != expected:
            raise ContractError("actor envelope must contain exactly four fields")

        raw_tools = payload["tool_calls"]
        raw_memories = payload["memory_candidates"]
        if not isinstance(raw_tools, list) or not isinstance(raw_memories, list):
            raise ContractError("tool_calls and memory_candidates must be arrays")

        tools: list[ToolCall] = []
        for item in raw_tools:
            if not isinstance(item, Mapping) or set(item) != {"name", "arguments"}:
                raise ContractError("tool call shape is invalid")
            arguments = item["arguments"]
            if not isinstance(arguments, Mapping) or set(arguments) != {"query"}:
                raise ContractError("tool arguments must contain only query")
            name = item["name"]
            query = arguments["query"]
            if not isinstance(name, str) or not isinstance(query, str):
                raise ContractError("tool name and query must be strings")
            tools.append(ToolCall(name, query))

        memories: list[MemoryCandidate] = []
        for item in raw_memories:
            if not isinstance(item, Mapping) or set(item) != {"kind", "text"}:
                raise ContractError("memory candidate shape is invalid")
            kind = item["kind"]
            text = item["text"]
            if not isinstance(kind, str) or not isinstance(text, str):
                raise ContractError("memory kind and text must be strings")
            memories.append(MemoryCandidate(kind, text))

        action = payload["action"]
        speech = payload["speech"]
        if not isinstance(action, str) or not isinstance(speech, str):
            raise ContractError("action and speech must be strings")

        return cls(
            action=action,
            speech=speech,
            tool_calls=tuple(tools),
            memory_candidates=tuple(memories),
        )


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    tool_name: str
    status: str
    evidence_id: str
    summary: str

    def __post_init__(self) -> None:
        values = (self.tool_name, self.status, self.evidence_id, self.summary)
        if not all(isinstance(value, str) for value in values):
            raise ContractError("tool outcome fields must be strings")
        if not self.tool_name.strip() or not self.evidence_id.strip():
            raise ContractError("tool outcome identity and evidence_id are required")
        identifiers = (self.tool_name, self.evidence_id)
        if not all(_SAFE_IDENTIFIER.fullmatch(value) for value in identifiers):
            raise ContractError("tool outcome identifiers are invalid")
        if self.status not in TOOL_OUTCOME_STATUSES:
            raise ContractError(f"unsupported tool outcome status: {self.status}")

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"


@dataclass(frozen=True, slots=True)
class ActorContext:
    user_text: str
    available_tools: tuple[str, ...]
    tool_outcome: ToolOutcome | None = None


class ActorBackend(Protocol):
    def generate(self, context: ActorContext) -> ActorEnvelope: ...


class ScriptedActorBackend:
    """Deterministic Actor used to exercise runtime contracts without an LLM."""

    def __init__(self, outputs: Sequence[ActorEnvelope]) -> None:
        self._outputs = deque(outputs)
        self.contexts: list[ActorContext] = []

    def generate(self, context: ActorContext) -> ActorEnvelope:
        self.contexts.append(context)
        if not self._outputs:
            raise RuntimeError("scripted Actor output queue is exhausted")
        return self._outputs.popleft()


@dataclass(frozen=True, slots=True)
class AuthoritySnapshot:
    actor_call_allowed: bool = True
    tool_use_allowed: bool = True
    delivery_allowed: bool = True
    persistent_memory_allowed: bool = False
    available_tools: tuple[str, ...] = ()
    allowed_memory_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.available_tools and not self.tool_use_allowed:
            raise ContractError("tools cannot be exposed without tool permission")
        if not self.persistent_memory_allowed and self.allowed_memory_kinds:
            raise ContractError("disabled persistence cannot allow memory kinds")
        if set(self.allowed_memory_kinds) - MEMORY_KINDS:
            raise ContractError("authority contains an unsupported memory kind")


class ReadOnlyExecutor(Protocol):
    def execute(self, call: ToolCall) -> ToolOutcome: ...


class FixtureReadOnlyExecutor:
    """Returns predefined evidence and never reaches a network or mutable resource."""

    def __init__(self, fixtures: Mapping[str, Mapping[str, str]]) -> None:
        self._fixtures = dict(fixtures)
        self.calls: list[tuple[str, str]] = []

    def execute(self, call: ToolCall) -> ToolOutcome:
        self.calls.append((call.name, _sha256(call.query)))
        fixture = self._fixtures.get(call.name)
        if fixture is None:
            return ToolOutcome(
                tool_name=call.name,
                status="unresolved",
                evidence_id=f"tool-{len(self.calls)}",
                summary="configured fixture is unavailable",
            )
        return ToolOutcome(
            tool_name=call.name,
            status=fixture.get("status", "resolved"),
            evidence_id=f"tool-{len(self.calls)}",
            summary=fixture.get("summary", ""),
        )


class DeliverySink(Protocol):
    def deliver(self, speech: str) -> bool: ...


class InMemoryDeliverySink:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def deliver(self, speech: str) -> bool:
        self.messages.append(speech)
        return True


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    operation: str
    status: str
    evidence_id: str
    target_sha256: str = ""


@dataclass(slots=True)
class TurnTrace:
    actor_calls: int = 0
    tool_calls: int = 0
    delivery_attempted: bool = False
    delivered: bool = False
    final_speech_sha256: str = ""
    authorized_memory_candidates: int = 0
    rejected_memory_candidates: int = 0
    issues: list[str] = field(default_factory=list)
    ledger: list[LedgerEntry] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        return {
            "actor_calls": self.actor_calls,
            "tool_calls": self.tool_calls,
            "delivery_attempted": self.delivery_attempted,
            "delivered": self.delivered,
            "final_speech_sha256": self.final_speech_sha256,
            "authorized_memory_candidates": self.authorized_memory_candidates,
            "rejected_memory_candidates": self.rejected_memory_candidates,
            "issues": list(self.issues),
            "ledger": [
                {
                    "operation": item.operation,
                    "status": item.status,
                    "evidence_id": item.evidence_id,
                    "target_sha256": item.target_sha256,
                }
                for item in self.ledger
            ],
        }


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    outcome: str
    speech: str
    trace: TurnTrace


class ExternalFirstRuntime:
    """Runs one bounded Actor turn and fails closed at external boundaries."""

    def __init__(
        self,
        *,
        actor: ActorBackend,
        executor: ReadOnlyExecutor,
        delivery: DeliverySink,
    ) -> None:
        self._actor = actor
        self._executor = executor
        self._delivery = delivery

    def run(self, user_text: str, authority: AuthoritySnapshot) -> RuntimeResult:
        trace = TurnTrace()
        if not authority.actor_call_allowed:
            trace.issues.append("ACTOR_CALL_NOT_ALLOWED")
            return RuntimeResult("blocked", "", trace)

        context = ActorContext(user_text, authority.available_tools)
        turn = self._call_actor(context, trace)
        if turn is None:
            return RuntimeResult("blocked", "", trace)
        self._gate_memories(turn, authority, trace)
        tool_outcome: ToolOutcome | None = None

        if turn.action == "use_tool":
            call = turn.tool_calls[0]
            if not authority.tool_use_allowed:
                trace.issues.append("TOOL_PERMISSION_DENIED")
                return RuntimeResult("blocked", "", trace)
            if call.name not in authority.available_tools:
                trace.issues.append("TOOL_NOT_AVAILABLE")
                return RuntimeResult("blocked", "", trace)
            if _contains_sensitive_query(call.query):
                trace.issues.append("SENSITIVE_TOOL_QUERY")
                return RuntimeResult("blocked", "", trace)

            outcome = self._execute_tool(call, trace)
            if outcome is None:
                return RuntimeResult("blocked", "", trace)
            tool_outcome = outcome
            post_tool = ActorContext(user_text, (), tool_outcome=outcome)
            turn = self._call_actor(post_tool, trace)
            if turn is None:
                return RuntimeResult("blocked", "", trace)
            self._gate_memories(turn, authority, trace)
            if turn.action == "use_tool":
                trace.issues.append("TOOL_BUDGET_EXHAUSTED")
                return RuntimeResult("blocked", "", trace)

        if turn.action == "silence":
            if tool_outcome is not None:
                trace.issues.append("TOOL_RESULT_DISCLOSURE_REQUIRED")
                return RuntimeResult("blocked", "", trace)
            return RuntimeResult("silence", "", trace)

        if tool_outcome is not None and not tool_outcome.resolved:
            if not _acknowledges_unresolved_tool(turn.speech):
                trace.issues.append("UNSUPPORTED_UNRESOLVED_TOOL_CLAIM")
                return RuntimeResult("blocked", "", trace)

        unsupported = _unsupported_execution_claim(turn.speech, trace.ledger)
        if unsupported:
            trace.issues.append(f"UNSUPPORTED_{unsupported.upper()}_CLAIM")
            return RuntimeResult("blocked", "", trace)
        if not authority.delivery_allowed:
            trace.issues.append("DELIVERY_NOT_AUTHORIZED")
            return RuntimeResult("blocked", "", trace)

        trace.delivery_attempted = True
        trace.final_speech_sha256 = _sha256(turn.speech)
        try:
            delivered = self._delivery.deliver(turn.speech)
        except Exception:
            trace.issues.append("DELIVERY_ERROR")
            trace.ledger.append(
                LedgerEntry(
                    operation="discord_delivery",
                    status="unknown",
                    evidence_id="delivery-1",
                    target_sha256=trace.final_speech_sha256,
                )
            )
            return RuntimeResult("delivery_unknown", "", trace)
        if not isinstance(delivered, bool):
            trace.issues.append("DELIVERY_INVALID_RESULT")
            trace.ledger.append(
                LedgerEntry(
                    operation="discord_delivery",
                    status="unknown",
                    evidence_id="delivery-1",
                    target_sha256=trace.final_speech_sha256,
                )
            )
            return RuntimeResult("delivery_unknown", "", trace)
        trace.delivered = delivered
        trace.ledger.append(
            LedgerEntry(
                operation="discord_delivery",
                status="succeeded" if trace.delivered else "failed",
                evidence_id="delivery-1",
                target_sha256=trace.final_speech_sha256,
            )
        )
        return RuntimeResult(
            "delivered" if trace.delivered else "delivery_failed",
            turn.speech if trace.delivered else "",
            trace,
        )

    def _call_actor(
        self, context: ActorContext, trace: TurnTrace
    ) -> ActorEnvelope | None:
        trace.actor_calls += 1
        if trace.actor_calls > MAX_TOOL_CALLS + 1:
            trace.issues.append("ACTOR_BUDGET_EXHAUSTED")
            return None
        try:
            turn = self._actor.generate(context)
        except Exception:
            trace.issues.append("ACTOR_ERROR")
            return None
        if not isinstance(turn, ActorEnvelope):
            trace.issues.append("ACTOR_INVALID_OUTPUT")
            return None
        return turn

    def _execute_tool(
        self, call: ToolCall, trace: TurnTrace
    ) -> ToolOutcome | None:
        trace.tool_calls += 1
        operation = f"tool:{call.name}"
        target_sha256 = _sha256(call.query)
        try:
            outcome = self._executor.execute(call)
        except Exception:
            trace.issues.append("EXECUTOR_ERROR")
            trace.ledger.append(
                LedgerEntry(
                    operation=operation,
                    status="error",
                    evidence_id=f"tool-attempt-{trace.tool_calls}",
                    target_sha256=target_sha256,
                )
            )
            return None
        if not isinstance(outcome, ToolOutcome):
            trace.issues.append("EXECUTOR_INVALID_OUTCOME")
            trace.ledger.append(
                LedgerEntry(
                    operation=operation,
                    status="invalid",
                    evidence_id=f"tool-attempt-{trace.tool_calls}",
                    target_sha256=target_sha256,
                )
            )
            return None
        if outcome.tool_name != call.name:
            trace.issues.append("TOOL_OUTCOME_IDENTITY_MISMATCH")
            trace.ledger.append(
                LedgerEntry(
                    operation=operation,
                    status="identity_mismatch",
                    evidence_id=outcome.evidence_id,
                    target_sha256=target_sha256,
                )
            )
            return None
        trace.ledger.append(
            LedgerEntry(
                operation=operation,
                status=outcome.status,
                evidence_id=outcome.evidence_id,
                target_sha256=target_sha256,
            )
        )
        return outcome

    @staticmethod
    def _gate_memories(
        turn: ActorEnvelope,
        authority: AuthoritySnapshot,
        trace: TurnTrace,
    ) -> None:
        for candidate in turn.memory_candidates:
            authorized = (
                authority.persistent_memory_allowed
                and candidate.kind in authority.allowed_memory_kinds
                and not _contains_sensitive_memory(candidate.text)
            )
            if authorized:
                trace.authorized_memory_candidates += 1
                status = "authorized"
            else:
                trace.rejected_memory_candidates += 1
                status = "blocked"
            trace.ledger.append(
                LedgerEntry(
                    operation="memory_candidate",
                    status=status,
                    evidence_id=(
                        "memory-"
                        f"{trace.authorized_memory_candidates + trace.rejected_memory_candidates}"
                    ),
                    target_sha256=_sha256(candidate.text),
                )
            )


_SENSITIVE = re.compile(
    r"(?:비밀번호|패스워드|API\s*키|토큰|주민등록|계좌|카드번호|"
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b)",
    re.IGNORECASE,
)
_EXECUTION_CLAIMS = {
    "message_delivery": re.compile(
        r"(?:메시지|문자|메일|공지).{0,20}(?:보냈어|전송했어|게시했어|올렸어)"
    ),
    "file_mutation": re.compile(r"(?:파일|폴더).{0,20}(?:만들었어|수정했어|삭제했어)"),
    "reminder_registration": re.compile(
        r"(?:알림|예약|리마인더).{0,20}(?:설정했어|등록했어|예약했어)"
    ),
}


def _contains_sensitive_query(query: str) -> bool:
    return bool(_SENSITIVE.search(" ".join(query.split())))


def _contains_sensitive_memory(text: str) -> bool:
    return bool(_SENSITIVE.search(" ".join(text.split())))


def _unsupported_execution_claim(
    speech: str,
    ledger: Sequence[LedgerEntry],
) -> str | None:
    succeeded = {entry.operation for entry in ledger if entry.status == "succeeded"}
    for operation, pattern in _EXECUTION_CLAIMS.items():
        if pattern.search(" ".join(speech.split())) and operation not in succeeded:
            return operation
    return None


def _acknowledges_unresolved_tool(speech: str) -> bool:
    compact = " ".join(speech.split())
    return bool(
        re.search(
            r"(?:확인할\s*수\s*없|알\s*수\s*없|결과가\s*없|도구.{0,10}실패|"
            r"응답하지\s*않|다시\s*알려)",
            compact,
        )
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
