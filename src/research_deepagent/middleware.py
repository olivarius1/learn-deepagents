"""Custom middleware for the research deep agent.

`ResearchBudgetMiddleware` keeps a per-thread budget of model turns and tool
calls, exposes the counters in the graph state (visible to the API/frontend),
and injects a convergence hint into the system prompt once the budget is nearly
spent. It is attached to both the main agent and the research sub-agent because
sub-agents get their own middleware stack and their own state.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ModelRequest, ModelResponse, OmitFromSchema
from langchain_core.messages import AIMessage, SystemMessage
from typing_extensions import NotRequired


class ResearchBudgetState(AgentState):
    """Budget counters.

    All three fields are produced inside the graph, so they are omitted from the
    input schema; they stay in the output schema so `stream.values` / thread
    state can expose them.

    Note: `OmitFromSchema` defaults both flags to True, so output visibility has
    to be requested explicitly with `output=False`.
    """

    research_model_calls: NotRequired[Annotated[int, OmitFromSchema(input=True, output=False)]]
    research_tool_calls: NotRequired[Annotated[int, OmitFromSchema(input=True, output=False)]]
    research_budget_exhausted: NotRequired[Annotated[bool, OmitFromSchema(input=True, output=False)]]


class ResearchBudgetMiddleware(AgentMiddleware):
    """Count model turns/tool calls and nudge the model to converge.

    The middleware instance is shared across threads, so all mutable state lives
    in the graph state rather than on `self`.
    """

    state_schema = ResearchBudgetState

    def __init__(self, max_model_calls: int = 24, warn_ratio: float = 0.75) -> None:
        super().__init__()
        self.max_model_calls = max(1, int(max_model_calls))
        self.warn_at = max(1, min(self.max_model_calls, int(self.max_model_calls * warn_ratio)))

    # -- state accounting -------------------------------------------------
    def before_model(self, state: ResearchBudgetState, runtime: Any) -> dict[str, Any]:
        used = (state.get("research_model_calls") or 0) + 1
        update: dict[str, Any] = {"research_model_calls": used}
        if used >= self.max_model_calls:
            update["research_budget_exhausted"] = True
        return update

    def after_model(self, state: ResearchBudgetState, runtime: Any) -> dict[str, Any] | None:
        latest_ai: AIMessage | None = None
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage):
                latest_ai = message
                break
        if latest_ai is None:
            return None
        calls = len(latest_ai.tool_calls or []) + len(latest_ai.invalid_tool_calls or [])
        if not calls:
            return None
        return {"research_tool_calls": (state.get("research_tool_calls") or 0) + calls}

    # -- prompt injection --------------------------------------------------
    def _budget_note(self, request: ModelRequest) -> str | None:
        used = request.state.get("research_model_calls") or 0
        if used < self.warn_at:
            return None

        remaining = max(0, self.max_model_calls - used)
        if remaining == 0:
            directive = (
                "No model turns remain. Do not call more tools. Answer now with the "
                "evidence already collected and state any gaps explicitly."
            )
        else:
            directive = (
                f"{remaining} model turn(s) remain. Stop opening new research branches, "
                "verify what is already collected, and start synthesizing the answer."
            )
        return f"[research-budget] {used}/{self.max_model_calls} turns used. {directive}"

    def _apply_note(self, request: ModelRequest, note: str) -> ModelRequest:
        blocks = list(request.system_message.content_blocks) if request.system_message else []
        blocks.append({"type": "text", "text": note})
        return request.override(system_message=SystemMessage(content=blocks))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        note = self._budget_note(request)
        if note is None:
            return handler(request)
        return handler(self._apply_note(request, note))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        note = self._budget_note(request)
        if note is None:
            return await handler(request)
        return await handler(self._apply_note(request, note))
