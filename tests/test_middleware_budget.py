"""Budget middleware: counters, prompt nudge, schema visibility."""
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from scripted_model import ScriptedModel, call

from research_deepagent.agent import build_graph, research_sub_agent
from research_deepagent.middleware import ResearchBudgetMiddleware

CONFIG = {"configurable": {"thread_id": "budget"}}


def _graph(model, budget):
    return build_graph(
        model,
        checkpointer=InMemorySaver(),
        middleware=[TodoListMiddleware(), ResearchBudgetMiddleware(max_model_calls=budget)],
    )


def test_budget_counts_turns_and_injects_notice():
    model = ScriptedModel(replies=[AIMessage(content="done")])
    graph = _graph(model, budget=1)
    snapshots = list(
        graph.stream({"messages": [{"role": "user", "content": "hi"}]}, CONFIG, stream_mode="values")
    )
    final = snapshots[-1]
    assert final["research_model_calls"] == 1
    assert final["research_budget_exhausted"] is True
    assert "[research-budget]" in model.prompts[-1]


def test_budget_counts_tool_calls_after_model():
    model = ScriptedModel(
        replies=[
            call("write_todos", {"todos": [{"content": "Check evidence", "status": "completed"}]}, "plan"),
            AIMessage(content="done"),
        ]
    )
    graph = _graph(model, budget=10)
    snapshots = list(
        graph.stream({"messages": [{"role": "user", "content": "hi"}]}, CONFIG, stream_mode="values")
    )
    final = snapshots[-1]
    assert final["research_model_calls"] == 2
    assert final["research_tool_calls"] == 1
    assert final["todos"][0]["status"] == "completed"


def test_budget_fields_are_not_part_of_the_input_schema():
    graph = _graph(ScriptedModel(replies=[AIMessage(content="done")]), budget=10)
    input_fields = set(graph.get_input_jsonschema().get("properties", {}))
    output_fields = set(graph.get_output_jsonschema().get("properties", {}))
    assert "messages" in input_fields
    assert "research_model_calls" not in input_fields
    assert "research_model_calls" in output_fields
    assert "research_budget_exhausted" in output_fields


def test_research_subagent_gets_its_own_budget_guard():
    guards = [m for m in research_sub_agent["middleware"] if isinstance(m, ResearchBudgetMiddleware)]
    assert len(guards) == 1


def test_graph_root_run_is_named_research():
    graph = _graph(ScriptedModel(replies=[AIMessage(content="done")]), budget=10)
    assert graph.name == "research"
