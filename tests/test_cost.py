"""Cost regressions using LiteLLM's calculator with fixed local price entries."""

import json

import litellm
import pytest
from harbor.models.agent.context import AgentContext

from apex_loop_truncated_tools_agent import (
    ApexLoopTruncatedToolsAgent,
    convert_trajectory,
)


@pytest.fixture
def model(monkeypatch, request):
    name = f"openai/apex-cost-test-{request.node.name}"
    monkeypatch.setitem(
        litellm.model_cost,
        name,
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 8e-6,
            "cache_read_input_token_cost": 0.5e-6,
            "cache_creation_input_token_cost": 2.5e-6,
        },
    )
    return name


def trajectory(*calls):
    return {
        "messages": [{"role": "assistant", "content": "done"} for _ in calls],
        "usage": {
            "call_log": list(calls),
            **{
                key: sum(call.get(key) or 0 for call in calls)
                for key in ("prompt_tokens", "completion_tokens", "cached_tokens")
            },
        },
    }


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"cached_tokens": 600}, 0.0019),
        ({"cached_tokens": 600, "cache_creation_tokens": 200}, 0.0020),
        ({"reasoning_tokens": 80}, 0.0028),
    ],
)
def test_cache_and_reasoning_costs(model, extra, expected):
    atif = convert_trajectory(
        trajectory({"prompt_tokens": 1000, "completion_tokens": 100, **extra}),
        model,
    )
    metrics = atif["steps"][0]["metrics"]
    assert metrics["cost_usd"] == pytest.approx(expected)
    assert atif["final_metrics"]["total_cost_usd"] == pytest.approx(expected)
    assert metrics["completion_tokens"] == 100


def test_long_context_pricing_is_selected_per_call(model, monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost[model], "input_cost_per_token_above_1000_tokens", 4e-6
    )
    monkeypatch.setitem(
        litellm.model_cost[model], "output_cost_per_token_above_1000_tokens", 16e-6
    )
    atif = convert_trajectory(
        trajectory(
            *(
                {"prompt_tokens": count, "completion_tokens": 100}
                for count in (900, 900, 1100)
            )
        ),
        model,
    )
    assert [step["metrics"]["cost_usd"] for step in atif["steps"]] == pytest.approx(
        [0.0026, 0.0026, 0.0060]
    )
    assert atif["final_metrics"]["total_cost_usd"] == pytest.approx(0.0112)


@pytest.mark.parametrize("missing", ["prompt_tokens", "completion_tokens"])
def test_incomplete_call_usage_keeps_total_unknown(model, missing):
    known_call = {"prompt_tokens": 1000, "completion_tokens": 100}
    incomplete_call = {
        key: value for key, value in known_call.items() if key != missing
    }
    atif = convert_trajectory(trajectory(known_call, incomplete_call), model)
    assert atif["steps"][0]["metrics"]["cost_usd"] == pytest.approx(0.0028)
    assert atif["steps"][1]["metrics"].get("cost_usd") is None
    assert atif["final_metrics"].get("total_cost_usd") is None


@pytest.mark.parametrize(
    "model_name", [None, "openai/apex-unknown-cost-regression-model"]
)
def test_unknown_model_cost_is_not_zero(model_name):
    atif = convert_trajectory(
        trajectory({"prompt_tokens": 1000, "completion_tokens": 100}), model_name
    )
    assert atif["steps"][0]["metrics"].get("cost_usd") is None
    assert atif["final_metrics"].get("total_cost_usd") is None
    assert atif["final_metrics"]["total_prompt_tokens"] == 1000


@pytest.mark.parametrize(
    ("missing_rate", "extra"),
    [
        ("input_cost_per_token", {}),
        ("output_cost_per_token", {}),
        ("cache_read_input_token_cost", {"cached_tokens": 100}),
        ("cache_creation_input_token_cost", {"cache_creation_tokens": 100}),
    ],
)
def test_incomplete_pricing_stays_unknown(model, monkeypatch, missing_rate, extra):
    monkeypatch.delitem(litellm.model_cost[model], missing_rate)
    atif = convert_trajectory(
        trajectory({"prompt_tokens": 1000, "completion_tokens": 100, **extra}), model
    )
    assert atif["steps"][0]["metrics"].get("cost_usd") is None
    assert atif["final_metrics"].get("total_cost_usd") is None


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_bare_model_price_requires_the_same_provider(model, monkeypatch, provider):
    pricing = {**litellm.model_cost[model], "litellm_provider": provider}
    monkeypatch.delitem(litellm.model_cost, model)
    monkeypatch.setitem(litellm.model_cost, model.split("/", 1)[1], pricing)
    atif = convert_trajectory(
        trajectory({"prompt_tokens": 1000, "completion_tokens": 100}), model
    )
    cost = atif["final_metrics"].get("total_cost_usd")
    if provider == "openai":
        assert cost == pytest.approx(0.0028)
    else:
        assert cost is None


def test_calculator_failure_preserves_usage(model, monkeypatch):
    def unavailable(**kwargs):
        raise ValueError("price unavailable")

    monkeypatch.setattr(litellm, "cost_per_token", unavailable)
    atif = convert_trajectory(
        trajectory(
            {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 600}
        ),
        model,
    )
    assert atif["final_metrics"].get("total_cost_usd") is None
    assert atif["final_metrics"]["total_prompt_tokens"] == 1000
    assert atif["final_metrics"]["total_completion_tokens"] == 100
    assert atif["final_metrics"]["total_cached_tokens"] == 600


def test_explicit_zero_pricing_is_a_known_cost(model, monkeypatch):
    for key in ("input_cost_per_token", "output_cost_per_token"):
        monkeypatch.setitem(litellm.model_cost[model], key, 0)
    atif = convert_trajectory(
        trajectory({"prompt_tokens": 1000, "completion_tokens": 100}), model
    )
    assert atif["steps"][0]["metrics"]["cost_usd"] == 0
    assert atif["final_metrics"]["total_cost_usd"] == 0


def test_absent_call_log_does_not_turn_aggregate_usage_into_zero_cost(model):
    atif = convert_trajectory(
        {"messages": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 100}},
        model,
    )
    assert atif["final_metrics"].get("total_cost_usd") is None
    assert atif["final_metrics"]["total_prompt_tokens"] == 1000


def test_calls_without_assistant_messages_still_contribute_to_total(model):
    native = trajectory(
        {"prompt_tokens": 1000, "completion_tokens": 100},
        {"prompt_tokens": 2000, "completion_tokens": 200},
    )
    native["messages"].pop(0)
    atif = convert_trajectory(native, model)
    assert atif["final_metrics"]["total_cost_usd"] == pytest.approx(0.0084)
    assert atif["steps"][0]["metrics"].get("cost_usd") is None


def test_missing_response_usage_does_not_report_a_partial_total(model):
    native = trajectory({"prompt_tokens": 1000, "completion_tokens": 100})
    native["messages"].append({"role": "assistant", "content": "no usage"})
    atif = convert_trajectory(native, model)
    assert atif["final_metrics"].get("total_cost_usd") is None
    assert all(
        step.get("metrics", {}).get("cost_usd") is None for step in atif["steps"]
    )


def test_synced_native_trajectory_populates_harbor_cost(model, tmp_path):
    native = trajectory(
        {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 600}
    )
    (tmp_path / "trajectory.native.json").write_text(json.dumps(native))
    agent = ApexLoopTruncatedToolsAgent(logs_dir=tmp_path, model_name=model)
    context = AgentContext()
    agent.populate_context_post_run(context)
    saved = json.loads((tmp_path / "trajectory.json").read_text())
    assert context.cost_usd == pytest.approx(0.0019)
    assert saved["final_metrics"]["total_cost_usd"] == pytest.approx(context.cost_usd)
    assert (
        context.n_input_tokens,
        context.n_output_tokens,
        context.n_cache_tokens,
    ) == (1000, 100, 600)
