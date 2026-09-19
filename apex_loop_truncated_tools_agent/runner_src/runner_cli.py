"""Standalone entry point for the pinned truncated-loop agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from atif import write_atif_trajectory
from runner.agents.loop_truncated_tools_agent.main import run
from runner.agents.models import AgentRunInput, AgentStatus


async def execute(args: argparse.Namespace) -> None:
    messages = json.loads(Path(args.initial_messages).read_text())
    config = json.loads(Path(args.agent_config).read_text())
    extra_raw = os.environ.get("APEX_MODEL_EXTRA_ARGS", "").strip()
    extra_args = json.loads(extra_raw) if extra_raw else None
    result = await run(
        AgentRunInput(
            trajectory_id=args.trajectory_id,
            initial_messages=messages,
            mcp_gateway_url=args.mcp_gateway_url,
            mcp_gateway_auth_token=None,
            orchestrator_model=args.orchestrator_model,
            orchestrator_extra_args=extra_args,
            agent_config_values=config["agent_config_values"],
        )
    )
    native_path = Path(args.output)
    native_json = result.model_dump_json(indent=2)
    native_path.write_text(native_json)
    # The verifier must not depend on a host download/convert/upload round trip.
    write_atif_trajectory(json.loads(native_json), native_path.with_name("trajectory.json"))
    if result.status in (AgentStatus.ERROR, AgentStatus.CANCELLED):
        raise RuntimeError(f"Agent execution failed; native trajectory saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--initial-messages", required=True)
    parser.add_argument("--mcp-gateway-url", required=True)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--orchestrator-model", required=True)
    parser.add_argument("--output", required=True)
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
