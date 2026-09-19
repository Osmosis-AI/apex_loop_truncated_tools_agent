"""Execution failures stay inspectable without changing model-failure semantics."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from litellm import ModelResponse
from litellm.exceptions import BadRequestError, RateLimitError

import runner_cli
from runner.agents.models import AgentStatus, AgentTrajectoryOutput
from runner.utils import llm


class FailureReportingTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_error_raises_after_preserving_trajectory(self):
        for status in (AgentStatus.ERROR, AgentStatus.FAILED, AgentStatus.COMPLETED):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                messages = [{"role": "user", "content": "Task"}]
                (root / "messages.json").write_text(json.dumps(messages))
                (root / "config.json").write_text('{"agent_config_values": {}}')
                args = argparse.Namespace(
                    initial_messages=root / "messages.json",
                    agent_config=root / "config.json",
                    trajectory_id="test",
                    mcp_gateway_url="http://world:8000/mcp/",
                    orchestrator_model="openai/test-model",
                    output=root / "trajectory.native.json",
                )
                result = AgentTrajectoryOutput(
                    messages=messages, status=status, time_elapsed=1,
                    usage={"prompt_tokens": 12},
                )
                with patch.dict("os.environ", {"APEX_MODEL_EXTRA_ARGS": ""}), patch(
                    "runner_cli.run", new=AsyncMock(return_value=result)
                ):
                    if status == AgentStatus.ERROR:
                        with self.assertRaisesRegex(RuntimeError, "native trajectory saved"):
                            await runner_cli.execute(args)
                    else:
                        await runner_cli.execute(args)
                self.assertEqual(
                    json.loads(args.output.read_text()), json.loads(result.model_dump_json())
                )

    async def test_image_patch_limit_is_not_retried(self):
        error = BadRequestError(
            message=("The image you provided requires 49266 patches after processing, "
                     "exceeding the limit of 30000. Please resize the image and try again."),
            model="test-model", llm_provider="openai",
        )
        with patch.object(llm, "acompletion", new=AsyncMock(side_effect=error)) as call, patch(
            "runner.utils.decorators.asyncio.sleep", new=AsyncMock()
        ) as sleep:
            with self.assertRaises(BadRequestError):
                await llm.generate_response("openai/test-model", [], [], 600, {})
        self.assertEqual(call.await_count, 1)
        sleep.assert_not_awaited()

    async def test_transient_rate_limit_still_retries(self):
        error = RateLimitError(
            message="Too many requests", model="test-model", llm_provider="openai"
        )
        response = ModelResponse(choices=[{"message": {"role": "assistant", "content": "Done"}}])
        with patch.object(llm, "acompletion", new=AsyncMock(side_effect=[error, response])) as call, patch(
            "runner.utils.decorators.asyncio.sleep", new=AsyncMock()
        ) as sleep:
            self.assertEqual(
                await llm.generate_response("openai/test-model", [], [], 600, {}), response
            )
        self.assertEqual(call.await_count, 2)
        sleep.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
