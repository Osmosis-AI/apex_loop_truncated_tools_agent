"""Task-agnostic APEX benchmark agent for harbor (import-path plugin).

One agent serves ALL tasks in this delivery: the per-task world slug and
world secrets reach the WORLD container via the task's compose file, and the
agent runs inside the MAIN container, reaching the world gateway over the
trial network (http://world:8000/mcp/). Everything executes in-sandbox via
env.exec — no host ports; setup() provisions only this delivery's own
vendored runner into the main container.

Run shape:
    PYTHONPATH=<delivery>/agent harbor run -p tasks/mercor-<slug> \
        -a apex_loop_truncated_tools_agent:ApexLoopTruncatedToolsAgent -m <model>

This package exposes only the canonical 100-step truncated-loop harness.
"""
from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from runner_src.atif import convert_trajectory as convert_trajectory
from runner_src.atif import write_atif_trajectory

_RUNNER_DIR = "/agent_runner"
_LOG = "/logs/agent"

# Canonical system prompt recorded for this benchmark harness.
_AGENT_SYSTEM_PROMPTS = {
    "loop_truncated_tools_agent": "You are an agent that completes tasks independently. Use the tools and files provided to you to complete the task to the best of your ability. You should use the code_exec tool when needed, such as when calculating values. When calculating numbers, unless specified otherwise, use the exact values without rounding them.\n\nYou must attempt to execute the task. You cannot ask for help or further clarification. \n\nYou should not scattergun your answers. Scattergunning is when you provide alternate answers based on information that is not explicitly requested in the task prompt. If you do this, it will be marked wrong. Please note, providing answers to legitimately different cases that are explicitly requested by the prompt is NOT scattergunning. Please respond to all parts of the task, but commit to answers instead of hedging.\n\nYou do not have access to the internet. Do not try to look up the answer through requests, beautifulsoup, or any other packages.\n\nFor every tool except the code_exec tool, you may assume that all relevant files are located under the root path /. For the code_exec tool, however, you must explicitly use /filesystem/ as the root path to locate all relevant files.",
}


def _env_int(name, default):
    """Integer env override; non-numeric values warn and keep the default."""
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"WARNING: ignoring non-numeric {name}={raw!r}", file=sys.stderr)
        return default


class ApexLoopTruncatedToolsAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "apex_loop_truncated_tools_agent"

    def version(self) -> str:
        return "1.0"

    def __init__(self, logs_dir, model_name=None, logger=None, mcp_servers=None,
                 skills_dir=None, *, runner_src: str | None = None,
                 gateway_url: str = "http://world:8000/mcp/",
                 max_steps: int | None = None,
                 agent_timeout_sec: int | None = None,
                 agent_config_id: str | None = None, **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, logger=logger,
                         mcp_servers=mcp_servers, skills_dir=skills_dir)
        # The vendored runner ships next to this file in the delivery.
        self._runner_src = runner_src or str(
            Path(__file__).resolve().parent / "runner_src"
        )
        self._gateway_url = gateway_url
        # Canonical defaults; environment variables may override them.
        self._max_steps = (
            max_steps if max_steps is not None else _env_int("MAX_STEPS", 100)
        )
        self._agent_timeout = (
            agent_timeout_sec
            if agent_timeout_sec is not None
            else _env_int("AGENT_TIMEOUT_SEC", 10800)
        )
        # The delivery intentionally exposes only the benchmark harness.
        self._agent_config_id = (
            agent_config_id
            if agent_config_id is not None
            else "loop_truncated_tools_agent"
        )
        self._agent_name = self._agent_config_id.replace("_", " ").title()
        # Recorded by run() for populate_context_post_run.
        self._run_tail = ""
        self._return_code: int | None = None

    async def setup(self, environment: BaseEnvironment) -> None:
        # The world boots itself (compose sidecar, POST-healthcheck-gated) —
        # setup only provisions the runner project into the MAIN container.
        src = Path(self._runner_src).resolve()
        if not src.exists():
            raise FileNotFoundError(f"runner_src not found: {src}")
        with tempfile.TemporaryDirectory() as td:
            tar_path = Path(td) / "runner.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tf:
                for item in src.iterdir():
                    if item.name in (".venv", "__pycache__", ".git"):
                        continue
                    tf.add(item, arcname=item.name)
            await environment.exec(command=f"mkdir -p {_RUNNER_DIR} {_LOG}")
            await environment.upload_file(source_path=str(tar_path),
                                          target_path=f"{_RUNNER_DIR}/runner.tar.gz")
        await environment.exec(
            command=f"tar xzf {_RUNNER_DIR}/runner.tar.gz -C {_RUNNER_DIR}"
        )

    async def _preflight_credentials(self, model, environment) -> None:
        """Fail in seconds when the model's provider credentials are absent
        from the MAIN container env (the runner reads container env, not the
        host's) — a missing key otherwise burns the whole agent timeout in
        LLM retries."""
        required = {
            "anthropic": ("ANTHROPIC_API_KEY",),
            "openai": ("OPENAI_API_KEY",),
            "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
            "fireworks_ai": ("FIREWORKS_AI_API_KEY", "FIREWORKS_API_KEY"),
            "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
            "litellm_proxy": ("LITELLM_PROXY_API_KEY",),
        }.get(model.split("/", 1)[0].lower())
        if not required:
            return
        probe = await environment.exec(command="env")
        present = set()
        for line in (probe.stdout or "").splitlines():
            key, sep, value = line.partition("=")
            if sep and value:
                present.add(key)
        # The vendored runner auto-routes through the LiteLLM proxy whenever
        # the proxy pair is set, regardless of the model's provider prefix.
        if {"LITELLM_PROXY_API_BASE", "LITELLM_PROXY_API_KEY"} <= present:
            return
        if not any(key in present for key in required):
            raise ValueError(
                f"no credentials for model {model!r} in the task container:"
                f" set one of {' / '.join(required)} (or the LITELLM_PROXY_*"
                " pair) before running"
            )

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        # No hardcoded model fallback: a missing model fails loudly.
        model = self.model_name
        if not model:
            raise ValueError(
                "no agent model configured: pass -m <provider/model> to "
                "harbor run (run_task.sh resolves it from models.json)"
            )
        await self._preflight_credentials(model, environment)
        msgs = []
        system_prompt = _AGENT_SYSTEM_PROMPTS.get(self._agent_config_id)
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": instruction})
        messages = json.dumps(msgs)
        agent_config = json.dumps({
            "agent_config_id": "loop_truncated_tools_agent",
            "agent_name": "Loop w/ Tool Truncation",
            "agent_config_values": {
                "timeout": self._agent_timeout,
                "max_steps": self._max_steps,
                "max_output_chars": _env_int("MAX_OUTPUT_CHARS", 32768),
                "max_output_lines": _env_int("MAX_OUTPUT_LINES", 200),
                "tool_call_timeout": _env_int("TOOL_CALL_TIMEOUT", 60),
                "llm_response_timeout": _env_int("LLM_RESPONSE_TIMEOUT", 600),
            },
        })
        await environment.exec(command=(
            f"mkdir -p {_LOG}; "
            f"cat > {_LOG}/messages.json <<'EOF'\n{messages}\nEOF\n"
            f"cat > {_LOG}/agent_config.json <<'EOF'\n{agent_config}\nEOF"))
        run_cmd = f"""
set -eu
command -v uv >/dev/null 2>&1 || {{ echo "uv missing" >&2; exit 1; }}
PROJ_FILE=$(find {_RUNNER_DIR} -maxdepth 3 -name pyproject.toml | head -1)
[ -n "$PROJ_FILE" ] || {{ echo "no pyproject under {_RUNNER_DIR}" >&2; exit 1; }}
PROJ=$(dirname "$PROJ_FILE")
cd "$PROJ"
find runner -type d -exec touch {{}}/__init__.py \\; 2>/dev/null || true
uv sync --frozen --no-install-project >&2
rc=0
uv run python -m runner_cli \
  --trajectory-id "harbor-${{WORLD_TASK_SLUG:-task}}" \
  --initial-messages {_LOG}/messages.json \
  --mcp-gateway-url {self._gateway_url} \
  --agent-config {_LOG}/agent_config.json \
  --orchestrator-model {model!r} \
  --output {_LOG}/trajectory.native.json > {_LOG}/agent_run.log 2>&1 || rc=$?
cat {_LOG}/agent_run.log
exit $rc
"""
        res = await environment.exec(command=run_cmd,
                                     timeout_sec=self._agent_timeout + 900)
        tail = "\n".join(p for p in (res.stdout, res.stderr) if p)[-2000:]
        self.logger.info(f"[apex-agent] run tail:\n{tail}")
        # The sandbox writes both trajectories. The context is filled in
        # populate_context_post_run: harbor calls it once /logs/agent is on the
        # host, after a successful and after a failed run alike, but only while
        # the context is still empty — so nothing is written to it here.
        self._run_tail = tail
        self._return_code = res.return_code
        if res.return_code != 0:
            raise RuntimeError(
                f"APEX runner exited {res.return_code}"
                f" (see {_LOG}/agent_run.log); tail:\n{tail}"
            )

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Read the synced ATIF (or convert legacy native output) for context.

        harbor invokes this after /logs/agent has been synced to the host (a
        no-op when the environment bind-mounts it, a download otherwise), on
        success and after a failed run, so a crashed runner still errors the
        trial with whatever trajectory it produced preserved.
        """
        atif = self._write_atif_trajectory()
        context.metadata = {"tail": self._run_tail, "return_code": self._return_code}
        metrics = (atif or {}).get("final_metrics") or {}
        context.n_input_tokens = metrics.get("total_prompt_tokens")
        context.n_output_tokens = metrics.get("total_completion_tokens")
        context.n_cache_tokens = metrics.get("total_cached_tokens")

    def _write_atif_trajectory(self) -> dict | None:
        """Preserve sandbox ATIF; reconstruct only from readable native output."""
        atif_path = self.logs_dir / "trajectory.json"
        try:
            atif = json.loads(atif_path.read_text())
            if (
                isinstance(atif, dict)
                and isinstance(atif.get("steps"), list)
                and all(isinstance(step, dict) for step in atif["steps"])
                and isinstance(atif.get("final_metrics", {}), (dict, type(None)))
            ):
                return atif
        except (OSError, ValueError):
            pass
        if atif_path.exists():
            # A partial download must not overwrite the complete sandbox file
            # when Harbor uploads host logs. Retain the bytes for diagnosis.
            invalid_path = atif_path.with_name(f"trajectory.invalid-{uuid4().hex}.json")
            atif_path.rename(invalid_path)
            self.logger.warning(f"[apex-agent] unreadable ATIF retained at {invalid_path}")

        native_path = self.logs_dir / "trajectory.native.json"
        try:
            native = json.loads(native_path.read_text())
            if not isinstance(native, dict) or not isinstance(native.get("messages"), list):
                raise ValueError("native trajectory has no messages list")
        except (OSError, ValueError):
            self.logger.warning(
                f"[apex-agent] native trajectory missing/unreadable at "
                f"{native_path}; no ATIF reconstructed"
            )
            return None
        return write_atif_trajectory(native, atif_path)
