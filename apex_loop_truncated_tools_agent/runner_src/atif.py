"""Convert APEX trajectories without depending on Harbor or the agent runtime."""

import json
from pathlib import Path


def _text(content):
    """Normalize message content to a string (handle content-parts lists)."""
    if content is None:
        return ""
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def _args_obj(arguments):
    """tool_call arguments arrive as a JSON string; ATIF wants an object."""
    if isinstance(arguments, dict):
        return arguments
    try:
        v = json.loads(arguments)
        return v if isinstance(v, dict) else {"_value": v}
    except Exception:
        return {"_raw": str(arguments)}


def _metrics_from_call(c):
    if not c:
        return None
    m = {
        "prompt_tokens": c.get("prompt_tokens"),
        "completion_tokens": c.get("completion_tokens"),
        "cached_tokens": c.get("cached_tokens"),
    }
    extra = {
        k: c[k]
        for k in ("cache_creation_tokens", "reasoning_tokens", "total_tokens")
        if k in c
    }
    if extra:
        m["extra"] = extra
    return {k: v for k, v in m.items() if v is not None or k == "extra"}


def convert_trajectory(traj: dict) -> dict:
    """APEX-native trajectory -> ATIF (the harbor trajectory standard)."""
    messages = traj.get("messages") or []
    call_log = (traj.get("usage") or {}).get("call_log") or []

    steps = []
    a_idx = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")
        if role in ("system", "user"):
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "source": role,
                    "message": _text(msg.get("content")),
                }
            )
            i += 1
            continue
        if role == "assistant":
            step = {
                "step_id": len(steps) + 1,
                "source": "agent",
                "message": _text(msg.get("content")),
            }
            atif_tcs = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                cid = tc.get("id") or fn.get("name")
                atif_tcs.append(
                    {
                        "tool_call_id": cid,
                        "function_name": fn.get("name"),
                        "arguments": _args_obj(fn.get("arguments")),
                    }
                )
            if atif_tcs:
                step["tool_calls"] = atif_tcs
            mx = _metrics_from_call(call_log[a_idx] if a_idx < len(call_log) else None)
            if mx:
                step["metrics"] = mx
            a_idx += 1
            results = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tm = messages[j]
                results.append(
                    {
                        "source_call_id": tm.get("tool_call_id"),
                        "content": _text(tm.get("content")),
                    }
                )
                j += 1
            if results:
                step["observation"] = {"results": results}
            steps.append(step)
            i = j
            continue
        steps.append(
            {
                "step_id": len(steps) + 1,
                "source": "agent",
                "message": _text(msg.get("content")),
                "extra": {"native_role": role},
            }
        )
        i += 1

    u = traj.get("usage") or {}
    final_metrics = {
        k: v
        for k, v in {
            "total_prompt_tokens": u.get("prompt_tokens"),
            "total_completion_tokens": u.get("completion_tokens"),
            "total_cached_tokens": u.get("cached_tokens"),
            "total_steps": len(steps),
        }.items()
        if v is not None
    }
    return {
        "schema_version": "ATIF-v1.5",
        "session_id": traj.get("session_id") or "apex-agent",
        "agent": {"name": "apex_loop_truncated_tools_agent", "version": "1.0"},
        "steps": steps,
        "final_metrics": final_metrics,
        "extra": {
            "converted_from": "apex-native",
            "native_status": traj.get("status"),
            "time_elapsed_sec": traj.get("time_elapsed"),
            "native_output": traj.get("output"),
        },
    }


def write_atif_trajectory(native: dict, path: Path) -> dict:
    """Publish a complete ATIF beside its temporary file on the same filesystem."""
    atif = convert_trajectory(native)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(atif, indent=2))
    temporary.replace(path)
    return atif
