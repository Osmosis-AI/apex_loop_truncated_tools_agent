"""Host fallback must not replace complete sandbox evidence with partial logs."""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apex_loop_truncated_tools_agent import ApexLoopTruncatedToolsAgent, convert_trajectory
from harbor.models.agent.context import AgentContext
from runner_src.atif import write_atif_trajectory


class TrajectoryHandoffTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.logs = Path(self.directory.name)
        self.native_path = self.logs / "trajectory.native.json"
        self.atif_path = self.logs / "trajectory.json"
        self.native = {
            "messages": [{"role": "user", "content": "Task"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            "status": "completed",
        }
        self.agent = ApexLoopTruncatedToolsAgent(
            logs_dir=self.logs, model_name="openai/test-model",
            logger=logging.getLogger("handoff-test"),
        )

    def test_downloaded_atif_is_preserved_and_populates_context(self):
        original = json.dumps(convert_trajectory(self.native), separators=(",", ":"))
        self.atif_path.write_text(original)
        # Prefer sandbox evidence even if a stale or damaged native copy exists.
        self.native_path.write_text('{"messages":')
        context = AgentContext()
        self.agent.populate_context_post_run(context)
        self.assertEqual(self.atif_path.read_text(), original)
        self.assertEqual(context.n_input_tokens, 12)
        self.assertEqual(context.n_output_tokens, 3)

    def test_native_only_legacy_output_is_reconstructed(self):
        self.native_path.write_text(json.dumps(self.native))
        self.assertEqual(self.agent._write_atif_trajectory(), convert_trajectory(self.native))
        self.assertEqual(json.loads(self.atif_path.read_text()), convert_trajectory(self.native))

    def test_missing_or_invalid_native_never_synthesizes_atif(self):
        for content in (None, '{"messages":', '{}', '[]', '{"messages": {}}'):
            with self.subTest(native=content):
                if content is not None:
                    self.native_path.write_text(content)
                context = AgentContext()
                self.agent.populate_context_post_run(context)
                self.assertFalse(self.atif_path.exists())
                self.assertIsNone(context.n_input_tokens)
                self.assertIsNotNone(context.metadata)

    def test_partial_atif_is_quarantined_before_optional_reconstruction(self):
        for has_native in (False, True):
            with self.subTest(has_native=has_native):
                partial = '{"steps":'
                self.atif_path.write_text(partial)
                if has_native:
                    self.native_path.write_text(json.dumps(self.native))
                result = self.agent._write_atif_trajectory()
                self.assertEqual(self.atif_path.exists(), has_native)
                self.assertEqual(result, convert_trajectory(self.native) if has_native else None)
                self.assertTrue(all(
                    path.read_text() == partial
                    for path in self.logs.glob("trajectory.invalid-*.json")
                ))
        self.assertEqual(len(list(self.logs.glob("trajectory.invalid-*.json"))), 2)

    def test_invalid_atif_structure_is_not_reuploaded(self):
        for content in ('[]', '{}', '{"steps": [null]}', '{"steps": [], "final_metrics": 1}'):
            with self.subTest(atif=content):
                self.atif_path.write_text(content)
                self.assertIsNone(self.agent._write_atif_trajectory())
                self.assertFalse(self.atif_path.exists())

    def test_quarantine_failure_stops_handoff(self):
        self.atif_path.write_text('{"steps":')
        with patch.object(Path, "rename", side_effect=OSError("cannot quarantine")):
            with self.assertRaisesRegex(OSError, "cannot quarantine"):
                self.agent._write_atif_trajectory()

    def test_failed_atomic_write_preserves_previous_atif(self):
        self.atif_path.write_text("previous complete trajectory")
        with patch.object(Path, "replace", side_effect=OSError("cannot publish")):
            with self.assertRaisesRegex(OSError, "cannot publish"):
                write_atif_trajectory(self.native, self.atif_path)
        self.assertEqual(self.atif_path.read_text(), "previous complete trajectory")


if __name__ == "__main__":
    unittest.main()
