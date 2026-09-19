import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backtalk.brains import config as router_config


class RouterConfigTests(unittest.TestCase):
    def test_defaults_are_all_disabled(self):
        for bid, spec in router_config.DEFAULTS["brains"].items():
            self.assertFalse(spec["enabled"], f"{bid} must default disabled")
        self.assertIsNone(router_config.DEFAULTS["active_brain"])

    def test_missing_file_returns_defaults(self):
        missing = Path(tempfile.gettempdir()) / "brain_router_does_not_exist.json"
        with mock.patch.object(router_config, "CONFIG_PATH", missing):
            cfg = router_config.load()
        self.assertEqual(cfg, router_config.DEFAULTS)

    def test_malformed_json_falls_back_to_defaults(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            f.write("{not valid json")
            path = Path(f.name)
        try:
            with mock.patch.object(router_config, "CONFIG_PATH", path):
                cfg = router_config.load()
            self.assertEqual(cfg, router_config.DEFAULTS)
        finally:
            path.unlink(missing_ok=True)

    def test_partial_override_merges_not_replaces(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            json.dump({"brains": {"qwen3-8b-local": {"enabled": True}}}, f)
            path = Path(f.name)
        try:
            with mock.patch.object(router_config, "CONFIG_PATH", path):
                cfg = router_config.load()
            self.assertTrue(cfg["brains"]["qwen3-8b-local"]["enabled"])
            # a partial override must never widen a brain the file
            # never mentioned
            self.assertFalse(cfg["brains"]["claude"]["enabled"])
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
