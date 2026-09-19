# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Config for the brain router -- kept in its OWN file, brain_router.json,
deliberately separate from backtalk.json. Two reasons: (1) backtalk.json
is upstream-shaped, and this stays untouched so a future `git pull` from
the upstream jaredrhod/backtalk remote never conflicts with router
config; (2) every brain ships off by default without touching the file
real users already have, so nothing about the live F8 path changes just
because this file exists.

Phase 1: every brain defaults to enabled=False, and active_brain defaults
to null. No code path here, or anywhere in router.py, flips one on by
itself.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent  # backtalk/
CONFIG_PATH = Path(os.environ.get("BRAIN_ROUTER_CONFIG")
                   or (REPO / "brain_router.json"))

DEFAULTS = {
    # Which brain a fresh router should activate on its own -- null
    # means "activate nothing; the caller decides." Phase 1 ships null:
    # nothing here picks a default brain for you.
    "active_brain": None,
    "brains": {
        "qwen3-8b-local":       {"enabled": False},
        "deepseek-r1-8b-local": {"enabled": False},
        "gemini":               {"enabled": False,
                                 "api_key_env": "GEMINI_API_KEY"},
        "claude":               {"enabled": False, "model": None},
    },
}


def load() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))          # deep copy
    try:
        user = json.loads(CONFIG_PATH.read_text())
    except FileNotFoundError:
        return cfg
    except ValueError as e:
        print(f"[brain_router] brain_router.json is not valid JSON "
              f"({e}) -- using all-disabled defaults", flush=True)
        return cfg
    if "active_brain" in user:
        cfg["active_brain"] = user["active_brain"]
    for bid, overrides in (user.get("brains") or {}).items():
        if bid in cfg["brains"] and isinstance(overrides, dict):
            cfg["brains"][bid].update(overrides)
        else:
            cfg["brains"][bid] = overrides
    return cfg
