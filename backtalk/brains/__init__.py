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
"""Brain adapters -- Phase 1 scaffolding. See backtalk/backtalk/router.py
for the seam that will use these once a later phase wires them into
main.py. Every adapter here defaults to disabled; nothing in this
package touches the live F8 path yet.
"""
from backtalk.brains.base import (
    BrainAdapter,
    BrainDisabledError,
    BrainHealth,
    BrainStatus,
    BrainUnavailableError,
)

__all__ = [
    "BrainAdapter",
    "BrainDisabledError",
    "BrainHealth",
    "BrainStatus",
    "BrainUnavailableError",
]
