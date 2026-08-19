"""
Deterministic setup identity.

The same underlying market setup, observed across many repeated scanner
cycles, must always produce the same identity string. A genuinely new
setup (a new structural boundary/level) must produce a different one.
Never based on a timestamp -- two scans of the identical setup a minute
apart must hash to the same identity.

`invalidation_level` is the right anchor for "setup generation/change":
for breakout setups it is (after the entry-zone algebra used by
setup_engine.py) exactly the structural boundary that was broken; for
pullback/trend-continuation setups it is Phase 3's own swing-based
support/resistance level. Both are stable across scans of the same setup
and only change when a genuinely new structural level forms.
"""

from __future__ import annotations

import hashlib
import math

from sentinel.setups.models import SetupCandidate

_SIGFIGS = 5


def _round_sigfigs(value: float, sigfigs: int = _SIGFIGS) -> str:
    if value == 0:
        return "0"
    digits = sigfigs - int(math.floor(math.log10(abs(value)))) - 1
    digits = max(digits, 0)
    return f"{round(value, digits):.{digits}f}"


def compute_setup_identity(candidate: SetupCandidate) -> str:
    level_key = _round_sigfigs(candidate.invalidation_level)
    raw = f"{candidate.symbol}|{candidate.direction}|{candidate.setup_type}|{level_key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
