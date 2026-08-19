"""
Entry-zone construction per setup type. Pure functions operating on plain
scalars (already extracted from the domain objects by setup_engine.py),
so each rule is easy to read and test in isolation.
"""

from __future__ import annotations

from sentinel.setups.models import EntryZone


def breakout_entry_zone(direction: str, current_price: float, boundary: float) -> EntryZone:
    """Zone between the broken structural boundary and the current price --
    covers both an immediate entry and a retest of the broken level."""
    return EntryZone(low=min(boundary, current_price), high=max(boundary, current_price))


def pullback_entry_zone(direction: str, current_price: float, structural_floor_or_ceiling: float) -> EntryZone:
    """Zone between the current (retraced) price and the structural level
    the pullback is retracing toward."""
    low = min(current_price, structural_floor_or_ceiling)
    high = max(current_price, structural_floor_or_ceiling)
    return EntryZone(low=low, high=high)


def trend_continuation_entry_zone(direction: str, current_price: float, recent_range_pct: float | None) -> EntryZone:
    """A tight zone around current price, sized from the market's own
    recent candle range (a real, computed value) rather than an arbitrary
    fixed percentage. No recent-range data -> a zero-width zone at the
    current price (a precise "enter now" rather than a fabricated buffer)."""
    buffer_pct = (recent_range_pct or 0.0) / 2.0
    buffer = current_price * buffer_pct / 100.0
    if direction == "long":
        return EntryZone(low=current_price - buffer, high=current_price)
    return EntryZone(low=current_price, high=current_price + buffer)
