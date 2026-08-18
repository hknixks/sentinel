from __future__ import annotations

from sentinel.market_state import Candle, SymbolState
from sentinel.scanner.scanner import MarketScanner, _percentile_ranks, _weighted_avg


def _candles(closes: list[float], volume: float = 100.0) -> tuple[Candle, ...]:
    out = []
    for i, close in enumerate(closes):
        open_ = closes[i - 1] if i > 0 else close
        out.append(
            Candle(
                open_time=i * 60_000,
                open=open_,
                high=max(open_, close),
                low=min(open_, close),
                close=close,
                volume=volume,
                is_closed=True,
            )
        )
    return tuple(out)


def _state(symbol: str, closes: list[float], volume: float = 100.0) -> SymbolState:
    return SymbolState(symbol=symbol, candle_history=_candles(closes, volume))


def test_percentile_ranks_orders_low_to_high_and_averages_ties():
    ranks = _percentile_ranks({"a": 1.0, "b": 2.0, "c": 2.0, "d": 3.0})

    assert ranks["a"] == 0.0
    assert ranks["d"] == 100.0
    assert ranks["b"] == ranks["c"] == 50.0


def test_percentile_ranks_single_value_defaults_to_fifty():
    assert _percentile_ranks({"a": 5.0}) == {"a": 50.0}


def test_weighted_avg_renormalizes_over_available_values():
    assert _weighted_avg([(None, 0.5), (80.0, 0.5)]) == 80.0


def test_weighted_avg_returns_none_when_nothing_available():
    assert _weighted_avg([(None, 1.0)]) is None
    assert _weighted_avg([]) is None


def test_scanner_rejects_markets_with_insufficient_history():
    states = {"AUSDT": _state("AUSDT", [100.0] * 5)}

    results = MarketScanner().scan(states)

    assert results == []


def test_scanner_skips_malformed_state_without_crashing():
    states = {
        "GOODUSDT": _state("GOODUSDT", [100.0] * 15 + [110.0]),
        "BADUSDT": SymbolState(symbol="BADUSDT"),
    }

    results = MarketScanner().scan(states)

    assert [r.symbol for r in results] == ["GOODUSDT"]


def test_scanner_ranks_bigger_momentum_higher():
    states = {
        "BIGUSDT": _state("BIGUSDT", [100.0] * 19 + [150.0]),
        "MIDUSDT": _state("MIDUSDT", [100.0] * 19 + [110.0]),
        "SMALLUSDT": _state("SMALLUSDT", [100.0] * 19 + [101.0]),
    }

    results = MarketScanner().scan(states)

    assert [r.symbol for r in results] == ["BIGUSDT", "MIDUSDT", "SMALLUSDT"]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    for r in results:
        assert 0.0 <= r.score <= 100.0
        assert r.direction == "bullish"


def test_scanner_top_n_limits_results():
    states = {
        "BIGUSDT": _state("BIGUSDT", [100.0] * 19 + [150.0]),
        "MIDUSDT": _state("MIDUSDT", [100.0] * 19 + [110.0]),
        "SMALLUSDT": _state("SMALLUSDT", [100.0] * 19 + [101.0]),
    }

    results = MarketScanner().scan(states, top_n=2)

    assert len(results) == 2
    assert [r.symbol for r in results] == ["BIGUSDT", "MIDUSDT"]


def test_scanner_direction_bearish_for_downtrend():
    states = {"DOWNUSDT": _state("DOWNUSDT", [100.0] * 19 + [80.0])}

    results = MarketScanner().scan(states)

    assert results[0].direction == "bearish"


def test_scanner_direction_neutral_for_flat_market():
    states = {"FLATUSDT": _state("FLATUSDT", [100.0] * 20)}

    results = MarketScanner().scan(states)

    assert results[0].direction == "neutral"


def test_scanner_returns_empty_list_for_no_markets():
    assert MarketScanner().scan({}) == []
