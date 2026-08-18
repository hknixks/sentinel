from __future__ import annotations

from sentinel.market_state import Candle, SymbolState
from sentinel.scanner import features
from sentinel.scanner.features import FeatureEngine


def _candles(closes: list[float], volumes: list[float] | None = None) -> tuple[Candle, ...]:
    if volumes is None:
        volumes = [100.0] * len(closes)
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
                volume=volumes[i],
                is_closed=True,
            )
        )
    return tuple(out)


def _state(symbol: str, closes: list[float], volumes: list[float] | None = None) -> SymbolState:
    return SymbolState(symbol=symbol, candle_history=_candles(closes, volumes))


def test_return_1m_calculated_from_last_two_candles():
    state = _state("BTCUSDT", [100.0, 105.0])
    result = FeatureEngine().compute(state)

    assert result.returns.return_1m == 5.0
    assert result.returns.return_5m is None
    assert result.returns.return_15m is None
    assert result.returns.return_1h is None


def test_return_5m_calculated_from_six_candles():
    state = _state("BTCUSDT", [100, 101, 102, 103, 104, 110.0])
    result = FeatureEngine().compute(state)

    assert result.returns.return_5m == 10.0
    assert result.returns.return_15m is None


def test_return_15m_calculated_from_sixteen_candles():
    closes = [50.0] * 15 + [55.0]
    state = _state("BTCUSDT", closes)
    result = FeatureEngine().compute(state)

    assert result.returns.return_15m == 10.0
    assert result.returns.return_1h is None


def test_return_1h_calculated_from_sixty_one_candles():
    closes = [200.0] * 60 + [220.0]
    state = _state("BTCUSDT", closes)
    result = FeatureEngine().compute(state)

    assert result.returns.return_1h == 10.0


def test_relative_volume_compares_recent_to_baseline():
    volumes = [100.0] * 15 + [200.0] * 5
    state = _state("BTCUSDT", [1.0] * 20, volumes)
    result = FeatureEngine().compute(state)

    assert result.volume.relative_volume == 1.6


def test_volume_acceleration_compares_recent_to_prior_window():
    volumes = [100.0] * 5 + [150.0] * 5
    state = _state("BTCUSDT", [1.0] * 10, volumes)
    result = FeatureEngine().compute(state)

    assert result.volume.volume_acceleration == 50.0


def test_volatility_stdev_of_two_returns_matches_half_the_spread():
    closes = [100.0, 110.0, 100.0]
    got = features._stdev_pct_returns(closes)
    expected = abs(0.1 - (-1 / 11)) / 2 * 100
    assert got == expected


def test_rolling_volatility_is_zero_for_constant_percentage_growth():
    closes = [100.0 * (1.01**i) for i in range(21)]
    state = _state("BTCUSDT", closes)
    result = FeatureEngine().compute(state)

    assert result.volatility.rolling_volatility == 0.0 or abs(
        result.volatility.rolling_volatility
    ) < 1e-9


def test_trend_direction_up_when_short_ma_above_long_ma():
    closes = [100.0] * 15 + [110.0] * 5
    state = _state("BTCUSDT", closes)
    result = FeatureEngine().compute(state)

    assert result.trend.direction == "up"
    assert result.trend.trend_strength > 0


def test_trend_direction_flat_for_constant_price():
    closes = [100.0] * 20
    state = _state("BTCUSDT", closes)
    result = FeatureEngine().compute(state)

    assert result.trend.direction == "flat"
    assert result.trend.trend_strength == 0.0
    assert result.trend.timeframe_alignment is None


def test_no_candles_returns_none():
    state = SymbolState(symbol="BTCUSDT")
    assert FeatureEngine().compute(state) is None


def test_malformed_incomplete_state_returns_none():
    state = SymbolState(symbol="XUSDT", price=None, volume_24h=None, last_candle_1m=None)
    assert FeatureEngine().compute(state) is None


def test_insufficient_history_yields_none_fields_not_fabricated():
    state = _state("BTCUSDT", [100.0])
    result = FeatureEngine().compute(state)

    assert result is not None
    assert result.candle_count == 1
    assert result.returns.return_1m is None
    assert result.volume.relative_volume is None
    assert result.volatility.rolling_volatility is None
    assert result.trend.direction is None


def test_zero_volume_does_not_crash_and_yields_none_ratios():
    closes = [100.0] * 20
    volumes = [0.0] * 20
    state = _state("BTCUSDT", closes, volumes)
    result = FeatureEngine().compute(state)

    assert result.volume.volume_current == 0.0
    assert result.volume.relative_volume is None
    assert result.volume.volume_acceleration is None


def test_flat_price_yields_zero_returns_and_zero_volatility():
    closes = [42.0] * 21
    state = _state("BTCUSDT", closes)
    result = FeatureEngine().compute(state)

    assert result.returns.return_1m == 0.0
    assert result.volatility.recent_range == 0.0
    assert result.volatility.rolling_volatility == 0.0


def test_extremely_large_price_movement_computes_without_error():
    state = _state("BTCUSDT", [1.0, 1_000_000.0])
    result = FeatureEngine().compute(state)

    assert result.returns.return_1m == 99_999_900.0
