"""
Human-readable historical performance reports over Phase 7's
AnalyticsReport. Formatting only -- no calculation happens here (see
sentinel.analytics.aggregations for that).

Run standalone against the live outcome database:

    PYTHONPATH=src python3 -m sentinel.analytics.reports

This only reads OUTCOME_DB_PATH; it never touches the live scanner/
structure/setup/alert pipeline.
"""

from __future__ import annotations

from sentinel.analytics.aggregations import LOW_CONFIDENCE_SAMPLE_SIZE
from sentinel.analytics.analytics import generate_report_from_store
from sentinel.analytics.models import AnalyticsReport, GroupPerformance, MetricStats, OutcomeRates
from sentinel.config import OUTCOME_DB_PATH
from sentinel.outcomes.store import OutcomeStore


def _f(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def _fmt_outcome_rates(o: OutcomeRates, indent: str = "  ") -> str:
    flag = " [LOW CONFIDENCE]" if o.low_confidence else ""
    return (
        f"{indent}n={o.sample_size}{flag}  "
        f"TP1={_f(o.tp1_hit_rate, 1, '%')} TP2={_f(o.tp2_hit_rate, 1, '%')} "
        f"SL={_f(o.sl_hit_rate, 1, '%')} AMBIGUOUS={_f(o.ambiguous_rate, 1, '%')} "
        f"EXPIRED={_f(o.expired_rate, 1, '%')} PENDING={_f(o.pending_rate, 1, '%')}"
    )


def _fmt_group(g: GroupPerformance) -> str:
    flag = " [LOW CONFIDENCE]" if g.low_confidence else ""
    lines = [f"  {g.group}: n={g.sample_size}{flag}"]
    lines.append(_fmt_outcome_rates(g.outcomes, indent="    "))
    lines.append(
        "    avg/median MFE={}/{}  avg/median MAE={}/{}  avg/median fwd-return={}/{}  "
        "median time-to-resolution={}".format(
            _f(g.avg_mfe, 2, "%"), _f(g.median_mfe, 2, "%"),
            _f(g.avg_mae, 2, "%"), _f(g.median_mae, 2, "%"),
            _f(g.avg_forward_return, 2, "%"), _f(g.median_forward_return, 2, "%"),
            _f(g.median_time_to_resolution_seconds, 0, "s"),
        )
    )
    return "\n".join(lines)


def _fmt_group_dict(title: str, groups: dict[str, GroupPerformance]) -> str:
    lines = [title]
    for g in groups.values():
        lines.append(_fmt_group(g))
    return "\n".join(lines)


def _fmt_metric_stats(label: str, stats: MetricStats) -> str:
    flag = " [LOW CONFIDENCE]" if stats.count < LOW_CONFIDENCE_SAMPLE_SIZE else ""
    return (
        f"    {label}: n={stats.count}{flag} mean={_f(stats.mean)} median={_f(stats.median)} "
        f"min={_f(stats.minimum)} max={_f(stats.maximum)} "
        f"+%={_f(stats.positive_pct, 1)} -%={_f(stats.negative_pct, 1)}"
    )


def format_report(report: AnalyticsReport) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("SENTINEL Phase 7 -- Historical Signal Analytics")
    lines.append("Observational only. No figure below is a probability, a prediction,")
    lines.append("or a guarantee of future results -- it is a record of what already")
    lines.append("happened to past alerts.")
    lines.append("=" * 78)
    lines.append(
        f"Generated at {report.generated_at:.0f} | execution time {report.execution_seconds * 1000:.2f}ms"
    )
    lines.append(
        f"Total signals={report.total_signals}  Completed={report.completed_signals}  "
        f"Pending={report.pending_signals}"
    )
    lines.append("")
    lines.append("Overall outcome distribution:")
    lines.append(_fmt_outcome_rates(report.outcome_distribution))
    lines.append("")

    lines.append(_fmt_group_dict("Performance by setup type:", report.by_setup_type))
    lines.append("")
    lines.append(_fmt_group_dict("Performance by direction:", report.by_direction))
    lines.append("")
    lines.append(_fmt_group_dict("Performance by setup-score bucket:", report.by_setup_score_bucket))
    lines.append("")
    lines.append(_fmt_group_dict("Performance by scanner-activity-score bucket:", report.by_scanner_score_bucket))
    lines.append("")
    lines.append(_fmt_group_dict("Performance by structure-confidence bucket:", report.by_structure_confidence_bucket))
    lines.append("")
    lines.append(_fmt_group_dict("Performance by initial R:R bucket:", report.by_rr_bucket))
    lines.append("")
    lines.append(_fmt_group_dict("Performance by timeframe alignment:", report.by_timeframe_alignment))
    lines.append("")

    lines.append("Forward-return distribution by window (only windows actually evaluated):")
    for label, stats in report.forward_returns_by_window.items():
        flag = " [LOW CONFIDENCE]" if stats.low_confidence else ""
        lines.append(
            f"  {label}: n={stats.sample_size}{flag} avg={_f(stats.avg_return, 2, '%')} "
            f"median={_f(stats.median_return, 2, '%')} "
            f"+%={_f(stats.positive_pct, 1)} -%={_f(stats.negative_pct, 1)}"
        )
    lines.append("")

    lines.append("MFE/MAE by direction and window:")
    for m in report.mfe_mae_by_direction_window:
        lines.append(f"  {m.direction} / {m.window_label}:")
        lines.append(_fmt_metric_stats("MFE", m.mfe))
        lines.append(_fmt_metric_stats("MAE", m.mae))
    lines.append("")

    t = report.time_to_outcome
    lines.append("Time to outcome (seconds from signal to event):")
    lines.append(f"  TP1: n={t.sample_size_tp1} median={_f(t.median_seconds_to_tp1, 0, 's')}")
    lines.append(f"  TP2: n={t.sample_size_tp2} median={_f(t.median_seconds_to_tp2, 0, 's')}")
    lines.append(f"  SL: n={t.sample_size_sl} median={_f(t.median_seconds_to_sl, 0, 's')}")
    lines.append(f"  Final resolution: n={t.sample_size_resolved} median={_f(t.median_seconds_to_resolution, 0, 's')}")
    lines.append("")

    r = report.historical_r
    flag = " [LOW CONFIDENCE]" if r.low_confidence else ""
    lines.append("Historical R expectancy (realized, backward-looking only):")
    lines.append(f"  n={r.sample_size}{flag} avg_R={_f(r.avg_r)} median_R={_f(r.median_r)} win_rate={_f(r.win_rate, 1, '%')}")
    lines.append(f"  {r.label}")
    lines.append("")

    m = report.momentum_volume_volatility
    lines.append("Momentum / volume / volatility analysis:")
    lines.append(f"  available={m.available}")
    lines.append(f"  {m.reason}")

    return "\n".join(lines)


def main() -> None:
    store = OutcomeStore(OUTCOME_DB_PATH)
    report = generate_report_from_store(store)
    print(format_report(report))


if __name__ == "__main__":
    main()
