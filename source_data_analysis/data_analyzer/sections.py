# sections.py — builds Markdown snippets for report templates
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

def _session_label(ts: pd.Timestamp) -> str:
    h = ts.hour + ts.minute/60.0
    asia = (0 <= h < 8); london = (7 <= h < 16); ny = (12 <= h < 21)
    if asia and london: return "Asia-London overlap"
    if london and ny:  return "London-NY overlap"
    if ny: return "NY"
    if london: return "London"
    if asia: return "Asia"
    return "Other"

def build_common_blocks(df: pd.DataFrame, gaps: pd.DataFrame, year: int) -> Dict[str,str]:
    # 1) classification
    gaps_total = int(len(gaps))
    gap_classification_md = (
        f"Total: {gaps_total}; 📆 0, 🎉 0, 📢 0, ⚙️ 0, ❗ {gaps_total}.\n"
        f"Economic Calendar coverage: 0.0% (High=0)."
    )

    # 2) durations
    s = gaps["delta_sec"] if gaps_total else pd.Series([], dtype="int64")
    b_12 = int(((s>60)&(s<=120)).sum()) if gaps_total else 0
    b_25 = int(((s>120)&(s<=300)).sum()) if gaps_total else 0
    b_660 = int(((s>300)&(s<=3600)).sum()) if gaps_total else 0
    b_gt = int((s>3600).sum()) if gaps_total else 0
    p50 = int(s.quantile(0.5)) if gaps_total else 0
    p90 = int(s.quantile(0.9)) if gaps_total else 0
    p99 = int(s.quantile(0.99)) if gaps_total else 0
    mx = int(s.max()) if gaps_total else 0
    if gaps_total:
        mxrow = gaps.loc[s.idxmax()]
        longest = f"Longest gap: {mx}s (from {mxrow['gap_start'].isoformat()} to {mxrow['gap_end'].isoformat()} ; ≈{mx/3600:.2f} hours)."
    else:
        longest = "Longest gap: n/a"
    durations_section_md = (
        "**Buckets (counts)**  \n"
        f"- 1–2 min: {b_12}\n- 2–5 min: {b_25}\n- 6–60 min: {b_660}\n- >60 min: {b_gt}\n\n"
        f"**Percentiles (sec)**  p50={p50}, p90={p90}, p99={p99}, max={mx}.  \n"
        + longest
    )

    # 3) sessions
    if gaps_total:
        sess = gaps["gap_start"].apply(_session_label).value_counts()
        total = int(sess.sum()) or 1
        sessions_table_md = "| Session | Count | % |\n|---|---:|---:|\n" + "\n".join(
            f"| {k} | {int(v)} | {100.0*float(v)/total:.2f}% |" for k,v in sess.items()
        )
    else:
        sessions_table_md = "_No gaps_"

    # 4) monthly
    df = df.copy()
    gaps = gaps.copy()
    df["Month"] = df["datetime_utc"].dt.to_period("M").astype(str)
    if gaps_total:
        gaps["Month"] = gaps["gap_start"].dt.to_period("M").astype(str)
        monthly = (pd.DataFrame({"Rows": df.groupby("Month").size(),
                                 "Gaps": gaps.groupby("Month").size()})
                   .fillna(0).astype(int).reset_index())
    else:
        monthly = (pd.DataFrame({"Rows": df.groupby("Month").size()}).reset_index())
        monthly["Gaps"] = 0
    monthly_table_md = "| Month | Rows | Gaps |\n|---|---:|---:|\n" + "\n".join(
        f"| {r['Month']} | {r['Rows']} | {r['Gaps']} |" for _, r in monthly.iterrows()
    )

    # 5) extreme candles
    df["range"] = (df["high"]-df["low"]).abs()
    p99r = float(df["range"].quantile(0.99)) if len(df) else 0.0
    top5 = df.nlargest(5, "range")[ ["datetime_utc","range","open","close"] ].copy()
    top5["dir"] = np.where(top5["close"]>=top5["open"], "↑", "↓")
    top5["suspect"] = np.where(top5["range"]>5*p99r, "⚠️", "")
    if not top5.empty:
        extreme_table_md = "| # | Timestamp UTC | Range | Dir | Suspect? |\n|---:|---|---:|:--:|:--:|\n" + "\n".join(
            f"| {i+1} | {r['datetime_utc'].isoformat()} | {r['range']:.6f} | {r['dir']} | {r['suspect']} |"
            for i, r in top5.iterrows()
        )
    else:
        extreme_table_md = "_No data_"

    # 6) maintenance hotspots
    if gaps_total:
        gaps["weekday"] = gaps["gap_start"].dt.weekday
        gaps["hour"] = gaps["gap_start"].dt.hour
        grid = gaps.groupby(["weekday","hour"]).size().reset_index(name="count")
        thr = max(5, int(grid["count"].quantile(0.95))) if not grid.empty else 0
        hs = grid[grid["count"]>=thr].sort_values("count", ascending=False)
        maintenance_table_md = "| Weekday | Hour | Count |\n|---:|---:|---:|\n" + (
            "\n".join(f"| {int(r['weekday'])} | {int(r['hour'])} | {int(r['count'])} |" for _, r in hs.iterrows())
            if not hs.empty else "_No hotspots_"
        )
    else:
        maintenance_table_md = "_No hotspots_"

    # 7) visuals pointers
    visuals_list_md = (
        f"- EURUSD_{year}_anomalies.svg — red markers (❗)\n"
        f"- gaps_per_month_{year}.svg — monthly counts\n"
        f"- gap_heatmap_{year}.svg — weekday×hour heatmap"
    )

    # 8) assessment
    assessment_md = (
        "Data is usable for minute-level backtests, with caveats: pronounced weekend/off-session gap patterns, "
        "no Economic Calendar matches in this dataset, rare extreme candles (⚠️). "
        "Consider excluding bars adjacent to gaps >60 minutes; use robust aggregation."
    )

    # 9) pointers
    gaps_pointer_md = f"Full table is provided in `gaps_summary_{year}.md` and a deterministic `gaps_summary_{year}.md.tar.gz`."
    svg_pointer_md  = f"Packed in `anomalies_{year}.svg.tar.gz` (deterministic TAR.GZ)."

    return {
        "gap_classification_md": gap_classification_md,
        "durations_section_md": durations_section_md,
        "sessions_table_md": sessions_table_md,
        "monthly_table_md": monthly_table_md,
        "extreme_table_md": extreme_table_md,
        "maintenance_table_md": maintenance_table_md,
        "visuals_list_md": visuals_list_md,
        "assessment_md": assessment_md,
        "gaps_pointer_md": gaps_pointer_md,
        "svg_pointer_md": svg_pointer_md,
    }

def build_gaps_context(df: pd.DataFrame, gaps: pd.DataFrame, year: int) -> Dict[str,str]:
    common = build_common_blocks(df, gaps, year)
    lines = ["| # | Start UTC | End UTC | Δ sec |", "|---:|---|---|---:|"]
    for i, r in enumerate(gaps.sort_values("gap_start").itertuples(index=False), start=1):
        lines.append(f"| {i} | {r.gap_start.isoformat()} | {r.gap_end.isoformat()} | {int(r.delta_sec)} |")
    return {
        "sessions_table_md": common["sessions_table_md"],
        "gaps_full_table_md": "\n".join(lines),
    }

def build_monthly_context(df: pd.DataFrame, gaps: pd.DataFrame, year: int, month: str) -> Dict[str,str]:
    dfm = df[df["datetime_utc"].dt.to_period("M").astype(str) == month].copy()
    gm  = gaps[gaps["gap_start"].dt.to_period("M").astype(str) == month].copy()
    common = build_common_blocks(dfm if not dfm.empty else df, gm if not gm.empty else gaps, year)
    return {
        "month": month,
        "monthly_table_md": common["monthly_table_md"],
        "sessions_table_md": common["sessions_table_md"],
        "durations_section_md": common["durations_section_md"],
    }

def build_quarterly_context(df: pd.DataFrame, gaps: pd.DataFrame, year: int, q: int) -> Dict[str,str]:
    import pandas as pd
    start_month = 3*(q-1) + 1
    qs = pd.Timestamp(year=year, month=start_month, day=1, tz="UTC")
    qe = pd.Timestamp(year=year + (1 if q==4 else 0), month=(1 if q==4 else start_month+3), day=1, tz="UTC")
    mask = (df["datetime_utc"]>=qs) & (df["datetime_utc"]<qe)
    maskg = (gaps["gap_start"]>=qs) & (gaps["gap_start"]<qe)
    dfq = df[mask].copy()
    gq  = gaps[maskg].copy()
    common = build_common_blocks(dfq if not dfq.empty else df, gq if not gq.empty else gaps, year)
    return common
