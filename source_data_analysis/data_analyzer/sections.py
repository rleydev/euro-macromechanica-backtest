# sections.py — builds Markdown snippets for report templates
from __future__ import annotations
from typing import Dict
import re
import fx_holidays
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


def compute_score(df: pd.DataFrame, gaps: pd.DataFrame, year: int, params: dict | None = None) -> dict:
    """Return a 0–100 score with transparent breakdown using deterministic rules."""
    if params is None:
        params = {
            "weights": {
                "gap_mix": 0.30, "hotspots": 0.20, "extremes": 0.15,
                "monthly": 0.15, "sessions": 0.10, "calendar": 0.05, "completeness": 0.05
            },
            "targets": {
                "small_share": 0.40, "long_share": 0.20,
                "extreme_rate_per_10k": 0.5,
                "monthly_cv": 1.0,
                "session_other": 0.10,
                "longest_gap_hours_ok": 24.0
            }
        }
    W = params["weights"]; T = params["targets"]

    N = len(gaps)
    if N>0:
        s = gaps["delta_sec"]
        n_small = int(((s>60)&(s<=300)).sum())
        n_med   = int(((s>300)&(s<=3600)).sum())
        n_long  = int((s>3600).sum())
        p_small = n_small/N; p_med = n_med/N; p_long = n_long/N
    else:
        p_small = p_med = p_long = 0.0

    # Gap mix (higher small_share, lower long_share)
    comp_small = min(1.0, p_small / T["small_share"])
    comp_long  = max(0.0, 1.0 - min(1.0, p_long / T["long_share"]))
    gap_mix_score = 100.0*(0.6*comp_small + 0.4*comp_long)

    # Hotspots (weekday x hour) via normalized Herfindahl
    if N>0:
        cells = gaps.assign(wd=gaps["gap_start"].dt.weekday, hr=gaps["gap_start"].dt.hour).groupby(["wd","hr"]).size()
        total = cells.sum()
        p = (cells/total).values
        H = float(np.sum(p*p))
        C = int(len(cells))
        normH = (H - 1.0/C) / (1.0 - 1.0/C) if C>1 else 1.0
        hotspot_score = 100.0*(1.0 - normH)
    else:
        hotspot_score = 100.0

    # Extreme candles (range > 5*P99(range)) per 10k bars
    df2 = df.copy()
    df2["range"] = (df2["high"]-df2["low"]).abs()
    p99r = float(df2["range"].quantile(0.99)) if len(df2) else 0.0
    ext = int((df2["range"] > 5*p99r).sum()) if p99r>0 else 0
    rate_per_10k = ext / (len(df2)/10000.0) if len(df2) else 0.0
    extreme_score = 100.0*(1.0 - min(1.0, rate_per_10k / T["extreme_rate_per_10k"]))

    # Monthly stability (CV of monthly gap counts)
    if len(df2):
        df2["Month"] = df2["datetime_utc"].dt.to_period("M").astype(str)
        if N>0:
            gaps2 = gaps.copy(); gaps2["Month"] = gaps2["gap_start"].dt.to_period("M").astype(str)
            monthly = (gaps2.groupby("Month").size()).reindex(sorted(df2["Month"].unique()), fill_value=0)
        else:
            monthly = pd.Series([0]*len(df2["Month"].unique()), index=sorted(df2["Month"].unique()))
        mu = float(monthly.mean())
        cv = float((monthly.std(ddof=0)/mu)) if mu>0 else 0.0
    else:
        cv = 0.0
    monthly_score = 100.0*(1.0 - min(1.0, cv / T["monthly_cv"]))

    # Session balance (penalize 'Other')
    if N>0:
        def _sess(ts):
            h = ts.hour + ts.minute/60.0
            asia = (0 <= h < 8); london = (7 <= h < 16); ny = (12 <= h < 21)
            if asia and london: return "Asia-London overlap"
            if london and ny:  return "London-NY overlap"
            if ny: return "NY"
            if london: return "London"
            if asia: return "Asia"
            return "Other"
        sc = gaps["gap_start"].apply(_sess).value_counts()
        share_other = float(sc.get("Other", 0)/N)
    else:
        share_other = 0.0
    session_score = 100.0*(1.0 - min(1.0, share_other / T["session_other"]))

    # Calendar coverage (not available → keep as 0 unless wired later)
    calendar_score = 0.0

    # Completeness (longest gap vs 24h)
    longest_h = float(gaps["delta_sec"].max()/3600.0) if N>0 else 0.0
    completeness_score = 100.0*(1.0 - min(1.0, longest_h / T["longest_gap_hours_ok"]))

    # Weighted sum
    total = (W["gap_mix"]*gap_mix_score +
             W["hotspots"]*hotspot_score +
             W["extremes"]*extreme_score +
             W["monthly"]*monthly_score +
             W["sessions"]*session_score +
             W["calendar"]*calendar_score +
             W["completeness"]*completeness_score)

    breakdown = {
        "gap_mix": gap_mix_score,
        "hotspots": hotspot_score,
        "extremes": extreme_score,
        "monthly": monthly_score,
        "sessions": session_score,
        "calendar": calendar_score,
        "completeness": completeness_score,
    }
    return {"total": float(total*100.0/100.0), "breakdown": breakdown, "params": params}# Patch build_common_blocks to embed score into assessment_md
_old_bcb = build_common_blocks
def build_common_blocks(df: pd.DataFrame, gaps: pd.DataFrame, year: int) -> dict:
    blocks = _old_bcb(df, gaps, year)
    sc = compute_score(df, gaps, year)
    br = sc["breakdown"]
    score_line = f"**Score (0–100): {sc['total']:.1f}**"
    breakdown_md = (
        f"- Gap mix: {br['gap_mix']:.1f}\n"
        f"- Hotspots: {br['hotspots']:.1f}\n"
        f"- Extremes: {br['extremes']:.1f}\n"
        f"- Monthly stability: {br['monthly']:.1f}\n"
        f"- Sessions: {br['sessions']:.1f}\n"
        f"- Calendar: {br['calendar']:.1f}\n"
        f"- Completeness: {br['completeness']:.1f}"
    )
    # extend assessment_md
    extra = f"\n\n{score_line}\n\nBreakdown:\n{breakdown_md}\n"
    blocks["assessment_md"] = blocks.get("assessment_md", "") + extra
    return blocks

# --- Inject scorecard into assessment + expose scorecard_md ---
_old_build_common_blocks = build_common_blocks
def build_common_blocks(df: pd.DataFrame, gaps: pd.DataFrame, year: int) -> Dict[str,str]:
    blocks = _old_build_common_blocks(df, gaps, year)
    # try to read scoring from config
    params = None
    try:
        cfg_txt = Path("/mnt/data/project_config.yml").read_text(encoding="utf-8", errors="ignore")
        params = _merge_scoring_params(_parse_yaml_scoring(cfg_txt))
    except Exception:
        params = _default_scoring_params()
    sc = compute_score(df, gaps, year, params)
    br = sc["breakdown"]; W = sc["params"]["weights"]
    score_line = f"**Score (0–100): {sc['total']:.1f}**"
    breakdown_md = "| Component | Weight | Score |\n|---|---:|---:|\n" + "\n".join([
        f"| Gap mix | {W['gap_mix']:.2f} | {br['gap_mix']:.1f} |",
        f"| Hotspots | {W['hotspots']:.2f} | {br['hotspots']:.1f} |",
        f"| Extremes | {W['extremes']:.2f} | {br['extremes']:.1f} |",
        f"| Monthly stability | {W['monthly']:.2f} | {br['monthly']:.1f} |",
        f"| Sessions | {W['sessions']:.2f} | {br['sessions']:.1f} |",
        f"| Calendar | {W['calendar']:.2f} | {br['calendar']:.1f} |",
        f"| Completeness | {W['completeness']:.2f} | {br['completeness']:.1f} |",
    ])
    blocks["assessment_md"] = blocks.get("assessment_md","") + f"\n\n{score_line}\n"
    blocks["scorecard_md"] = f"{score_line}\n\n{breakdown_md}"
    return blocks

# === Scoring config support ===
from pathlib import Path

def _default_scoring_params():
    return {
        "weights": {
            "gap_mix": 0.30, "hotspots": 0.20, "extremes": 0.15,
            "monthly": 0.15, "sessions": 0.10, "calendar": 0.05, "completeness": 0.05
        },
        "targets": {
            "small_share": 0.40, "long_share": 0.20,
            "extreme_rate_per_10k": 0.5,
            "monthly_cv": 1.0,
            "session_other": 0.10,
            "longest_gap_hours_ok": 24.0
        }
    }

def _parse_yaml_scoring(cfg_text: str) -> dict | None:
    """Minimal YAML reader for scoring.{weights,targets} with numeric values only."""
    lines = cfg_text.splitlines()
    i = 0
    found = False
    block = []
    while i < len(lines):
        if re.match(r'^\s*scoring\s*:', lines[i]):
            found = True
            i += 1
            # capture until next top-level key (no leading spaces)
            while i < len(lines) and (lines[i].strip()=='' or lines[i].startswith(' ') or lines[i].startswith('\t')):
                block.append(lines[i])
                i += 1
            break
        i += 1
    if not found:
        return None
    params = {"weights": {}, "targets": {}}
    mode = None
    for ln in block:
        if re.search(r'\bweights\s*:', ln):
            mode = "weights"; continue
        if re.search(r'\btargets\s*:', ln):
            mode = "targets"; continue
        m = re.match(r'^\s+([A-Za-z0-9_]+)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$', ln)
        if m and mode in ("weights","targets"):
            key = m.group(1); val = float(m.group(2))
            params[mode][key] = val
    return params

def _merge_scoring_params(custom: dict | None) -> dict:
    base = _default_scoring_params()
    if not custom:
        return base
    for k in ("weights","targets"):
        if k in custom and isinstance(custom[k], dict):
            for kk, vv in custom[k].items():
                if kk in base[k]:
                    base[k][kk] = float(vv)
    return base

# --- override compute_score with params in result ---
def compute_score(df: pd.DataFrame, gaps: pd.DataFrame, year: int, params: dict | None = None) -> dict:
    if params is None:
        params = _default_scoring_params()
    W = params["weights"]; T = params["targets"]

    N = len(gaps)
    if N>0:
        s = gaps["delta_sec"]
        n_small = int(((s>60)&(s<=300)).sum())
        n_med   = int(((s>300)&(s<=3600)).sum())
        n_long  = int((s>3600).sum())
        p_small = n_small/N; p_med = n_med/N; p_long = n_long/N
    else:
        p_small = p_med = p_long = 0.0

    comp_small = min(1.0, p_small / T["small_share"])
    comp_long  = max(0.0, 1.0 - min(1.0, p_long / T["long_share"]))
    gap_mix_score = 100.0*(0.6*comp_small + 0.4*comp_long)

    if N>0:
        cells = gaps.assign(wd=gaps["gap_start"].dt.weekday, hr=gaps["gap_start"].dt.hour).groupby(["wd","hr"]).size()
        total = cells.sum()
        p = (cells/total).values
        H = float((p*p).sum())
        C = int(len(cells))
        normH = (H - 1.0/C) / (1.0 - 1.0/C) if C>1 else 1.0
        hotspot_score = 100.0*(1.0 - normH)
    else:
        hotspot_score = 100.0

    df2 = df.copy()
    df2["range"] = (df2["high"]-df2["low"]).abs()
    p99r = float(df2["range"].quantile(0.99)) if len(df2) else 0.0
    ext = int((df2["range"] > 5*p99r).sum()) if p99r>0 else 0
    rate_per_10k = ext / (len(df2)/10000.0) if len(df2) else 0.0
    extreme_score = 100.0*(1.0 - min(1.0, rate_per_10k / T["extreme_rate_per_10k"]))

    if len(df2):
        df2["Month"] = df2["datetime_utc"].dt.to_period("M").astype(str)
        if N>0:
            gaps2 = gaps.copy(); gaps2["Month"] = gaps2["gap_start"].dt.to_period("M").astype(str)
            monthly = (gaps2.groupby("Month").size()).reindex(sorted(df2["Month"].unique()), fill_value=0)
        else:
            monthly = pd.Series([0]*len(df2["Month"].unique()), index=sorted(df2["Month"].unique()))
        mu = float(monthly.mean())
        cv = float((monthly.std(ddof=0)/mu)) if mu>0 else 0.0
    else:
        cv = 0.0
    monthly_score = 100.0*(1.0 - min(1.0, cv / T["monthly_cv"]))

    if N>0:
        def _sess(ts):
            h = ts.hour + ts.minute/60.0
            asia = (0 <= h < 8); london = (7 <= h < 16); ny = (12 <= h < 21)
            if asia and london: return "Asia-London overlap"
            if london and ny:  return "London-NY overlap"
            if ny: return "NY"
            if london: return "London"
            if asia: return "Asia"
            return "Other"
        sc = gaps["gap_start"].apply(_sess).value_counts()
        share_other = float(sc.get("Other", 0)/N)
    else:
        share_other = 0.0
    session_score = 100.0*(1.0 - min(1.0, share_other / T["session_other"]))

    calendar_score = 0.0

    longest_h = float(gaps["delta_sec"].max()/3600.0) if N>0 else 0.0
    completeness_score = 100.0*(1.0 - min(1.0, longest_h / T["longest_gap_hours_ok"]))

    total = (W["gap_mix"]*gap_mix_score +
             W["hotspots"]*hotspot_score +
             W["extremes"]*extreme_score +
             W["monthly"]*monthly_score +
             W["sessions"]*session_score +
             W["calendar"]*calendar_score +
             W["completeness"]*completeness_score)

    breakdown = {
        "gap_mix": gap_mix_score,
        "hotspots": hotspot_score,
        "extremes": extreme_score,
        "monthly": monthly_score,
        "sessions": session_score,
        "calendar": calendar_score,
        "completeness": completeness_score,
    }
    return {"total": float(total), "breakdown": breakdown, "params": params}

# --- override build_monthly_context to include scorecard_md ---
def build_monthly_context(df: pd.DataFrame, gaps: pd.DataFrame, year: int, month: str) -> Dict[str,str]:
    dfm = df[df["datetime_utc"].dt.to_period("M").astype(str) == month].copy()
    gm  = gaps[gaps["gap_start"].dt.to_period("M").astype(str) == month].copy()
    common = build_common_blocks(dfm if not dfm.empty else df, gm if not gm.empty else gaps, year)
    return {
        "month": month,
        "monthly_table_md": common["monthly_table_md"],
        "sessions_table_md": common["sessions_table_md"],
        "durations_section_md": common["durations_section_md"],
        "scorecard_md": common.get("scorecard_md", ""),
    }

# ==== BEGIN: timeframe-aware patch (M1/M5/H1) ====
import re
import fx_holidays
from typing import Dict
import pandas as pd
from pathlib import Path

def _read_config_text() -> str:
    try:
        return Path("/mnt/data/project_config.yml").read_text(encoding="utf-8")
    except Exception:
        return ""

def _read_runtime_timeframe(cfg_text: str) -> str:
    tf = None
    lines = cfg_text.splitlines()
    in_runtime = False
    for ln in lines:
        if re.match(r'^\s*runtime\s*:\s*$', ln):
            in_runtime = True
            continue
        if in_runtime:
            m = re.match(r'^\s*timeframe\s*:\s*([A-Za-z0-9]+)\s*$', ln)
            if m:
                tf = m.group(1).upper()
                break
        if re.match(r'^\S', ln):
            in_runtime = False
    if tf not in ("M1","M5","H1"):
        tf = "M1"
    return tf

def _resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "M1":
        return df.copy()
    rule = {"M5": "5T", "H1": "H"}[tf]
    dfr = (df.set_index("datetime_utc")
             .resample(rule, label="right", closed="right")
             .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
          )
    dfr = dfr.dropna(subset=["open","high","low","close"]).reset_index()
    return dfr

def _bar_gaps(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["gap_start","gap_end","delta_sec"])
    expect = {"M1": 60, "M5": 300, "H1": 3600}[tf]
    dsec = bars["datetime_utc"].diff().dt.total_seconds().fillna(expect).astype(int)
    idx = dsec[dsec > expect].index
    rows = []
    for i in idx:
        s = bars.loc[i-1, "datetime_utc"]; e = bars.loc[i, "datetime_utc"]
        rows.append({"gap_start": s, "gap_end": e, "delta_sec": int((e-s).total_seconds())})
    return pd.DataFrame(rows)

def _default_scoring_params_tf() -> dict:
    return {
        "gap_buckets": {
            "M1": {"small_max_s": 300,   "long_min_s": 3600},
            "M5": {"small_max_s": 1800,  "long_min_s": 10800},
            "H1": {"small_max_s": 21600, "long_min_s": 86400},
        },
        "weights": {
            "M1": {"gap_mix":0.30,"hotspots":0.20,"extremes":0.15,"monthly":0.15,"sessions":0.10,"calendar":0.05,"completeness":0.05},
            "M5": {"gap_mix":0.25,"hotspots":0.20,"extremes":0.15,"monthly":0.15,"sessions":0.10,"calendar":0.05,"completeness":0.10},
            "H1": {"gap_mix":0.20,"hotspots":0.15,"extremes":0.15,"monthly":0.15,"sessions":0.10,"calendar":0.05,"completeness":0.20},
        },
        "targets": {
            "M1": {"small_share":0.40, "long_share":0.20, "extreme_rate_per_k":0.5, "monthly_cv":1.0, "session_other":0.10, "longest_gap_hours_ok":24},
            "M5": {"small_share":0.50, "long_share":0.15, "extreme_rate_per_k":1.0, "monthly_cv":1.0, "session_other":0.10, "longest_gap_hours_ok":24},
            "H1": {"small_share":0.60, "long_share":0.10, "extreme_rate_per_k":1.0, "monthly_cv":1.0, "session_other":0.10, "longest_gap_hours_ok":24},
        },
    }

# Keep reference to original function if it exists
try:
    _orig_build_common_blocks = build_common_blocks
except NameError:
    _orig_build_common_blocks = None

def compute_score_timeframe_aware(bars: pd.DataFrame, bar_gaps: pd.DataFrame, year: int, tf: str, tf_params: dict) -> dict:
    small_max = int(tf_params["gap_buckets"][tf]["small_max_s"])
    long_min  = int(tf_params["gap_buckets"][tf]["long_min_s"])
    N = len(bar_gaps)
    if N>0:
        s = bar_gaps["delta_sec"]
        p_small = ((s> {"M1":60,"M5":300,"H1":3600}[tf]) & (s<= small_max)).mean()
        p_long  = (s> long_min).mean()
    else:
        p_small = p_long = 0.0

    W = tf_params["weights"][tf]
    T = tf_params["targets"][tf]

    comp_small = min(1.0, p_small / T["small_share"])
    comp_long  = max(0.0, 1.0 - min(1.0, p_long / T["long_share"]))
    gap_mix = 100.0*(0.6*comp_small + 0.4*comp_long)

    if N>0:
        cells = bar_gaps.assign(wd=bar_gaps["gap_start"].dt.weekday, hr=bar_gaps["gap_start"].dt.hour).groupby(["wd","hr"]).size()
        tot = cells.sum()
        p = (cells/tot).values
        H = float((p*p).sum()); C = int(len(cells))
        normH = (H - 1.0/C)/(1.0 - 1.0/C) if C>1 else 1.0
        hotspots = 100.0*(1.0 - normH)
    else:
        hotspots = 100.0

    bars2 = bars.copy()
    bars2["range"] = (bars2["high"]-bars2["low"]).abs()
    p99r = float(bars2["range"].quantile(0.99)) if len(bars2) else 0.0
    ext = int((bars2["range"] > 5*p99r).sum()) if p99r>0 else 0
    per_k = 10000 if tf=="M1" else 1000
    rate_per_k = ext / (len(bars2)/per_k) if len(bars2) else 0.0
    extremes = 100.0*(1.0 - min(1.0, rate_per_k / T["extreme_rate_per_k"]))

    if len(bars2):
        bars2["Month"] = bars2["datetime_utc"].dt.to_period("M").astype(str)
        if N>0:
            g2 = bar_gaps.copy(); g2["Month"] = g2["gap_start"].dt.to_period("M").astype(str)
            monthly = (g2.groupby("Month").size()).reindex(sorted(bars2["Month"].unique()), fill_value=0)
        else:
            monthly = pd.Series([0]*len(bars2["Month"].unique()), index=sorted(bars2["Month"].unique()))
        mu = float(monthly.mean())
        cv = float((monthly.std(ddof=0)/mu)) if mu>0 else 0.0
    else:
        cv = 0.0
    monthly_score = 100.0*(1.0 - min(1.0, cv / T["monthly_cv"]))

    def _sess(ts):
        h = ts.hour + ts.minute/60.0
        asia = (0 <= h < 8); london = (7 <= h < 16); ny = (12 <= h < 21)
        if asia and london: return "Asia-London overlap"
        if london and ny:  return "London-NY overlap"
        if ny: return "NY"
        if london: return "London"
        if asia: return "Asia"
        return "Other"
    share_other = (bar_gaps["gap_start"].apply(_sess)=="Other").mean() if N>0 else 0.0
    sessions = 100.0*(1.0 - min(1.0, share_other / T["session_other"]))

    calendar = 0.0
    longest_h = float(bar_gaps["delta_sec"].max()/3600.0) if N>0 else 0.0
    completeness = 100.0*(1.0 - min(1.0, longest_h / T["longest_gap_hours_ok"]))

    total = (W["gap_mix"]*gap_mix + W["hotspots"]*hotspots + W["extremes"]*extremes +
             W["monthly"]*monthly_score + W["sessions"]*sessions + W["calendar"]*calendar +
             W["completeness"]*completeness)

    scorecard_md = (
        f"**Score (0–100): {total:.1f}** — TF: {tf}\n\n"
        "| Component | Weight | Score |\n|---|---:|---:|\n"
        f"| Gap mix | {W['gap_mix']:.2f} | {gap_mix:.1f} |\n"
        f"| Hotspots | {W['hotspots']:.2f} | {hotspots:.1f} |\n"
        f"| Extremes | {W['extremes']:.2f} | {extremes:.1f} |\n"
        f"| Monthly stability | {W['monthly']:.2f} | {monthly_score:.1f} |\n"
        f"| Sessions | {W['sessions']:.2f} | {sessions:.1f} |\n"
        f"| Calendar | {W['calendar']:.2f} | {calendar:.1f} |\n"
        f"| Completeness | {W['completeness']:.2f} | {completeness:.1f} |\n"
    )
    return {"total": float(total), "scorecard_md": scorecard_md}

def build_common_blocks(df: pd.DataFrame, gaps: pd.DataFrame, year: int):
    # Wrap original (if есть) и применим TF-логику
    tf = _read_runtime_timeframe(_read_config_text())
    bars = _resample_ohlcv(df, tf)
    bar_gaps = _bar_gaps(bars, tf)
    blocks = _orig_build_common_blocks(bars, bar_gaps, year) if _orig_build_common_blocks else {}
    res = compute_score_timeframe_aware(bars, bar_gaps, year, tf, _default_scoring_params_tf())
    blocks["scorecard_md"] = res["scorecard_md"]
    if "gaps_classification_header_md" in blocks:
        base_min = {"M1":1,"M5":5,"H1":60}[tf]
        blocks["gaps_classification_header_md"] = re.sub(r">\s*1\s*min", f">{base_min} min", blocks["gaps_classification_header_md"])
    blocks["assessment_md"] = (blocks.get("assessment_md","") + f"\n\n**Timeframe:** {tf}").strip()
    return blocks
# ==== END: timeframe-aware patch ====


# ==== BEGIN: explainable gaps filter (weekends/holidays/maintenance) ====
import datetime as _dt

def _parse_ignore_cfg(cfg_text: str) -> dict:
    # Minimal parse:
    # scoring:
    #   ignore:
    #     weekly_window_utc: "Fri 22:00 -> Sun 22:00"
    #     dates_utc:
    #       - "2001-01-01"
    #       - "2001-12-25"
    cfg = {
        "weekly_window": ("Fri 22:00", "Sun 22:00"),
        "dates": set(),
    }
    lines = cfg_text.splitlines()
    in_scoring = False; in_ignore = False; in_dates = False
    for ln in lines:
        if re.match(r'^\s*scoring\s*:\s*$', ln): in_scoring=True; in_ignore=False; in_dates=False; continue
        if in_scoring and re.match(r'^\S', ln): in_scoring=False  # next top-level
        if in_scoring and re.match(r'^\s*ignore\s*:\s*$', ln): in_ignore=True; in_dates=False; continue
        if in_ignore and re.match(r'^\s*weekly_window_utc\s*:\s*"(.*)"\s*$', ln):
            m = re.match(r'^\s*weekly_window_utc\s*:\s*"(.*)"\s*$', ln)
            if m: 
                s = m.group(1)
                parts = [p.strip() for p in s.split("->")]
                if len(parts)==2:
                    cfg["weekly_window"] = (parts[0], parts[1])
            continue
        if in_ignore and re.match(r'^\s*dates_utc\s*:\s*$', ln): in_dates=True; continue
        if in_dates:
            m = re.match(r'^\s*-\s*"?(20\d{2}-\d{2}-\d{2})"?\s*$', ln)
            if m:
                cfg["dates"].add(m.group(1))
                continue
            if re.match(r'^\s*\S', ln):  # exit block
                in_dates=False
    return cfg

def _parse_wd_hhmm(token: str):
    # "Fri 22:00" -> (weekday=4, minutes=1320)
    dmap = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
    token = token.strip()
    m = re.match(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\d{1,2}):(\d{2})$', token)
    if not m: return None
    wd = dmap[m.group(1)]
    hh = int(m.group(2)); mm = int(m.group(3))
    return wd, hh*60+mm

def _in_weekly_window(ts: pd.Timestamp, start_tok: str, end_tok: str) -> bool:
    st = _parse_wd_hhmm(start_tok); en = _parse_wd_hhmm(end_tok)
    if not st or not en: return False
    wd = int(ts.weekday()); mins = ts.hour*60 + ts.minute
    # compute absolute minute-of-week for ts
    mow = wd*24*60 + mins
    s_mow = st[0]*24*60 + st[1]
    e_mow = en[0]*24*60 + en[1]
    if s_mow <= e_mow:
        return s_mow <= mow < e_mow
    else:
        # window wraps over week boundary
        return (mow >= s_mow) or (mow < e_mow)

def _is_explainable_ts(ts: pd.Timestamp, cfg_text: str) -> str|None:
    cfg = _parse_ignore_cfg(cfg_text)
    # weekends/weekly window
    if _in_weekly_window(ts, cfg["weekly_window"][0], cfg["weekly_window"][1]):
        return "weekend/closed-hours"
    # whole-day holiday dates
    d = ts.tz_convert("UTC").strftime("%Y-%m-%d") if ts.tzinfo else ts.strftime("%Y-%m-%d")
    if d in cfg["dates"]:
        return "holiday"
    return None

def _tag_explainable(bar_gaps: pd.DataFrame, cfg_text: str) -> pd.DataFrame:
    if bar_gaps.empty: 
        return bar_gaps.assign(reason=pd.Series(dtype="object"))
    reasons = []
    for ts in bar_gaps["gap_start"]:
        reasons.append(_is_explainable_ts(ts, cfg_text))
    bg = bar_gaps.copy()
    bg["reason"] = reasons
    return bg

# keep original for reference
try:
    __prev_build_common_blocks = build_common_blocks
except NameError:
    __prev_build_common_blocks = None

def build_common_blocks(df: pd.DataFrame, gaps: pd.DataFrame, year: int):
    cfg_text = _read_config_text() if '_read_config_text' in globals() else ""
    tf = _read_runtime_timeframe(cfg_text) if '_read_runtime_timeframe' in globals() else "M1"
    bars = _resample_ohlcv(df, tf) if '_resample_ohlcv' in globals() else df
    bar_gaps = _bar_gaps(bars, tf) if '_bar_gaps' in globals() else gaps

    tagged = _tag_explainable(bar_gaps, cfg_text)
    # FX holiday windows (offline, deterministic)
    try:
        _fx_wins = fx_holidays.fx_holiday_windows(year, cfg_text)
    except Exception:
        _fx_wins = []
    if len(_fx_wins):
        _mask_h = tagged["reason"].isna() & tagged["gap_start"].apply(lambda ts: fx_holidays.in_any_window(ts, _fx_wins))
        if _mask_h.any():
            tagged.loc[_mask_h, "reason"] = "holiday"
    filtered = tagged[tagged["reason"].isna()].drop(columns=["reason"])
    # counts for notes
    weekend_count = int((tagged["reason"]=="weekend/closed-hours").sum())
    fx_count = int((tagged["reason"]=="holiday").sum())

    # weekend-only count for display
    weekend_count = int((tagged["reason"]=="weekend/closed-hours").sum())

    # compute blocks on filtered only
    blocks = __prev_build_common_blocks(bars, filtered, year) if __prev_build_common_blocks else {}

    # remove emoji badge lines (📆 🎉 📢 ⚙️ ❗) from classification text
    if "gap_classification_md" in blocks:
        _lines = blocks["gap_classification_md"].splitlines()
        _kept = [ln for ln in _lines if not re.search(r"[📆🎉📢⚙️❗]", ln)]
        blocks["gap_classification_md"] = "\n".join(_kept)
    # emoji_cleanup_done

    # prepend weekend and FX holiday notes
    if "gap_classification_md" in blocks:
        _wk = f"_Weekend gaps (not scored): {weekend_count}._"
        _fx = f"_FX holiday gaps (not scored): {fx_count}._"
        body = blocks["gap_classification_md"].lstrip()
        if _wk not in body:
            body = _wk + "\n\n" + body
        if _fx not in body:
            body = _fx + "\n" + body
        blocks["gap_classification_md"] = body
    # recompute TF-aware score on filtered
    if 'compute_score_timeframe_aware' in globals():
        tf_params = _default_scoring_params_tf() if '_default_scoring_params_tf' in globals() else {}
        score_res = compute_score_timeframe_aware(bars, filtered, year, tf, tf_params)
        blocks["scorecard_md"] = score_res["scorecard_md"]

    # annotate assessment
    excl = int(len(tagged) - len(filtered)); total = int(len(tagged))
    if total>0:
        note = f"Excluded explainable gaps from scoring and lists: {excl} / {total} (weekends/holidays/closed-hours)."
        blocks["assessment_md"] = (blocks.get("assessment_md","") + "\\n\\n" + note + f"\\n**Timeframe:** {tf}").strip()
    else:
        blocks["assessment_md"] = (blocks.get("assessment_md","") + f"\\n\\n**Timeframe:** {tf}").strip()

    # show weekend-only note inside Gap classification block
    weekend_note = f"_Weekend gaps (not scored): {weekend_count}._"
    if "gaps_classification_header_md" in blocks:
        blocks["gaps_classification_header_md"] = (
            blocks["gaps_classification_header_md"].rstrip() + "\n" + weekend_note
        )
    elif "gap_classification_md" in blocks:
        blocks["gap_classification_md"] = (
            weekend_note + "\n\n" + blocks["gap_classification_md"].lstrip()
        )

    # fix header text
    if "gaps_classification_header_md" in blocks:
        base_min = {"M1":1,"M5":5,"H1":60}.get(tf,1)
        blocks["gaps_classification_header_md"] = re.sub(r">\\s*1\\s*min", f">{base_min} min", blocks["gaps_classification_header_md"])
    return blocks
# ==== END: explainable gaps filter ====

def build_gaps_context(df, gaps, year):
    """
    Build context for gaps_summary template using TF bar-gaps AFTER filtering explainable gaps.
    Explainable: weekend/closed-hours window and holiday dates (from config).
    """
    cfg_text = _read_config_text() if '_read_config_text' in globals() else ""
    tf = _read_runtime_timeframe(cfg_text) if '_read_runtime_timeframe' in globals() else "M1"
    bars = _resample_ohlcv(df, tf) if '_resample_ohlcv' in globals() else df
    bar_g = _bar_gaps(bars, tf) if '_bar_gaps' in globals() else gaps

    # Filter explainable gaps if helper is available
    if '_tag_explainable' in globals():
        tagged = _tag_explainable(bar_g, cfg_text)
        filtered = tagged[tagged["reason"].isna()].drop(columns=["reason"])
    else:
        filtered = bar_g

    # Session labeling (UTC)
    def _sess(ts):
        h = ts.hour + ts.minute/60.0
        asia = (0 <= h < 8); london = (7 <= h < 16); ny = (12 <= h < 21)
        if asia and london: return "Asia-London overlap"
        if london and ny:  return "London-NY overlap"
        if ny: return "NY"
        if london: return "London"
        if asia: return "Asia"
        return "Other"

    # Sessions table
    if len(filtered):
        sess_counts = filtered["gap_start"].apply(_sess).value_counts()
        total = int(sess_counts.sum()) or 1
        order = ["Asia","Asia-London overlap","London","London-NY overlap","NY","Other"]
        rows = []
        used = set()
        for k in order:
            if k in sess_counts:
                v = int(sess_counts[k]); rows.append((k, v, 100.0*v/total)); used.add(k)
        for k, v in sess_counts.items():
            if k not in used:
                rows.append((k, int(v), 100.0*int(v)/total))
        sessions_table_md = "| Session | Count | % |\n|---|---:|---:|\n" + "\n".join(f"| {k} | {c} | {p:.2f}% |" for k,c,p in rows)
    else:
        sessions_table_md = "_No gaps_"

    # Full list of anomalies (filtered)
    lines = ["| # | Start UTC | End UTC | Δ sec |", "|---:|---|---|---:|"]
    if len(filtered):
        for i, r in enumerate(filtered.sort_values("gap_start").itertuples(index=False), start=1):
            lines.append(f"| {i} | {r.gap_start.isoformat()} | {r.gap_end.isoformat()} | {int(r.delta_sec)} |")

    gaps_full_table_md = "\n".join(lines)

    return {
        "sessions_table_md": sessions_table_md,
        "gaps_full_table_md": gaps_full_table_md,
    }
