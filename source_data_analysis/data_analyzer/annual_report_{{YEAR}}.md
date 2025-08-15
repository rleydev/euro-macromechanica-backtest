# EURUSD 1m — Annual Report {{YEAR}}

**Input files (names + SHA-256):**
{{inputs_table_md}}

---

## 1) Gap classification (>1 min)
Total: {{gaps_total}}; 📆 {{wk_cnt}}, 🎉 {{hol_cnt}}, 📢 {{news_cnt}}, ⚙️ {{tech_cnt}}, ❗ {{anom_cnt}}.  
Economic calendar coverage: {{te_coverage}} (High={{te_high}}).

## 2) Durations (micro/medium/large)
Distribution table and comments on tails.

## 3) Sessions (UTC)
Distribution across Asia/London/NY and their overlaps.

## 4) Monthly statistics
Tables/plots by month.

## 5) Extreme candles
TOP-5 by range (timestamp, range, direction).  
Mark possible data-glitches, if any.

## 6) Cross-check with CME/EBS maintenance windows
Matches/exceptions, conclusions; weekly repeatability.

## 7) Visualizations
Gap heatmap; monthly counts chart.

## 8) Final assessment of the year
Backtest suitability, risks/quirks.

## 9) Full list of ❗ anomalies
Timestamps and durations (complete list).

## 10) PNG with ❗ anomalies
File: `EURUSD_{{YEAR}}_anomalies.png` (red markers).

---

## Transparency footer
- Goal: make the backtest dataset as transparent as possible.  
- Deterministic analysis stamp (UTC): `{{analysis_utc_ts}}` (content-based: max(datetime_utc) from input CSVs).  
- Algorithm ID/version, language and libraries: {{algo_id}} / {{algo_version}}; Python + pandas + matplotlib.  
- Defaults/config and its hash: {{config_hash}}.  
- Economic calendar coverage (year): {{te_coverage}} (High={{te_high}}).  
- SHA-256 of all inputs/outputs/artifacts and the MANIFEST HASH: {{artifacts_hash_summary}}.  
- Run environment (UTC timestamp): {{run_env_info}}.  
- Authorship note: “Report/analysis prepared with the assistance of ChatGPT ({{model_name}})”.  
- References/IDs and SHA-256 of the “full logic code” and “logic description” (if provided).
