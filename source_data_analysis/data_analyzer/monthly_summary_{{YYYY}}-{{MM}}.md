# EURUSD 1m — Monthly Summary {{YYYY}}-{{MM}}

**Input files (names + SHA-256):**
{{inputs_table_md}}

---

## 1) Gap classification (>1 min)
Total: {{gaps_total}}; 📆 {{wk_cnt}}, 🎉 {{hol_cnt}}, 📢 {{news_cnt}}, ⚙️ {{tech_cnt}}, ❗ {{anom_cnt}}.  
Economic calendar coverage: {{te_coverage}} (High={{te_high}}).

## 2) Durations (micro/medium/large)
Distribution + comments.

## 3) Sessions (UTC)
Asia/London/NY.

## 4) Extreme candles (TOP-3)
Top-3 by range (timestamp, range, direction).  
Mark possible data-glitches, if any.

## 5) PNG with ❗ anomalies
File: `EURUSD_{{YYYY}}-{{MM}}_anomalies.png` (red markers).

---

## Transparency footer
Deterministic analysis stamp (UTC): `{{analysis_utc_ts}}` (content-based: max(datetime_utc) from input CSVs).  
Hashes, environment, and authorship — see annual template for the exact fields.
