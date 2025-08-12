
# Economic Calendar 2001–2025 (until July 31, 2025)

The project compiles an economic calendar by year (2001–2025, for 2025 — until July 31) from official sources and prioritized providers.  
The output format is a compact `csv.gz` file with UTC time.

---

## Coverage

- **Countries and regions:** United States (US), Euro Area/EA (incl. key countries DE/FR/IT/ES), United Kingdom (UK), Switzerland (CH).
- **Event importance:** only `medium` and `high` are included. The importance filter is strictly enforced during validation.
- **Special importance rules:**
  - **US:**
    - FOMC unscheduled — `high`; scheduled — `medium`.
    - FOMC minutes and statements — always `medium`.
    - ISM PMI (Manufacturing/Services/Composite) — always `high`.
  - **ECB:** rate decisions — `high`; post-decision press conference — `high`. Any unscheduled events — `high`.
  - **Euro Area:** Unemployment Rate — `high`.
  - **Germany:** Flash CPI — `medium`.
  - **UK and CH:** all monetary policy events — `medium`.

---

## Sources and Provider Priority

Priority when collecting (`--providers all`):

`bls, bea, eurostat, ecb, ons, destatis, insee, istat, ine, ism, spglobal, snb, fso_seco, kof, procure_ch, census, fed_ip, confboard, umich, fomc`

**Note:** some providers may be integrated as “stubs” (interface ready, population in stages). In any case, priority and structure are already built into the pipeline.

---

## Data Formats

### Input CSV (`manual_events.csv`)
**Minimum columns** (order not critical):
- `date_local` — YYYY-MM-DD (local release date)
- `time_local` — HH:MM (local time; may be empty)
- `tz` — IANA timezone (e.g., `America/New_York`, `Europe/Zurich`); if empty — apply source-specific rules or UTC.
- `country` — US, EA, DE, FR, IT, ES, UK, CH
- `importance` — `medium` or `high`
- `title` — event name
- `source_url` — link to primary source (preferably domain from `official_domains` in `config.yaml`)
- Optional: `ticker`, `notes`, `certainty`

**Optional fields:**
- `certainty`:
  - `estimated` — no exact time from the primary source, time set via rule/heuristic
  - `secondary` — exact time taken from Reuters/Bloomberg (primary site did not provide)
  - Empty = confirmed time from the primary source.
- `notes` may contain manual importance overrides: `impact_override=high|medium`

---

### Output Calendar (`calendar_<year>.csv.gz`)
**Required columns:**
- `datetime_utc`, `event`, `country`, `impact`

**Optional:**  
`certainty`, `ticker`, `source_url`, `notes`

Format — `csv.gz` (gzip-compressed CSV).  
Most analytical tools read `*.csv.gz` directly.  
For Excel viewing, decompress:
- **macOS/Linux:** `gunzip -c calendar_2001.csv.gz > calendar_2001.csv`
- **Windows (PowerShell):** `tar -xzf .\calendar_2001.csv.gz`

---

## UTC Time Conversion

- Uses IANA `zoneinfo` with DST and “spring forward” gaps handling.
- Handles `fold=0/1` and non-existent local times (DST shift days) by adjusting to the nearest valid moment.
- If `tz` is missing and no rule applies — default to UTC.
- If time is from Reuters/Bloomberg — mark `certainty=secondary`.  
- If time set via rules — `certainty=estimated`.

---

## Metrics in Report

**Backtest Suitability** = weighted sum:
- **Authenticity** — share of records with official domains/primary sources
- **Timing** — share of records with confirmed primary time (excluding `estimated`/`secondary`)

Weights are from `config.yaml` → `weights` (default 0.95 / 0.05).  
Coverage is not used.

---

## Hashes, Manifest, and Bundles

- `manifest_<year>.json` — SHA-256 of all artifacts for the year (calendar, report, `state.json`, `config.yaml`)
- `bundle_<year>.tar.gz` — full year snapshot (calendar, report, manifest, `state.json`, `config.yaml`)
- `state.json` — pipeline state: year artifacts, `updated_at` and input signatures:
  - `inputs.year_slice_sha256` — SHA-256 of parsed CSV slice for the year (filtered by `date_local`), stable to column order.
  - `inputs.config_sha256` — SHA-256 of `config.yaml` content.

**Check SHA-256 locally:**
- **Windows (CMD):** `certutil -hashfile file.ext SHA256`
- **Windows (PowerShell):** `Get-FileHash .\file.ext -Algorithm SHA256`
- **macOS/Linux:** `shasum -a 256 file.ext` or `sha256sum file.ext`

---

## Commands

### Quick start
```bash
python -m venv .venv
. .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Yearly run
```bash
# Dry run (no files written)
python core.py assemble --year 2001 --providers all --dry-run

# Full run for the year
python core.py run --year 2001 --bundle
```

**Separate stages:**
```bash
python core.py validate --year 2001 --infile manual_events.csv
python core.py build    --year 2001 --infile manual_events.csv --outfile calendar_2001.csv.gz
python core.py report   --year 2001 --calendar calendar_2001.csv.gz
python core.py bundle   --year 2001
```

**Parameters:**
- `--providers` — choose subset of sources (`all` = by priority above)
- `--cache-dir` — cache directory for providers
- `--dry-run` — in `assemble`: collect everything in memory and show summary without writing

---

## Validation Rules

- Only `importance ∈ {medium, high}` (drop all `low`)
- Respect manual overrides in `notes`: `impact_override=...`
- If exact time from primary source is unavailable — `certainty=estimated`
- If time is from Reuters/Bloomberg — `certainty=secondary`

---

## Resume After Interruption

- After each key stage, `state.json` is updated.
- Manifest and report recalculated together to avoid desync.
- To resume in a new session, just load `bundle_<year>.tar.gz` (or artifacts separately) — the pipeline will skip completed steps by hashes.

---

## Configuration (`config.yaml`)

Minimal valid config:
```yaml
official_domains:
  domains:
    - federalreserve.gov
    - ecb.europa.eu
    - bls.gov
    - bea.gov
    - eurostat.ec.europa.eu
    - ons.gov.uk
    - bankofengland.co.uk
    - snb.ch
    - destatis.de
    - insee.fr
    - istat.it
    - ine.es
    - ismworld.org
    - spglobal.com
weights:
  authenticity: 0.95
  timing: 0.05
time_rules: {}
```

---

## Environment & Reproducibility

- Python 3.11 (or `python:3.11-slim` container)
- Packages pinned in `requirements.txt` (incl. `tzdata`)
- Dockerfile for fully reproducible environment

---

## FAQ

**Why `csv.gz`?**  
Smaller, faster to transfer, natively read by pandas/R/CLI tools.

**Can I load only `calendar_<year>.csv.gz` and continue?**  
Yes — for reporting/merging years.  
For re-validation/filling, better to have `manual_events.csv` and `config.yaml` (or `bundle_<year>.tar.gz`).

**What about 2025?**  
Covers Jan 1 — Jul 31, 2025.

---

## Timezones and Aliases
In `config.yaml` you can set `tz_aliases` (e.g., `ET → America/New_York`, `CET → Europe/Berlin`).  
If `tz` is missing/invalid, UTC is used with a warning to `stderr`.

---

## Robust CSV Handling

- Auto-detect encoding: try `utf-8`, `utf-8-sig`, `cp1251`, `latin-1`
- Normalize headers: lowercase + aliases (`date→date_local`, `time→time_local`, `timezone→tz`, `event→title`, `impact→importance`, `url/source→source_url`, etc.)
- Validate required columns/rows; drop invalid date or `importance∉{medium,high}`
- Output: `validated_<year>.csv` (filtered snapshot) and `validation_report_<year>.json` (summary)
- Won’t crash on “broken” CSV — invalid rows are dropped, rest is processed

---

## Stable Builds (Stable Hashes)

- `calendar_<year>.csv.gz` written with fixed `mtime=0` in gzip header → identical SHA-256 for same content
- `bundle_<year>.tar.gz` created with normalized metadata (`uid/gid=0`, empty `uname/gname`, `mtime=0`) → stable archive hash

---

## Year Filtering

In `validate` and `build`, a strict filter is applied on `date_local` for the given `--year`.  
Extra years are dropped and recorded in `validation_report_<year>.json` (`other_years`).

---

## Policy Updates

- **Authenticity** = official source:  
  events with `certainty=estimated` from official domains are considered **official** same as `confirmed`;  
  `secondary` = Reuters/Bloomberg, etc.
- **Backtest suitability** uses weights from `config.yaml` (`weights.authenticity_weight`, `weights.timing_weight`). Default: **0.95** and **0.05**.
- **Exclusions** from `config.yaml` are applied at `validate` and `build` stages.  
  Events matching `titles_exact` or `weekly_series` are automatically excluded.

---

## Official Source Logic & Auto-Confirm

**Official source** = union of:
- `official_domains` — explicit list (stat agencies, etc.)
- `gov_like_patterns` — patterns for central bank domains (Fed/FRB, ECB, Bundesbank, Banque de France, Banca d’Italia, Banco de España, SNB, BoE)

**Report rules (Authenticity):**
- `secondary` — unofficial
- `confirmed` — official
- `estimated` or empty — official **only if** domain ∈ (`official_domains` ∪ `gov_like_patterns`)

**Promotion to `confirmed` in build stage:**
- If domain ∈ (`official_domains` ∪ `gov_like_patterns`) **and** valid timezone **and** exact `time_local` — then `'' | estimated → confirmed`
- If timezone invalid/missing — marked `estimated` and `tz_fallback=utc` is added to `notes` (no promotion)
