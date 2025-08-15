<h1 align="center">📅 Economic Calendar for EUR/USD Backtest</h1>
<p align="center"><strong>2001-01-01 → 2025-07-31</strong> · <em>24 years, 7 months</em> · <code>UTC+0</code></p>
<p align="center">
  <img alt="Coverage" src="https://img.shields.io/badge/Coverage-2001%E2%80%932025-blue">
  <img alt="Time zone" src="https://img.shields.io/badge/Time%20Zone-UTC%2B0-black">
  <img alt="Sources" src="https://img.shields.io/badge/Sources-Official%20only-brightgreen">
  <img alt="Use case" src="https://img.shields.io/badge/Use%20case-Backtesting-orange">
</p>

---

> A compact, source‑verified economic calendar tailored to the **EUR/USD** macro context.  
> Times are normalized to **UTC+0**, accounting for daylight saving/standard time transitions per original source.

## Table of Contents
<details>
<summary>Open</summary>

- [Overview](#overview)
- [Time Standardization](#time-standardization)
- [Purpose](#purpose)
- [Structure](#structure)
- [Data Coverage](#data-coverage)
- [Reports](#reports)
- [Countries](#countries)
- [News Scope (Author’s Filter)](#news-scope-authors-filter)
  - [United States](#-united-states-us)
  - [Euro Area](#-euro-area-eu)
  - [United Kingdom](#-united-kingdom-uk)
  - [Switzerland](#-switzerland-ch)
- [Author’s Adjustments to Importance](#-authors-adjustments-to-importance)
- [FAQ — Why are some countries/indicators missing?](#-faq--why-are-some-countriesindicators-missing)
- [Exclusions](#-exclusions)
- [Compilation & Transparency](#-compilation--transparency)
- [Process Integrity](#-process-integrity)
</details>

---

## Overview

The calendar is adapted to the macroeconomic context of the **EUR/USD** pair and covers **January 1, 2001** → **July 31, 2025**.  
The database is compiled from **official sources**.

---

## Time Standardization

All events are converted to **UTC+0**. Transitions between daylight saving and standard time are handled according to each **original source**, ensuring transparent and reproducible timestamps.

> **Note**  
> Converted timestamps are intended for backtests and data alignment; they remain consistent with official publication schedules.

---

## Purpose

The macro context (**news filtering**, **importance levels**, **country list**) was adapted by the author specifically for **strategy backtesting**.  
The strategy is primarily sensitive to **high‑importance** events; some **medium‑importance** events are included to support future session logic.

---

## Structure

The calendar is split by **years** (for **2025** — January–July as of the project start).  
Each `calendar_YYYY/` folder contains the main artifacts:

- **`.csv.gz` archive**  
- **CSV** (extracted/plain)  
- **SHA‑256 hashes** for `.csv.gz` and `.csv`  
- **GPG signatures** of the SHA‑256 files

```text
economic_calendar/
├─ bundles_2001-2025/        # step-by-step build artifacts
├─ calendar_2001/
│  ├─ calendar_2001.csv.gz
│  ├─ calendar_2001.csv
│  ├─ csv(gz).sha256
│  └─ csv(gz).sha256.asc     # GPG signature
├─ calendar_2002/
│  └─ ...
└─ calendar_2025/            # Jan–Jul 2025
```

Annual CSVs include the key fields for backtests — **UTC date‑time** and **importance** — for windowing logic and **EUR/USD minute‑data** analysis. The strategy does **not** trade at the release instant; therefore indicator values (previous/actual) are out of scope.

> **Tip**  
> To verify integrity, compare annual `.csv.gz` files and extracted `.csv` files against the published **SHA‑256** hashes.

---

## Data Coverage

The dataset contains all necessary **important** macroeconomic releases for **historical EUR/USD minute‑data analysis** and **backtests**.

---

## Reports

- **Authenticity** — determined by whether the URL belongs to an **official domain**.  
- **Backtest suitability** — a function of (1) source officialness and (2) availability of an **exact publication time**.

These grades are **auxiliary** for developing the **calendar builder** and are **not final**. The focus is on extracting **exact publication times** from official sources; some links may be generic or blank to speed up compilation.

> **Important**  
> Even if an event time in the compiled CSV differs from an official archive, this is **not critical** for the strategy, which does **not** trade at the release moment. High‑importance events define **observation windows**, not entry timestamps.

**All events were gathered from official sources.**

---

## Countries

**United States (US)** · **Euro Area (EU)** · **United Kingdom (UK)** · **Switzerland (CH)**

---

## News Scope (Author’s Filter)

### 🇺🇸 United States (US)

- Fed rate decision; press conferences; unscheduled events  
- FOMC statements and minutes  
- NFP  
- ISM PMI  
- PPI  
- GDP  
- CPI  
- PCE  
- Retail Sales

### 🇪🇺 Euro Area (EU)

- ECB rate decision; press conferences; unscheduled events  
- Unemployment Rate  
- HICP (final)  
- HICP (flash, since 2002)  
- GDP (final)  
- GDP (flash, since 2009)  
- **All PMI: final; flash (since 2014)**

### 🇬🇧 United Kingdom (UK)

- Bank of England rate decision; press conferences; unscheduled events  
- **All PMI: final; flash (since 2021)**

### 🇨🇭 Switzerland (CH)

- SNB rate decision; press conferences; unscheduled events

---

## ✏️ Author’s Adjustments to Importance

- **FOMC statements and minutes** — always **medium** (exception: **unscheduled** releases). Other **US economic events** — **high**.  
- **Euro Area** — **all PMI** → **medium**; other EU economic events — **high**.  
- **UK** and **CH** — all listed events → **medium**.

---

## ❓ FAQ — Why are some countries/indicators missing?

During builder development, all EU countries were left as options for potential future use.

The calendar is tuned to the **strategy’s requirements**: it does not execute during the narrow European session windows when most releases occur, and implementing that efficiently in code is impractical. Major **Euro Area** releases are included as exceptions. **PMIs** (EU, UK) are collected as **control points** for future use. Overall, these additional releases are **not required** for the current strategy.

---

## ⚠️ Exclusions

Unscheduled politico‑economic headlines are **not included**, since their exact publication times are not predictable.  
The strategy already includes minimal, code‑feasible logic to react to sudden news.

---

## 🛠 Compilation & Transparency

The calendar was compiled using **[…/core_builder](https://github.com/rleydev/euro-macromechanica-backtest/tree/master/economic_calendar/calendar_builder)** and **ChatGPT‑5 (Thinking & Pro modes)** to improve efficiency. The builder implements the core data‑gathering logic; some parts are intentionally left light to keep the research‑mode environment flexible without unnecessary code.

Use the builder for your needs or to verify the transparency of this calendar, following the sections **Countries**, **News Scope**, and **Author’s Adjustments to Importance**.

---

## 🔒 Process Integrity

Integrity is ensured via **SHA‑256** hashes of final files, **GPG signatures**, and **Git history**, as well as **OTS anchoring** of the archive of all CSVs from 2001 through the end of July 2025. Taken together, this guarantees the calendar has **not** been curve‑fitted to backtest results.
