# helpers.py — unified utilities for deterministic artifacts & report rendering
# - Deterministic gzip writer (mtime=0, controllable compresslevel)
# - Deterministic PNG saver (strips tIME/iTXt/tEXt/zTXt, fixed DPI/bbox/padding)
# - Strict Matplotlib font setup (Agg + DejaVu Sans)
# - Content-based analysis timestamp from CSVs
# - Simple {{key}} template renderer
# - Deprecated mtime-based analysis stamp (kept as *_mtime alias); stub on old name

from __future__ import annotations
import os, io, gzip, hashlib, re
from typing import List, Dict

__all__ = [
    "write_gzip_deterministic",
    "save_png_deterministic",
    "set_matplotlib_strict_font",
    "compute_analysis_utc_ts_from_csvs",
    "compute_analysis_utc_ts_mtime",
    "compute_analysis_utc_ts",
    "render_template_file",
    "quarter_bounds",
    "month_bounds",
]
# --------------------------
# Deterministic GZIP writer
# --------------------------
def write_gzip_deterministic(src_path: str, dst_path: str, compresslevel: int = 6) -> None:
    """Write gz with mtime=0 for stable SHA-256 across runs."""
    with open(src_path, "rb") as f_in:
        raw = f_in.read()
    # Note: pass compresslevel explicitly (was ignored before)
    with gzip.GzipFile(
        filename=os.path.basename(src_path),
        mode="wb",
        fileobj=open(dst_path, "wb"),
        mtime=0,
        compresslevel=compresslevel,
    ) as gz:
        gz.write(raw)

# --------------------------
# Deterministic PNG saver
# --------------------------
PNG_SIG = b"\x89PNG\r\n\x1a\n"

def _strip_png_variable_chunks(png_bytes: bytes) -> bytes:
    """
    Remove variable chunks (tIME, iTXt, tEXt, zTXt) from a PNG to improve determinism.
    Keeps other chunks intact (including CRCs).
    """
    if not png_bytes.startswith(PNG_SIG):
        raise ValueError("Not a PNG")
    out = bytearray()
    out += PNG_SIG
    i = len(PNG_SIG)
    n = len(png_bytes)
    while i + 8 <= n:
        length = int.from_bytes(png_bytes[i:i+4], "big")
        ctype = png_bytes[i+4:i+8]
        data_start = i + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > n:
            break  # corrupted or truncated
        chunk = png_bytes[i:crc_end]
        # Skip variable chunks
        if ctype not in (b"tIME", b"tEXt", b"zTXt", b"iTXt"):
            out += chunk
        i = crc_end
    return bytes(out)

def save_png_deterministic(fig, path: str, dpi: int = 120) -> str:
    """
    Save matplotlib figure to PNG deterministically (fixed DPI/bbox/pad and no tIME/text chunks).
    Returns SHA-256 hex of the saved file.
    """
    # Lazy import to avoid hard dependency at import time
    import matplotlib.pyplot as plt  # noqa: F401
    buf = io.BytesIO()
    # Save with fixed DPI and without extra metadata
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    raw = buf.getvalue()
    cleaned = _strip_png_variable_chunks(raw)
    with open(path, "wb") as f:
        f.write(cleaned)
    h = hashlib.sha256(); h.update(cleaned); return h.hexdigest()

# -----------------------------------------
# Strict font setup for Matplotlib (deterministic)
# -----------------------------------------
def set_matplotlib_strict_font(dpi: int = 120):
    """
    Enable Agg backend, lock DejaVu Sans family, fix DPI and savefig parameters.
    Must be called before creating/saving figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import rcParams
    rcParams["font.family"] = "DejaVu Sans"
    rcParams["font.sans-serif"] = ["DejaVu Sans"]
    rcParams["mathtext.fontset"] = "dejavusans"
    rcParams["figure.dpi"] = dpi
    rcParams["savefig.dpi"] = dpi
    rcParams["savefig.bbox"] = "tight"
    rcParams["savefig.pad_inches"] = 0

# ----------------------------------------------------
# Analysis timestamp (content-based and mtime fallback)
# ----------------------------------------------------
def compute_analysis_utc_ts_from_csvs(csv_paths: List[str], datetime_col: str = "datetime_utc", tail_lines: int = 2000) -> str:
    """
    Return ISO timestamp (UTC) equal to max(datetime_utc) across given CSV files.
    Reads only the tail of each file for performance.
    """
    import pandas as pd
    from io import StringIO
    import datetime as _dt

    def _tail_text(path: str, n: int = 2000) -> str:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            block = 4096
            data = b""
            while len(data.splitlines()) <= n and f.tell() > 0:
                step = min(block, f.tell())
                f.seek(-step, os.SEEK_CUR)
                chunk = f.read(step)
                f.seek(-step, os.SEEK_CUR)
                data = chunk + data
                if f.tell() == 0:
                    break
        return data.decode("utf-8", errors="ignore")

    max_ts = None
    for p in csv_paths:
        if not (isinstance(p, str) and os.path.isfile(p) and p.lower().endswith(".csv")):
            continue
        try:
            txt = _tail_text(p, n=tail_lines)
            df_tail = pd.read_csv(StringIO(txt))
            if datetime_col in df_tail.columns:
                s = pd.to_datetime(df_tail[datetime_col], utc=True, errors="coerce")
                if len(s):
                    ts = s.max()
                    if pd.notna(ts):
                        if (max_ts is None) or (ts > max_ts):
                            max_ts = ts
        except Exception:
            continue

    if max_ts is None:
        return _dt.datetime(1970,1,1,tzinfo=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    return max_ts.strftime("%Y-%m-%d %H:%M:%S %Z")

def compute_analysis_utc_ts_mtime(paths: List[str]) -> str:
    """Legacy: ISO timestamp (UTC) from max mtime of given paths (files only)."""
    import datetime as _dt
    mtimes = []
    for p in paths:
        if isinstance(p, str) and os.path.isfile(p):
            mtimes.append(os.path.getmtime(p))
    if not mtimes:
        return _dt.datetime(1970,1,1,tzinfo=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    mx = max(mtimes)
    dt = _dt.datetime.fromtimestamp(mx, tz=_dt.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def compute_analysis_utc_ts(*args, **kwargs):
    """
    DEPRECATED: не используйте mtime-метку.
    Вместо этого вызывайте compute_analysis_utc_ts_from_csvs(csv_paths, datetime_col="datetime_utc").
    Эта функция отключена намеренно и всегда бросает исключение, чтобы её не использовали по ошибке.
    """
    raise RuntimeError("Deprecated: use compute_analysis_utc_ts_from_csvs(...) (content-based analysis stamp).")

# --------------------------
# Simple template renderer
# --------------------------
def render_template_file(template_path: str, output_path: str, context: Dict[str, str]) -> None:
    """Very simple {{key}} replacement renderer for Markdown templates."""
    with open(template_path, "r", encoding="utf-8") as f:
        txt = f.read()
    for k, v in context.items():
        txt = txt.replace("{{"+k+"}}", str(v))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(txt)


# -----------------------------------------
# Strict period bounds (UTC) for slicing
# -----------------------------------------
def quarter_bounds(year: int, q: int):
    """
    Возвращает (start_utc, end_utc) квартала в UTC с семантикой интервала [start, end).
    Q ∈ {1,2,3,4}. Пример: 2021,Q3 → [2021-07-01 00:00:00Z, 2021-10-01 00:00:00Z)
    """
    assert q in (1,2,3,4), "q must be 1..4"
    from datetime import datetime, timezone
    starts = {
        1: datetime(year, 1, 1, tzinfo=timezone.utc),
        2: datetime(year, 4, 1, tzinfo=timezone.utc),
        3: datetime(year, 7, 1, tzinfo=timezone.utc),
        4: datetime(year,10, 1, tzinfo=timezone.utc),
    }
    start = starts[q]
    end = starts[q+1] if q < 4 else datetime(year+1, 1, 1, tzinfo=timezone.utc)
    return start, end

def month_bounds(year: int, month: int):
    """
    Возвращает (start_utc, end_utc) месяца в UTC с семантикой интервала [start, end).
    Пример: 2025,7 → [2025-07-01 00:00:00Z, 2025-08-01 00:00:00Z)
    """
    from datetime import datetime, timezone
    if not (1 <= month <= 12):
        raise ValueError("month must be 1..12")
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year+1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month+1, 1, tzinfo=timezone.utc)
    return start, end
