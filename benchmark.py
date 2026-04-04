"""
Benchmark comparing Polars vs Pandas performance for SBDF import and export.

Addresses the copy-performance concerns raised in PR #99.

Usage:
    python benchmark.py
"""

import datetime
import gc
import os
import sys
import tempfile
import time
import warnings

import psutil
import numpy as np
import pandas as pd
import polars as pl

import spotfire.sbdf as sbdf

REPS = 7
SIZES = [10_000, 100_000]

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def make_polars(size, profile):
    if profile == "numeric":
        return pl.DataFrame({
            "b": pl.Series(RNG.integers(0, 2, size).astype(bool)),
            "i": pl.Series(RNG.integers(0, 1_000_000, size, dtype=np.int64)),
            "f": pl.Series(RNG.random(size)),
        })
    if profile == "numeric_nulls":
        mask = RNG.random(size) < 0.1
        ints = RNG.integers(0, 1_000_000, size, dtype=np.int64).tolist()
        for idx in np.where(mask)[0]:
            ints[idx] = None
        floats = RNG.random(size).tolist()
        for idx in np.where(mask)[0]:
            floats[idx] = None
        return pl.DataFrame({
            "i": pl.Series(ints, dtype=pl.Int64),
            "f": pl.Series(floats, dtype=pl.Float64),
        })
    if profile == "string":
        words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        return pl.DataFrame({
            "s": pl.Series([words[i % len(words)] for i in range(size)]),
        })
    if profile == "string_nulls":
        words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        vals = [words[i % len(words)] if RNG.random() > 0.1 else None for i in range(size)]
        return pl.DataFrame({"s": pl.Series(vals, dtype=pl.Utf8)})
    if profile == "temporal":
        base = datetime.datetime(2000, 1, 1)
        dts = [base + datetime.timedelta(seconds=int(x)) for x in RNG.integers(0, 86400 * 365 * 20, size)]
        return pl.DataFrame({
            "dt": pl.Series(dts, dtype=pl.Datetime),
            "d":  pl.Series([d.date() for d in dts], dtype=pl.Date),
            "td": pl.Series([datetime.timedelta(seconds=int(x)) for x in RNG.integers(0, 86400, size)],
                            dtype=pl.Duration),
            "t":  pl.Series([datetime.time(h, m, s)
                              for h, m, s in zip(
                                  RNG.integers(0, 24, size),
                                  RNG.integers(0, 60, size),
                                  RNG.integers(0, 60, size))],
                             dtype=pl.Time),
        })
    if profile == "temporal_nulls":
        base = datetime.datetime(2000, 1, 1)
        mask = RNG.random(size) < 0.1
        dts = [base + datetime.timedelta(seconds=int(x)) for x in RNG.integers(0, 86400 * 365 * 20, size)]
        dts_n = [None if mask[i] else dts[i] for i in range(size)]
        dates_n = [None if mask[i] else dts[i].date() for i in range(size)]
        tds_n = [None if mask[i] else datetime.timedelta(seconds=int(x))
                 for i, x in enumerate(RNG.integers(0, 86400, size))]
        times_n = [None if mask[i] else datetime.time(int(h), int(m), int(s))
                   for i, (h, m, s) in enumerate(zip(RNG.integers(0, 24, size),
                                                     RNG.integers(0, 60, size),
                                                     RNG.integers(0, 60, size)))]
        return pl.DataFrame({
            "dt": pl.Series(dts_n, dtype=pl.Datetime),
            "d":  pl.Series(dates_n, dtype=pl.Date),
            "td": pl.Series(tds_n, dtype=pl.Duration),
            "t":  pl.Series(times_n, dtype=pl.Time),
        })
    if profile == "binary":
        blobs = [bytes(RNG.integers(0, 256, 64, dtype=np.uint8)) for _ in range(size)]
        return pl.DataFrame({"b": pl.Series(blobs, dtype=pl.Binary)})
    if profile == "binary_nulls":
        blobs = [None if RNG.random() < 0.1 else bytes(RNG.integers(0, 256, 64, dtype=np.uint8))
                 for _ in range(size)]
        return pl.DataFrame({"b": pl.Series(blobs, dtype=pl.Binary)})
    raise ValueError(profile)


def make_pandas(polars_df):
    return polars_df.to_pandas()


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

_proc = psutil.Process(os.getpid())

def bench(fn, reps=REPS):
    """Return (mean_ms, delta_mb, total_mb). First rep is a warmup and excluded.

    Memory is measured as RSS (resident set size) so it captures Arrow/Rust/C
    allocations that tracemalloc misses.  delta_mb is the increase during the
    call; total_mb is the absolute peak RSS of the process.
    """
    times = []
    delta_mb = 0.0
    total_mb = 0.0
    for i in range(reps + 1):
        gc.collect()
        rss_before = _proc.memory_info().rss
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        rss_after = _proc.memory_info().rss
        if i > 0:  # skip warmup
            times.append(t1 - t0)
            delta_mb = max(delta_mb, (rss_after - rss_before) / 1024 / 1024)
            total_mb = max(total_mb, rss_after / 1024 / 1024)
    return (sum(times) / len(times)) * 1000, delta_mb, total_mb


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run():
    profiles = [
        ("numeric",       "Numeric (int/float/bool), no nulls"),
        ("numeric_nulls", "Numeric (int/float), ~10% nulls"),
        ("string",        "String, no nulls"),
        ("string_nulls",  "String, ~10% nulls"),
        ("temporal",      "Temporal (datetime/date/duration/time), no nulls"),
        ("temporal_nulls", "Temporal (datetime/date/duration/time), ~10% nulls"),
        ("binary",        "Binary (bytes, 64 B each), no nulls"),
        ("binary_nulls",  "Binary (bytes, 64 B each), ~10% nulls"),
    ]

    for size in SIZES:
        print(f"\n{'='*72}")
        print(f"  {size:,} rows")
        print(f"{'='*72}")

        for profile, label in profiles:
            pol_df = make_polars(size, profile)
            pan_df = make_pandas(pol_df)

            with tempfile.TemporaryDirectory() as tmp:
                pol_path = f"{tmp}/pol.sbdf"
                pan_path = f"{tmp}/pan.sbdf"

                # --- Export ---
                sbdf.export_data(pol_df, pol_path)  # pre-create for import bench
                sbdf.export_data(pan_df, pan_path)

                exp_pan_ms,  exp_pan_dm,  exp_pan_tm  = bench(lambda: sbdf.export_data(pan_df, f"{tmp}/x.sbdf"))
                exp_pol_ms,  exp_pol_dm,  exp_pol_tm  = bench(lambda: sbdf.export_data(pol_df, f"{tmp}/x.sbdf"))
                exp_via_ms,  exp_via_dm,  exp_via_tm  = bench(lambda: sbdf.export_data(pol_df.to_pandas(), f"{tmp}/x.sbdf"))

                # --- Import ---
                imp_pan_ms,     imp_pan_dm,     imp_pan_tm     = bench(lambda: sbdf.import_data(pan_path))
                imp_pol_old_ms, imp_pol_old_dm, imp_pol_old_tm = bench(lambda: pl.from_pandas(sbdf.import_data(pan_path)))
                imp_pol_ms,     imp_pol_dm,     imp_pol_tm     = bench(lambda: sbdf.import_data(pol_path, output_format=sbdf.OutputFormat.POLARS))

            print(f"\n  {label}")
            print(f"  {'':35s}  {'time (ms)':>10}  {'delta (MB)':>11}  {'total RSS (MB)':>14}")
            print(f"  {'-'*76}")
            print(f"  {'Export: pandas df':35s}  {exp_pan_ms:>10.1f}  {exp_pan_dm:>11.1f}  {exp_pan_tm:>14.1f}")
            print(f"  {'Export: polars df (old: via pandas)':35s}  {exp_via_ms:>10.1f}  {exp_via_dm:>11.1f}  {exp_via_tm:>14.1f}")
            print(f"  {'Export: polars df (new: direct)':35s}  {exp_pol_ms:>10.1f}  {exp_pol_dm:>11.1f}  {exp_pol_tm:>14.1f}")
            print(f"  {'Import: -> pandas df':35s}  {imp_pan_ms:>10.1f}  {imp_pan_dm:>11.1f}  {imp_pan_tm:>14.1f}")
            print(f"  {'Import: -> polars df (old: via pandas)':35s}  {imp_pol_old_ms:>10.1f}  {imp_pol_old_dm:>11.1f}  {imp_pol_old_tm:>14.1f}")
            print(f"  {'Import: -> polars df (new: direct)':35s}  {imp_pol_ms:>10.1f}  {imp_pol_dm:>11.1f}  {imp_pol_tm:>14.1f}")
            sys.stdout.flush()


if __name__ == "__main__":
    import sys
    warnings.filterwarnings("ignore", category=sbdf.SBDFWarning)
    print(f"Python {sys.version}")
    print(f"Polars {pl.__version__}  Pandas {pd.__version__}  NumPy {np.__version__}")
    run()
