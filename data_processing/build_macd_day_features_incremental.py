#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ticker_filter import is_non_tradable_ticker


# ---------- MACD helpers ----------

def add_macd(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add MACD(12,26,9) on 'close', computed per ticker across time order.
    Assumes df has columns: ticker, dt, close.
    """
    df = df.sort(["ticker", "dt"])

    # EMA via ewm_mean; per-ticker with over()
    df = df.with_columns(
        (pl.col("close").ewm_mean(span=12, adjust=False).over("ticker") -
         pl.col("close").ewm_mean(span=26, adjust=False).over("ticker")).alias("macd")
    ).with_columns(
        pl.col("macd").ewm_mean(span=9, adjust=False).over("ticker").alias("signal")
    ).with_columns(
        (pl.col("macd") - pl.col("signal")).alias("hist")
    )
    return df


def resample_5m(df_1m: pl.DataFrame) -> pl.DataFrame:
    """
    Resample 1m bars to 5m bars per ticker. Keeps dt as window start and adds date.
    """
    out = (
        df_1m.sort(["ticker", "dt"])
        .group_by_dynamic("dt", every="5m", period="5m", closed="left", label="left", group_by="ticker")
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .sort(["ticker", "dt"])
    )

    # Add date derived from dt (needed by day_summary)
    out = out.with_columns(pl.col("dt").dt.date().alias("date"))
    return out



def day_summary(df_macd: pl.DataFrame, tf: str) -> pl.DataFrame:
    """
    Collapse bar-level MACD into ticker-day summary for a single timeframe.
    """
    df_macd = df_macd.sort(["ticker", "dt"]).with_columns(
        pl.col("hist").shift(1).over("ticker").alias("hist_prev")
    ).with_columns(
        [
            ((pl.col("hist_prev") <= 0) & (pl.col("hist") > 0)).cast(pl.Int32).alias("xup"),
            ((pl.col("hist_prev") >= 0) & (pl.col("hist") < 0)).cast(pl.Int32).alias("xdn"),
        ]
    )

    out = (
        df_macd.group_by(["ticker", "date"])
        .agg(
            [
                pl.col("close").last().alias(f"close_last__{tf}"),
                pl.col("macd").last().alias(f"macd_last__{tf}"),
                pl.col("signal").last().alias(f"signal_last__{tf}"),
                pl.col("hist").last().alias(f"hist_last__{tf}"),
                pl.col("hist").min().alias(f"hist_min__{tf}"),
                pl.col("hist").max().alias(f"hist_max__{tf}"),
                pl.col("xup").sum().alias(f"hist_zero_cross_up_count__{tf}"),
                pl.col("xdn").sum().alias(f"hist_zero_cross_down_count__{tf}"),
            ]
        )
    )
    return out


def merge_timeframes(sum_1m: pl.DataFrame, sum_5m: pl.DataFrame) -> pl.DataFrame:
    """
    Join 1m + 5m summaries into a single ticker-day rowset.
    """
    return sum_1m.join(sum_5m, on=["ticker", "date"], how="outer")


# ---------- Session filters (optional) ----------

def filter_rth_et(df: pl.DataFrame) -> pl.DataFrame:
    """
    Keep 09:30-16:00 ET inclusive of pre/post? No: this is RTH-only.
    We convert dt->US/Eastern for filtering, then drop tz.
    """
    dt_et = pl.col("dt").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
    return (
        df.with_columns(dt_et.alias("dt_et"))
          .filter(
              (pl.col("dt_et").dt.hour() > 9) |
              ((pl.col("dt_et").dt.hour() == 9) & (pl.col("dt_et").dt.minute() >= 30))
          )
          .filter(
              (pl.col("dt_et").dt.hour() < 16) |
              ((pl.col("dt_et").dt.hour() == 16) & (pl.col("dt_et").dt.minute() == 0))
          )
          .drop("dt_et")
    )


# ---------- IO ----------

def read_one_day_csv(path: Path, filter_tickers: bool = True) -> pl.DataFrame:
    """
    Reads one daily minute agg file. Your window_start is epoch nanoseconds.
    Filters out non-tradable tickers by default.
    """
    df = pl.read_csv(path)

    df = df.with_columns(
        [
            pl.col("ticker").cast(pl.Utf8),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
            pl.col("window_start").cast(pl.Int64, strict=False).alias("ws"),
        ]
    ).filter(pl.col("ws").is_not_null())

    df = df.with_columns(
        pl.from_epoch(pl.col("ws"), time_unit="ns").alias("dt")
    ).with_columns(
        pl.col("dt").dt.date().alias("date")
    )

    df = df.select(["ticker", "dt", "date", "open", "high", "low", "close", "volume"])
    
    # Filter out non-tradable tickers
    if filter_tickers:
        all_tickers = set(df["ticker"].unique().to_list())
        non_tradable = {t for t in all_tickers if is_non_tradable_ticker(t)}
        if non_tradable:
            df = df.filter(~pl.col("ticker").is_in(list(non_tradable)))
    
    return df


def write_day_parquet(df: pl.DataFrame, out_dir: Path, date_str: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"date={date_str}.parquet"
    df.write_parquet(out_path, compression="zstd")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="data/2025/polygon_minute_aggs")
    ap.add_argument("--out-root", default="data/2025/cache/macd_day_features_inc")
    ap.add_argument("--mode", choices=["all", "rth"], default="all",
                   help="Compute features using all sessions or RTH-only (ET).")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    out_root = Path(args.out_root) / f"mode={args.mode}"
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.rglob("*.csv.gz"))
    if not files:
        raise SystemExit(f"No csv.gz files found under {input_root}")

    for f in files:
        # Derive date string from filename: 2025-01-02.csv.gz -> 2025-01-02
        date_str = f.name.replace(".csv.gz", "")

        out_path = out_root / f"date={date_str}.parquet"
        if out_path.exists():
            # already built; skip
            continue

        df = read_one_day_csv(f)

        if args.mode == "rth":
            df = filter_rth_et(df)

        if df.height == 0:
            print(f"Skipping {f.name} (no rows after filtering)")
            continue
        
        # 1m MACD summary
        df_1m = add_macd(df)
        s1 = day_summary(df_1m, "1m")

        # 5m MACD summary
        df_5m = resample_5m(df)
        df_5m = add_macd(df_5m)
        s5 = day_summary(df_5m, "5m")

        out = merge_timeframes(s1, s5)

        # Add partition helpers
        out = out.with_columns(
            [
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month"),
            ]
        )

        write_day_parquet(out, out_root, date_str)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
