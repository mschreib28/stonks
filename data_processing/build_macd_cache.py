#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ticker_filter import is_non_tradable_ticker


def filter_tradable_tickers_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Filter out non-tradable tickers from LazyFrame."""
    # Collect unique tickers, filter them, then filter the LazyFrame
    df = lf.collect()
    all_tickers = set(df["ticker"].unique().to_list())
    non_tradable = {t for t in all_tickers if is_non_tradable_ticker(t)}
    print(f"Filtering tickers: {len(all_tickers):,} total, {len(non_tradable):,} non-tradable, {len(all_tickers) - len(non_tradable):,} tradable")
    return df.lazy().filter(~pl.col("ticker").is_in(list(non_tradable)))


def scan_raw_minute_aggs(input_glob: str, filter_tickers: bool = True) -> pl.LazyFrame:
    lf = pl.scan_csv(input_glob, has_header=True)

    lf = lf.with_columns(
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

    lf = lf.with_columns(
        [
            pl.from_epoch(pl.col("ws"), time_unit="ns").alias("dt"),
        ]
    ).with_columns(
        [
            pl.col("dt").dt.date().alias("date"),
            pl.col("dt").dt.year().alias("year"),
            pl.col("dt").dt.month().alias("month"),
        ]
    )

    lf = lf.select(["dt", "date", "year", "month", "ticker", "open", "high", "low", "close", "volume"])
    
    # Filter out non-tradable tickers
    if filter_tickers:
        lf = filter_tradable_tickers_lazy(lf)
    
    return lf


def add_macd(lf: pl.LazyFrame, tf_label: str) -> pl.LazyFrame:
    """
    MACD(12,26,9) on 'close' using EMA via ewm_mean(span=..., adjust=False).
    Computed per ticker across time (dt order).
    """
    lf = lf.sort(["ticker", "dt"])

    ema12 = pl.col("close").ewm_mean(span=12, adjust=False).over("ticker")
    ema26 = pl.col("close").ewm_mean(span=26, adjust=False).over("ticker")

    lf = lf.with_columns((ema12 - ema26).alias("macd"))
    lf = lf.with_columns(pl.col("macd").ewm_mean(span=9, adjust=False).over("ticker").alias("signal"))
    lf = lf.with_columns((pl.col("macd") - pl.col("signal")).alias("hist"))

    return lf.with_columns(pl.lit(tf_label).alias("tf"))


def resample_to_5m(lf_1m: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf_1m.group_by_dynamic(
            index_column="dt",
            every="5m",
            period="5m",
            closed="left",
            label="left",
            group_by="ticker",   # <-- renamed from by=
        )
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("date").first().alias("date"),
                pl.col("year").first().alias("year"),
                pl.col("month").first().alias("month"),
            ]
        )
        .sort(["ticker", "dt"])
    )

def macd_day_summary(lf_macd: pl.LazyFrame) -> pl.LazyFrame:
    """
    Collapse bar-level MACD into ticker-day summary features (per timeframe).
    Good for screening.
    """
    lf = lf_macd.sort(["ticker", "dt"]).with_columns(
        pl.col("hist").shift(1).over("ticker").alias("hist_prev")
    ).with_columns(
        [
            ((pl.col("hist_prev") <= 0) & (pl.col("hist") > 0)).cast(pl.Int32).alias("hist_zero_x_up"),
            ((pl.col("hist_prev") >= 0) & (pl.col("hist") < 0)).cast(pl.Int32).alias("hist_zero_x_dn"),
        ]
    )

    return (
        lf.group_by(["ticker", "date", "year", "month", "tf"])
        .agg(
            [
                pl.col("close").last().alias("close_last"),
                pl.col("macd").last().alias("macd_last"),
                pl.col("signal").last().alias("signal_last"),
                pl.col("hist").last().alias("hist_last"),
                pl.col("hist").min().alias("hist_min"),
                pl.col("hist").max().alias("hist_max"),
                pl.col("hist_zero_x_up").sum().alias("hist_zero_cross_up_count"),
                pl.col("hist_zero_x_dn").sum().alias("hist_zero_cross_down_count"),
            ]
        )
        .sort(["ticker", "date", "tf"])
    )


def wide_day_summary(summary: pl.LazyFrame) -> pl.LazyFrame:
    """
    Convert per-(ticker,date,tf) rows into a single (ticker,date) row with:
      *_1m and *_5m columns.
    No pivot needed (works across Polars versions).
    """
    metrics = [
        "close_last",
        "macd_last",
        "signal_last",
        "hist_last",
        "hist_min",
        "hist_max",
        "hist_zero_cross_up_count",
        "hist_zero_cross_down_count",
    ]

    base = summary.select(["ticker", "date"]).unique()

    s1 = (
        summary.filter(pl.col("tf") == "1m")
        .select(["ticker", "date"] + metrics)
        .rename({m: f"{m}__1m" for m in metrics})
    )

    s5 = (
        summary.filter(pl.col("tf") == "5m")
        .select(["ticker", "date"] + metrics)
        .rename({m: f"{m}__5m" for m in metrics})
    )

    out = (
        base.join(s1, on=["ticker", "date"], how="left")
            .join(s5, on=["ticker", "date"], how="left")
            .with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                ]
            )
    )
    return out

def write_parquet_dataset(df: pl.DataFrame, out_path: Path, partitioned: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if partitioned:
        df.write_parquet(out_path, compression="zstd", partition_by=["year", "month"], use_pyarrow=True)
    else:
        df.write_parquet(out_path, compression="zstd")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="data/2025/polygon_minute_aggs", help="Local raw minute files root")
    ap.add_argument("--out-root", default="data/cache/macd_day_features", help="Output cache root")
    ap.add_argument("--partitioned", action="store_true", help="Partition parquet by year/month")
    ap.add_argument("--write-bar-level", action="store_true", help="Also write bar-level MACD parquet (large)")
    args = ap.parse_args()

    input_glob = str(Path(args.input_root) / "**" / "*.csv.gz")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1m bars
    lf_1m = scan_raw_minute_aggs(input_glob)

    # 1m MACD bar-level
    lf_1m_macd = add_macd(lf_1m, "1m")

    # 5m bars + MACD
    lf_5m = resample_to_5m(lf_1m)
    lf_5m_macd = add_macd(lf_5m, "5m")

    # --- Day summaries (recommended cache) ---
    sum_1m = macd_day_summary(lf_1m_macd)
    sum_5m = macd_day_summary(lf_5m_macd)
    summary = pl.concat([sum_1m, sum_5m], how="vertical")

    wide = wide_day_summary(summary)
    wide_df = wide.collect(engine="streaming")
    write_parquet_dataset(wide_df, out_root / "macd_day_features.parquet", partitioned=args.partitioned)
    print(f"Wrote ticker-day MACD feature cache -> {out_root / 'macd_day_features.parquet'}")

    # --- Optional: bar-level caches (big) ---
    if args.write_bar_level:
        df_1m = lf_1m_macd.collect(engine="streaming")
        write_parquet_dataset(df_1m, out_root / "macd_1m_bars.parquet", partitioned=args.partitioned)
        print(f"Wrote 1m bar-level MACD cache -> {out_root / 'macd_1m_bars.parquet'}")

        df_5m = lf_5m_macd.collect(engine="streaming")
        write_parquet_dataset(df_5m, out_root / "macd_5m_bars.parquet", partitioned=args.partitioned)
        print(f"Wrote 5m bar-level MACD cache -> {out_root / 'macd_5m_bars.parquet'}")


if __name__ == "__main__":
    main()
