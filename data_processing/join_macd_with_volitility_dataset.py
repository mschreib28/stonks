import polars as pl

vol = pl.scan_parquet("data/2025/cache/minute_features.parquet")
macd = pl.scan_parquet("data/2025/cache/macd_day_features_inc/mode=all/*.parquet")

enriched = (
    vol.join(macd, on=["ticker","date"], how="left")
       .collect()
)

enriched.write_parquet(
    "data/2025/cache/minute_features_plus_macd.parquet",
    compression="zstd"
)