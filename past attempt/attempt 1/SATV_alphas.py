"""Example code for computing alphas and predicted betas prior to running an MVO backtest."""

import polars as pl
import datetime as dt
import sf_quant.data as sfd

start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "satv"
price_filter = 5
IC = 0.05

data = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date",
        "barrid",
        "price",
        "return",
        "specific_risk",
        "predicted_beta",
        "market_cap",
        "daily_volume",
    ],
    in_universe=True,
).with_columns(pl.col("return", "specific_risk").truediv(100))

# # =====================================================
# # 1. Momentum
# # =====================================================
# signals = data.sort("date", "barrid").with_columns(
#     pl.col("return")
#     .log1p()
#     .rolling_sum(230)
#     .shift(21)
#     .over("barrid")
#     .alias("momentum")
# )

# =====================================================
# 2. SATV 
# =====================================================
SATV = (
    data
    .sort("date", "barrid")
    .with_columns(
        (pl.col("market_cap") / pl.col("price")).alias("shrout"),
        (pl.col("daily_volume") /
         (pl.col("market_cap") / pl.col("price"))).alias("turnover"),
    )
    .with_columns(
        pl.col("turnover")
        .rolling_mean(230)
        .over("barrid")
        .alias("turnover_mean"),

        pl.col("turnover")
        .rolling_std(230)
        .over("barrid")
        .alias("turnover_std"),
    )
    .with_columns(
        ((pl.col("turnover") - pl.col("turnover_mean"))
         / pl.col("turnover_std"))
        .alias("SATV")
    )
)

# =====================================================
# 3. Filter universe (unchanged logic, extended vars)
# =====================================================
filtered = SATV.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col("SATV").is_not_null(),
    pl.col("predicted_beta").is_not_null(),
    pl.col("specific_risk").is_not_null(),
)

# =====================================================
# 4. Compute scores (ONLY change is interaction)
# =====================================================
scores = filtered.select(
    "date",
    "barrid",
    "predicted_beta",
    "specific_risk",

    # SATV z-score (this is now the signal)
    pl.col("SATV")
    .sub(pl.col("SATV").mean())
    .truediv(pl.col("SATV").std())
    .over("date")
    .alias("score"),
)

# =====================================================
# 5. Compute alphas (EXACTLY instructor logic)
# =====================================================
alphas = (
    scores
    .with_columns(pl.col("score").mul(IC).mul("specific_risk").alias("alpha"))
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)

alphas.write_parquet(f"{signal_name}_alphas.parquet")
print(alphas.sort("date"))