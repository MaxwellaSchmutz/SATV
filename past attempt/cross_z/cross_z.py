import polars as pl
import datetime as dt
import sf_quant.data as sfd

start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "satv_z"
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

satv_z = (
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
        .shift(21)                 
        .over("barrid")
        .alias("turnover_mean"),

        pl.col("turnover")
        .rolling_std(230)
        .shift(21)                 
        .over("barrid")
        .alias("turnover_std"),
    )
    .with_columns(
        ((pl.col("turnover") - pl.col("turnover_mean"))
         / pl.col("turnover_std"))
        .alias("satv_z")
    )
)
# Filter universe
filtered = satv_z.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col("predicted_beta").is_not_null(),
    pl.col("specific_risk").is_not_null(),
)

# Compute scores
scores = (
    filtered
    #Clip extreme signal values
    .with_columns(
        pl.col(signal_name)
        .clip(-10, 10)
        .alias("signal_clipped")
    )
    #Cross-sectional mean and std
    .with_columns(
        pl.col("signal_clipped").mean().over("date").alias("cs_mean"),
        pl.col("signal_clipped").std().over("date").alias("cs_std"),
    )
    #Handle zero's
    .with_columns(
        pl.when(pl.col("cs_std") > 1e-8)
        .then(
            (pl.col("signal_clipped") - pl.col("cs_mean"))
            / pl.col("cs_std")
        )
        .otherwise(0.0)
        .alias("score")
    )
    .select(
        "date",
        "barrid",
        "predicted_beta",
        "specific_risk",
        "score",
    )
)

# Compute alphas
alphas = (
    scores.with_columns(pl.col("score").mul(IC).mul("specific_risk").alias("alpha"))
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)

alphas.write_parquet(f"{signal_name}_alphas.parquet")