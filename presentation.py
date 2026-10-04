"""Small marimo presentation app for fetching equity data with OpenBB."""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")

with app.setup:
    import yfinance as yf
    import polars as pl
    from pathlib import Path
    import plotly.express as px
    from collections.abc import Sequence
    from pyochain import Seq, Iter

    API_KEY = "fzTOj2HNoH5bunjIGJjVSuUQ0v7qor33"


@app.cell
def _():
    path: Path = Path("VZN.csv")
    lf: pl.LazyFrame = pl.scan_csv(path)
    log_ret: pl.Expr = pl.col("log_return")


    def bootstraped(n_bootstraps: int) -> pl.LazyFrame:

        def _concatenated(df: pl.LazyFrame) -> pl.LazyFrame:
            height = df.select(pl.len()).collect().item()
            return pl.concat([
                df,  # Original data as bootstrap 0
                *df.pipe(
                    lambda returns: (
                        Seq(range(n_bootstraps))
                        .iter()
                        .map(
                            lambda bootstrap_id: returns.with_columns(
                                log_ret.sample(
                                    n=height, with_replacement=True, shuffle=True
                                ),
                                pl.lit(bootstrap_id + 1, pl.UInt32()).alias(
                                    "bootstrap_id"
                                ),
                            ),
                        )
                    )
                ),
            ])

        return (
            lf
            .select("time", "close")
            .with_columns(pl.col("close").log().diff().alias("log_return"))
            .filter(log_ret.is_not_null())
            .with_columns(pl.lit(0, pl.UInt32()).alias("bootstrap_id"))
            .pipe(_concatenated)
            .with_columns(log_ret.cum_sum().over("bootstrap_id").alias("equity"))
        )


    res: pl.DataFrame = bootstraped(1000).collect()
    return bootstraped, lf, log_ret, res


@app.cell
def _(lf: pl.LazyFrame):
    average_returns = (
        lf
        .select("time", "close")
        .select(
            pl
            .col("close")
            .log()
            .diff()
            .rolling_sum(252)
            .mean()
            .alias("log_return")
        )
        .collect()
    )
    return


@app.cell
def _(bootstraped):
    px.line(
        bootstraped(25).collect(),
        x="time",
        y="equity",
        color="bootstrap_id",
        render_mode="webgl",
        title="Bootstraped Equity Curves (25 Bootstraps + Original Data)",
        labels={
            "time": "Time",
            "equity": "Cumulative Log Return",
            "bootstrap_id": "Bootstrap ID",
        },
        height=800,
    )
    return


@app.cell
def _(log_ret: pl.Expr, res: pl.DataFrame):
    yearly: int = 252
    median_ret = pl.col("median annual return")

    stats = (
        res
        .lazy()
        .group_by("bootstrap_id")
        .agg(
            pl.col("equity").last().alias("total_returns"),
            log_ret.rolling_sum(yearly).median().alias("median annual return"),
            log_ret.rolling_std(yearly).mean().mul(16).alias("standard deviation"),
            log_ret.rolling_mean(yearly).skew().alias("skewness"),
        )
        .collect()
    )

    px.histogram(
        stats.select(median_ret),
        title="median annual return distribution, 10'000 bootstraps",
        nbins=50,
    )
    return (stats,)


@app.cell
def _(stats):
    def _quantiles(col_name: str) -> Iter[pl.Expr]:
        return Iter([
            pl.col(col_name).quantile(0.10).alias(f"q10_{col_name}"),
            pl.col(col_name).quantile(0.25).alias(f"q25_{col_name}"),
            pl.col(col_name).median().alias(f"q50_{col_name}"),
            pl.col(col_name).quantile(0.75).alias(f"q75_{col_name}"),
            pl.col(col_name).quantile(0.90).alias(f"q90_{col_name}"),
        ]).map(lambda c: c.round(2))


    quantiles_confidence = (
        stats
        .lazy()
        .select(
            *_quantiles("standard deviation"),
            *_quantiles("skewness"),
            *_quantiles("median annual return"),
        )
        .unpivot()
        .with_columns(pl.col("variable").str.splitn("_", 2).alias("parts"))
        .with_columns(
            pl.col("parts").struct.field("field_0").alias("quantile"),
            pl.col("parts").struct.field("field_1").alias("metric"),
        )
        .drop("variable", "parts")
        .collect()
        .pivot(on="metric", index="quantile", values="value")
    )
    quantiles_confidence
    return


@app.cell
def _():
    fundamentals = Path("db", "fundamentals", "vzn.csv")


    def _get_financial(name: str) -> pl.DataFrame:
        return (
            pl
            .scan_csv(fundamentals)
            .select(
                pl
                .col(name)
                .cast(pl.Float64())
                .median()
                .round(2)
                .alias(f"median value of {name}"),
                pl
                .col(name)
                .cast(pl.Float64())
                .drop_nulls()
                .last()
                .round(2)
                .alias(f"last value of {name}"),
            )
            .collect()
        )


    _get_financial("Gross margin %")
    pl.scan_csv(fundamentals).collect()
    return


if __name__ == "__main__":
    app.run()
