from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import plotly.express as px
import polars as pl
from pyochain import Iter

if TYPE_CHECKING:
    import pandas as pd
DB = Path("db")
TICKERS = ("SPY", "DBC", "GLD", "TLT")


def get_data(ticker: str, *, auto_adjust: bool = True) -> None:
    import yfinance as yf

    file = DB.joinpath(f"{ticker}").with_suffix(".parquet")

    pd_df: pd.DataFrame = yf.download(
        ticker, period="max", interval="1d", auto_adjust=auto_adjust
    )
    return (
        pl
        .from_pandas(pd_df, include_index=True)
        .lazy()
        .select(pl.lit(ticker).alias("ticker"), format_name(ticker))
        .cast(ticker_schema())
        .sink_parquet(file)
    )


def ticker_schema() -> pl.Schema:
    return pl.Schema({
        "ticker": pl.String,
        "date": pl.Date,
        "open": pl.Float32,
        "high": pl.Float32,
        "low": pl.Float32,
        "close": pl.Float32,
        "volume": pl.UInt32,
    })


def format_name(ticker: str) -> pl.Expr:
    return (
        pl
        .all()
        .exclude("ticker")
        .name.replace("('", "", literal=True)
        .name.replace(f"', '{ticker}')", "", literal=True)
        .name.to_lowercase()
    )


def price_to_log_price(expr: pl.Expr) -> pl.Expr:
    return expr.log().sub(expr.log().first()).add(1)


def price_to_log_return(expr: pl.Expr) -> pl.Expr:
    return expr.log().diff()


def corr_exprs(tickers: Iterable[str]) -> Iter[pl.Expr]:
    return (
        Iter(tickers)
        .combinations(2)
        .map_star(
            lambda left, right: pl.rolling_corr(
                pl.col(left), pl.col(right), window_size=250
            ).alias(f"{left}_{right}")
        )
    )


def corr_matrix(df: pl.DataFrame) -> pl.DataFrame:
    pairs = (
        df
        .pivot(index="date", on="ticker", values="log_returns")
        .lazy()
        .sort("date")
        .select("date", *corr_exprs(TICKERS))
        .unpivot(index="date", value_name="corr", variable_name="ticker")
        .drop_nulls("corr")
        .select(pl.col("corr").mean().over("ticker"), "ticker")
        .unique("ticker")
        .select(
            pl.col("ticker").str.split("_").list.get(0).alias("left"),
            pl.col("ticker").str.split("_").list.get(1).alias("right"),
            "corr",
        )
    )
    return (
        pl
        .concat([
            pairs,
            pairs.select(
                pl.col("right").alias("left"), pl.col("left").alias("right"), "corr"
            ),
        ])
        .collect()
        .pivot(index="left", on="right", values="corr", sort_columns=True)
        .lazy()
        .sort("left")
        .drop("left")
        .collect()
    )


def main() -> pl.LazyFrame:
    close = pl.col("close")

    return pl.scan_parquet(DB).with_columns(
        close.pct_change().alias("pct_change"),
        close.pipe(price_to_log_price).alias("log_price"),
        close.pipe(price_to_log_return).alias("log_returns"),
    )


def plot_corr_matrix(mat: pl.DataFrame) -> None:
    px.imshow(
        mat,
        y=mat.columns,
        text_auto=".2f",
        color_continuous_scale=["red", "orange", "yellow", "green", "lime"],
        template="plotly_dark",
    ).show()


if __name__ == "__main__":
    df = main().collect()
    # px.line(df, x="date", y="log_price", color="ticker", template="plotly_dark").show()
    px.violin(
        df.select("log_returns", "ticker"), color="ticker", template="plotly_dark"
    ).show()
    # plot_corr_matrix(corr_matrix(df))
