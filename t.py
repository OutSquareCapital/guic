from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import pandas as pd


def get_data(ticker: str) -> None:
    import yfinance as yf

    pd_df: pd.DataFrame = yf.download(ticker, period="max", interval="1d")
    return (
        pl
        .from_pandas(pd_df, include_index=True)
        .lazy()
        .select(pl.lit(ticker).alias("ticker"), format_name(ticker))
        .cast(ticker_schema())
        .sink_parquet(f"{ticker}.parquet")
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


if __name__ == "__main__":
    file = "dbc.parquet"
    pl.read_parquet(file).show()
