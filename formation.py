import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", layout_file="layouts/formation.slides.json")

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Formation contents

    - Quantitative analysis
    - Technical analysis
    - Financial products characteristics
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ##Quantitative analysis

    According to [investopedia](https://www.investopedia.com/articles/investing/041114/simple-overview-quantitative-analysis.asp), *Quantitative analysis uses mathematical models to analyze financial data and develop algorithm-based trading strategies.*

    Correct, but overly specific.

    ## It's a tool to help you get answers to your questions

    - Should I buy AAPL stock?
    - Should I hedge my US dollar exposure?
    - Should I add a short-term mean-reversion strategy overlay to my long-term trend-following strategy?
    - What is the correlation between my portfolio and the S&P 500 index?
    - etc...
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Example situation

    You have a portfolio with 100% in `SPY` (S&P500 ETF).

    You heard yesterday about volatility targeting in your favorite financial podcast, and you want to know if it is a good idea to implement it for you.

    If seasoned professionnals use it, it surely has merits, right?

    ## Volatility targeting definition

    According to [Quantpedia](https://quantpedia.com/an-introduction-to-volatility-targeting/):

    *"The main aim of the volatility targeting technique is to manage the portfolio’s exposure in such a way that the volatility of a portfolio is as close to the target value as possible.*

    *In other words, to ensure that the amount of dollar risk remains the same."*

    But words are still just words, you want to see by yourself the impact of this technique!

    **How can you do that?**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Know your tools

    Enter Python.

    With it, you can **very** easily obtain the necessary data, perform the necessary calculations, and visualize the results.

    ## Why not use Excel?

    Excel is a great tool, but it has some limitations when it comes to quantitative analysis:

    - It is not designed for large datasets. The hard limit is around 1 millions rows per sheet. You can process **hundreds of millions of rows** with python in a matter of seconds.
    - It can do only a *subset* of what Python can do
    - Once it becomes more complex than a few formulas, it becomes hard to maintain
    - LLMs can interact way more easily with code than with Excel sheets
    """)
    return


@app.cell
def _():
    mo.md(r"""
    # Setup

    Start by installing uv and an IDE of your choice (VScode, Zed, Jetbrains).


    Once done, open a terminal and type:

    ```bash
    uv init <project name>
    ```
    Open said folder with your IDE, then type the following to set up your environnement:

    ```bash
    uv add polars, plotly[express], plotly-stubs, yfinance, marimo, basedpyright, pyochain
    ```

    Once everything downloaded/installed, type

    ```bash
    uv run marimo edit <file_name>.py
    ```

    Your browser will open a notebook. You are ready to go!
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Getting the data

    So, you want to get the data for your portfolio, which is 100% in `SPY`.

    In the first place, we'll download it, save it to disk, then check the first row to confirm everything is good.

    It can be done in a few lines of code.

    - `yfinance` will let us download the data from Yahoo Finance
    - `polars` will let us manipulate said data, save it and read it.
    - `pathlib.Path` simply represent file paths, e.g `users/my_folder/my_file.txt`

    We'll save the data in `parquet` format. Think of it as a **much** better `CSV` file.
    """)
    return


@app.cell
def _():
    import yfinance as yf
    import polars as pl
    from pathlib import Path

    TICKER = "SPY"
    TICKER_RAW = Path(f"{TICKER}_raw.parquet")

    data = yf.download(TICKER, period="max")

    pl.from_pandas(data, include_index=True).write_parquet(TICKER_RAW)

    pl.read_parquet(TICKER_RAW).head(5)
    return Path, TICKER, TICKER_RAW, pl


@app.cell
def _():
    mo.md(r"""
    # Cleaning up data

    Nice, we have our `SPY` prices! But the column names are weird.

    We want to rename them.

    Also, we don't need the hour/minute/second part of the date, so we'll remove it.
    """)
    return


@app.cell
def _(TICKER_RAW, pl):
    def clean_up_raw_data():
        return (
            pl
            .scan_parquet(TICKER_RAW)
            .select(
                pl.col("Date").cast(pl.Date).alias("date"),
                pl.col("('Open', 'SPY')").alias("open"),
                pl.col("('High', 'SPY')").alias("high"),
                pl.col("('Low', 'SPY')").alias("low"),
                pl.col("('Close', 'SPY')").alias("close"),
            )
            .head(5)
            .collect()
        )

    clean_up_raw_data()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It works, great!

    We see that there's a common structure for the name, and we know that we will probably download again from yfinance.

    So, we extract this logic in a reusable function, and save our cleaned-up data in a new file.

    Finally, we plot our full data to visualize it (that's what we want, right?) and re-check that everything is good.
    """)
    return


@app.cell
def _(Path, TICKER, TICKER_RAW, pl):
    import plotly.express as px
    from plotly.graph_objects import Figure

    TICKER_CLEAN = Path(f"{TICKER}.parquet")

    def rename_all(ticker: str) -> str:
        prefix = "('"
        suffix = f"', '{ticker}')"
        return (
            pl.col("Date").cast(pl.Date).name.to_lowercase(),
            pl
            .all()
            .exclude("Date")
            .name.replace(suffix, "", literal=True)
            .name.replace(prefix, "", literal=True)
            .name.to_lowercase(),
        )

    _ = (
        pl
        .scan_parquet(TICKER_RAW)
        .select(rename_all(TICKER))
        .sink_parquet(TICKER_CLEAN)
    )

    px.line(
        pl.read_parquet(TICKER_CLEAN),
        x="date",
        y="close",
        log_y=True,
        template="plotly_dark",
    )
    return TICKER_CLEAN, px


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Getting into it

    Good news, we are done with the **hardest** part now!

    So, let's start computing that volatility targeting strategy.

    ## The maths

    RiskHub defines volatility targeting with the following formula:

    $$w_t = \frac{\sigma_{\text{target}}}{\hat{\sigma}_t}$$

    Where:
    - $w_t$ is the position weigth, our adjustment factor
    - $\sigma_{\text{target}}$ is our target volatility.
    - $\hat{\sigma}_t$  is the current forecast of the asset's volatility.

    We also define $w_c$ as the current position weight, which is the **current**, actual exposure of our portfolio to SPY.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## The Core Idea

    - We want to define our  $\sigma_{\text{target}}$ as a constant annualized volatility of 15%. In other words: *I expect to gain or lose 15% in average in a given year*.
    - The volatility forecast $\hat{\sigma}_t$ will be computed on the last 20 days (1 month). This is a common choice.
    - On a given day, sometimes SPY will be higher, or lower. We wan't to act once per day.
    - This means we'll have to buy or sell shares daily to ensure that $w_c$ is as close as possible to $w_t$

    Basically, we want to ensure that the following equality is true as much as possible :

    $$w_t = w_c$$
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Example

    If we go from an exposure of $1$, i.e "I have 100 dollars in my portfolio, and they are all in SPY right now", then if `SPY` vol is at:

    - **5%** => We want to buy 200 dollars more to amplify our exposure and reach our target of $3$.
    - **15%** => We do nothing, since $current = target$
    - **30%** => We want to sell half of our shares to reduce our exposure and reach our target of $0.5$.

    Note that this imply access to cheap leverage, which is easy to do with SPY (that's not always the case!)
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Forecasting the volatility

    To estimate the volatility, we can simply assume that if today vol is at x, then it's *reasonnable* to expect tomorrow to be more or less in the same ballpark.

    As such, we'll use a 20-day rolling window sample standard deviation of the daily log returns.

    The formula for the window stdev being:

    $s_t = \sqrt{\frac{1}{19}\sum_{i=t-19}^{t}\left(x_i-\bar{x}_t\right)^2}$

    With the arithmetic mean $\bar{x}_t$ as :

    $\bar{x}_t = \frac{1}{20}\sum_{i=t-19}^{t}x_i$


    But first we need to compute our log returns $r_t$ from our raw price data.

    We can define

    $r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$

    Which then let us continue with our next formula: ...
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ... Or we could go to code where obscure maths signs are replaced by clear english words, shall we?
    """)
    return


@app.cell
def _(TICKER_CLEAN, pl, px):
    TARGET = 0.15
    df = (
        pl
        .scan_parquet(TICKER_CLEAN)
        .with_columns(pl.col("close").log().diff().alias("log_return"))
        .with_columns(
            pl
            .col("log_return")
            .rolling_std(window_size=20, min_samples=20)
            .shift(1)
            .mul(16)
            .alias("volatility")
        )
        .with_columns(
            pl.lit(TARGET).truediv(pl.col("volatility")).alias("vol_target")
        )
        .drop_nulls()
        .collect()
    )
    px.line(
        df,
        x="date",
        y=["volatility", "vol_target"],
        template="plotly_dark",
    )
    return


if __name__ == "__main__":
    app.run()
