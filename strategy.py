import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full", layout_file="layouts/strategy.slides.json")

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Regime switching & Dragon Portfolio
    ![alt](public\dragon_portfolio.png)
    """)
    return


if __name__ == "__main__":
    app.run()
