import polars as pl

def momentum() -> pl.Expr:
    return (
        pl.col('return')
        .log1p()
        .rolling_sum(230)
        .shift(21)
        .over('barrid')
    )

def reversal() -> pl.Expr:
    return (
        pl.col('return')
        .log1p()
        .rolling_sum(21)
        .mul(-1)
        .over('barrid')
    )

def bab() -> pl.Expr:
    return (
        pl.col('predicted_beta')
        .mul(-1)
    )