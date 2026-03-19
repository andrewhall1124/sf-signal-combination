import polars as pl
import sf_quant.data as sfd
import datetime as dt

def get_assets_data(start: dt.date, end: dt.date) -> pl.DataFrame:
    return (
        sfd.load_assets(
            start=start,
            end=end,
            in_universe=True,
            columns=[
                'date',
                'barrid',
                'price',
                'return',
                'specific_return',
                'specific_risk',    
                'predicted_beta',
            ]
        )
        .sort('date', 'barrid')
        .with_columns(
            pl.col('return') / 100,
            pl.col('specific_return') / 100,
            pl.col('specific_risk') / 100,
        )
        .with_columns(
            pl.col('return').shift(-1).over('barrid').alias('forward_return')
        )
    )
