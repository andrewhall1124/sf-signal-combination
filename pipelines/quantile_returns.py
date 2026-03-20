"""Script for computing quantile returns for each signal"""
from signals.quantile_returns import compute_quantile_returns
from signals.expr import momentum, reversal, bab
import datetime as dt
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
from experiments.utils import save_returns_plot

start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)

for signal_expr in [momentum, reversal, bab]:
    signal_name = signal_expr.__name__

    returns = compute_quantile_returns(
        signal_expr=signal_expr(),
        start=start,
        end=end
    )

    cumulative_returns = (
        returns
        .sort('date')
        .with_columns(
            pl.col('return')
            .log1p()
            .cum_sum()
            .mul(100)
        )
    )

    # Save plot
    file_path = Path(f"results/quantile_returns/{signal_name}.png")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    title = {
        'momentum': 'Momentum',
        'reversal': 'Reversal',
        'bab': 'BAB'
    }[signal_name]

    save_returns_plot(
        returns=returns,
        file_path=file_path,
        title=title
    )

    file_path = Path(f"data/quantile_returns/{signal_name}.parquet")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    returns.write_parquet(file_path)